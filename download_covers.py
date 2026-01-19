"""
Open Library Cover Downloader
==============================
Downloads book covers from Open Library Covers API using ISBNs.

Respects rate limits: 100 requests per 5 minutes (1 request per 3 seconds)

Usage:
    python download_covers.py
    python download_covers.py --input top_100k_books.csv --limit 1000
    python download_covers.py --size L --delay 5

Author: Data Engineering Team
"""

import argparse
import csv
import json
import os
import ssl
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus


def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


load_env()

# SSL context that doesn't verify certificates (for Open Library)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


# =============================================================================
# CONFIGURATION
# =============================================================================

ISBN_URL = "https://covers.openlibrary.org/b/isbn"
COVER_ID_URL = "https://covers.openlibrary.org/b/id"
COVER_SIZE = "-L"  # -S (small), -M (medium), -L (large)
MIN_GOOD_COVER_SIZE = 10000  # 10KB - covers smaller than this are likely placeholders
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"
GOOGLE_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")
DEFAULT_DELAY = 3.0  # Seconds between requests (100 req / 5 min = 3 sec)
DEFAULT_INPUT = Path("./output/top_100k_books.csv")
DEFAULT_OUTPUT_DIR = Path("./output/covers")
TIMEOUT = 30  # Request timeout in seconds
MIN_FILE_SIZE = 1000  # Bytes - blank placeholder images are smaller


# =============================================================================
# DOWNLOADER
# =============================================================================

def fetch_image(url: str) -> Optional[bytes]:
    """
    Fetch image data from URL.
    Returns: bytes if successful, None if not found or error
    """
    try:
        request = Request(url, headers={
            'User-Agent': 'OpenLibraryCoverDownloader/1.0 (Educational Project)'
        })

        with urlopen(request, timeout=TIMEOUT, context=SSL_CONTEXT) as response:
            data = response.read()
            if len(data) >= MIN_FILE_SIZE:
                return data
            return None

    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except URLError:
        return None
    except Exception:
        return None


def fetch_google_cover(isbn: str) -> Optional[bytes]:
    """
    Fetch cover from Google Books API using ISBN.
    Returns: bytes if successful, None if not found or error
    """
    try:
        url = f"{GOOGLE_BOOKS_API}?q=isbn:{isbn}&key={GOOGLE_API_KEY}"
        request = Request(url, headers={
            'User-Agent': 'OpenLibraryCoverDownloader/1.0 (Educational Project)'
        })

        with urlopen(request, timeout=TIMEOUT, context=SSL_CONTEXT) as response:
            data = json.loads(response.read().decode('utf-8'))

            if data.get('totalItems', 0) == 0:
                return None

            items = data.get('items', [])
            if not items:
                return None

            volume_info = items[0].get('volumeInfo', {})
            image_links = volume_info.get('imageLinks', {})

            # Try larger images first
            for size in ['extraLarge', 'large', 'medium', 'thumbnail', 'smallThumbnail']:
                image_url = image_links.get(size)
                if image_url:
                    # Remove zoom parameter for better quality
                    image_url = image_url.replace('&edge=curl', '')
                    if 'zoom=' in image_url:
                        image_url = image_url.split('zoom=')[0] + 'zoom=1'

                    img_request = Request(image_url, headers={
                        'User-Agent': 'OpenLibraryCoverDownloader/1.0'
                    })
                    with urlopen(img_request, timeout=TIMEOUT, context=SSL_CONTEXT) as img_response:
                        img_data = img_response.read()
                        if len(img_data) >= MIN_FILE_SIZE:
                            return img_data

            return None

    except Exception:
        return None


def get_existing_covers(output_dir: Path) -> Set[str]:
    """
    Get set of identifiers that already have covers downloaded.
    Returns set of filenames without extension.
    """
    existing = set()
    if output_dir.exists():
        for f in output_dir.glob("*.jpg"):
            if f.stat().st_size >= MIN_FILE_SIZE:
                existing.add(f.stem)
    return existing


def download_cover(isbns: List[str], cover_ids: List[int],
                   output_dir: Path, work_id: str,
                   existing_covers: Optional[Set[str]] = None) -> tuple[Optional[Path], str, int]:
    """
    Download the best cover image from multiple sources.
    Tries multiple ISBNs and cover_ids, picks the largest (best quality) one.
    Order: Open Library Cover IDs -> Open Library ISBNs -> Google Books
    Returns: (Path to saved file or None, method used, attempts made)
    """
    # Check if any of our identifiers already have a good cover
    if existing_covers:
        for isbn in isbns:
            if isbn in existing_covers:
                cover_path = output_dir / f"{isbn}.jpg"
                if cover_path.exists() and cover_path.stat().st_size >= MIN_GOOD_COVER_SIZE:
                    return cover_path, 'cached', 0
        for cover_id in cover_ids:
            if str(cover_id) in existing_covers:
                cover_path = output_dir / f"{cover_id}.jpg"
                if cover_path.exists() and cover_path.stat().st_size >= MIN_GOOD_COVER_SIZE:
                    return cover_path, 'cached', 0

    filename = isbns[0] if isbns else (str(cover_ids[0]) if cover_ids else work_id)
    output_path = output_dir / f"{filename}.jpg"

    if output_path.exists() and output_path.stat().st_size >= MIN_GOOD_COVER_SIZE:
        return output_path, 'cached', 0

    attempts = 0
    best_data = None
    best_size = 0
    best_method = 'none'
    best_id = None

    # Try ISBNs first (prioritized - usually newer editions with better covers)
    for isbn in isbns[:5]:
        attempts += 1
        url = f"{ISBN_URL}/{isbn}{COVER_SIZE}.jpg?default=false"
        data = fetch_image(url)
        if data and len(data) > best_size:
            best_data = data
            best_size = len(data)
            best_method = f'ol_isbn#{attempts}'
            best_id = isbn
            # If we found a good cover (>50KB), use it
            if best_size >= 50000:
                break

    # Fallback to Open Library Cover IDs
    if best_size < MIN_GOOD_COVER_SIZE:
        for cover_id in cover_ids[:5]:
            attempts += 1
            url = f"{COVER_ID_URL}/{cover_id}{COVER_SIZE}.jpg?default=false"
            data = fetch_image(url)
            if data and len(data) > best_size:
                best_data = data
                best_size = len(data)
                best_method = f'ol_cover#{attempts}'
                best_id = str(cover_id)
                if best_size >= 50000:
                    break

    # Final fallback: Google Books API
    if best_size < MIN_GOOD_COVER_SIZE:
        for isbn in isbns[:3]:
            attempts += 1
            data = fetch_google_cover(isbn)
            if data and len(data) > best_size:
                best_data = data
                best_size = len(data)
                best_method = f'google#{attempts}'
                best_id = isbn
                if best_size >= 50000:
                    break

    # Save the best cover we found
    if best_data and best_size >= MIN_FILE_SIZE:
        save_path = output_dir / f"{best_id}.jpg"
        with open(save_path, 'wb') as f:
            f.write(best_data)
        return save_path, best_method, attempts

    return None, 'none', attempts


def download_covers(input_path: Path, output_dir: Path,
                    delay: float, limit: int = 0, start_rank: int = 1):
    """
    Download covers for all books in the input CSV.
    Tries: Open Library ISBN -> Open Library Cover ID -> Google Books ISBN
    """
    print("=" * 70)
    print("BOOK COVER DOWNLOADER (Open Library + Google Books)")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input:      {input_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Delay:      {delay}s between requests")
    print(f"Strategy:   OL ISBN -> OL Cover ID -> Google Books ISBN")
    if limit:
        print(f"Limit:      {limit:,} covers")
    if start_rank > 1:
        print(f"Start Rank: {start_rank:,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing covers for fast skip
    existing_covers = get_existing_covers(output_dir)
    print(f"Existing:   {len(existing_covers):,} covers already downloaded")

    start_time = datetime.now()
    processed = 0
    by_ol_isbn = 0
    by_ol_cover = 0
    by_google = 0
    cached = 0
    failed = 0
    skipped = 0
    total_attempts = 0

    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            rank = int(row.get('rank', 0))

            if rank < start_rank:
                continue

            processed += 1

            if limit and processed > limit:
                break

            # Parse pipe-separated ISBNs and cover_ids
            isbns_str = row.get('isbns', '').strip()
            cover_ids_str = row.get('cover_ids', '').strip()
            work_id = row.get('work_id', 'unknown')
            title = row.get('title', 'Unknown')[:50]

            isbns = [i.strip() for i in isbns_str.split('|') if i.strip()] if isbns_str else []
            cover_ids = [int(c.strip()) for c in cover_ids_str.split('|') if c.strip()] if cover_ids_str else []

            if not isbns and not cover_ids:
                skipped += 1
                continue

            try:
                result_path, method, attempts = download_cover(
                    isbns, cover_ids, output_dir, work_id, existing_covers
                )
                total_attempts += attempts

                if method == 'cached':
                    cached += 1
                    if cached <= 5 or cached % 50 == 0:
                        print(f"  [{processed:>6}] #{rank} - Cached (skipping...)")
                elif method.startswith('ol_isbn'):
                    by_ol_isbn += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    if by_ol_isbn % 10 == 0 or by_ol_isbn <= 10:
                        print(f"  [{processed:>6}] #{rank} - {method} ({size_kb:.0f}KB) - {title}")
                elif method.startswith('ol_cover'):
                    by_ol_cover += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    print(f"  [{processed:>6}] #{rank} - {method} ({size_kb:.0f}KB) - {title}")
                elif method.startswith('google'):
                    by_google += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    print(f"  [{processed:>6}] #{rank} - {method} ({size_kb:.0f}KB) - {title}")
                else:
                    failed += 1
                    if failed <= 30:
                        print(f"  [{processed:>6}] #{rank} - NOT FOUND after {attempts} attempts")

            except HTTPError as e:
                if e.code == 403:
                    print(f"\n  RATE LIMITED! Waiting 60 seconds...")
                    time.sleep(60)
                    try:
                        result_path, method, attempts = download_cover(
                            isbns, cover_ids, output_dir, work_id, existing_covers
                        )
                        total_attempts += attempts
                        if method.startswith('ol_isbn'):
                            by_ol_isbn += 1
                        elif method.startswith('ol_cover'):
                            by_ol_cover += 1
                        elif method.startswith('google'):
                            by_google += 1
                        else:
                            failed += 1
                    except:
                        failed += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1

            if processed % 100 == 0:
                elapsed = datetime.now() - start_time
                total_downloaded = by_ol_isbn + by_ol_cover + by_google
                rate = total_downloaded / max(elapsed.total_seconds(), 1) * 3600
                avg_attempts = total_attempts / max(total_downloaded, 1)
                print(f"\n  Progress: {processed:,} | OL ISBN: {by_ol_isbn:,} | OL Cover: {by_ol_cover:,} | "
                      f"Google: {by_google:,} | Failed: {failed:,} | Cached: {cached:,} | {rate:.0f}/hr\n")

            # Only delay if we actually made API requests (not cached)
            if method != 'cached':
                time.sleep(delay)

    elapsed = datetime.now() - start_time
    total_downloaded = by_ol_isbn + by_ol_cover + by_google
    avg_attempts = total_attempts / max(total_downloaded + failed, 1)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time:          {elapsed}")
    print(f"Processed:             {processed:,}")
    print(f"Downloaded (OL ISBN):  {by_ol_isbn:,}")
    print(f"Downloaded (OL Cover): {by_ol_cover:,}")
    print(f"Downloaded (Google):   {by_google:,}")
    print(f"Already Cached:        {cached:,}")
    print(f"Not Found:             {failed:,}")
    print(f"Skipped (no ID):       {skipped:,}")
    print(f"Total API Requests:    {total_attempts:,}")
    print(f"Avg Attempts/Success:  {avg_attempts:.1f}")

    success_rate = (total_downloaded + cached) / max(processed - skipped, 1) * 100
    print(f"Success Rate:          {success_rate:.1f}%")

    if total_downloaded + cached > 0:
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.jpg"))
        print(f"Total Size:            {total_size / (1024*1024):.1f} MB")
        print(f"Avg Size:              {total_size / max(total_downloaded + cached, 1) / 1024:.1f} KB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download book covers from Open Library Covers API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_covers.py                          # Download all from top_100k
  python download_covers.py --limit 1000             # Download first 1000
  python download_covers.py --delay 5                # Slower rate
  python download_covers.py --start-rank 5000        # Resume from rank 5000
        """
    )

    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Input CSV with isbn column (default: {DEFAULT_INPUT})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for covers (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between requests in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of covers to download (0=all)')
    parser.add_argument('--start-rank', type=int, default=1,
                        help='Start from this rank (for resuming)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run extract_top_books.py first to generate the input file.")
        sys.exit(1)

    download_covers(
        input_path=args.input,
        output_dir=args.output_dir,
        delay=args.delay,
        limit=args.limit,
        start_rank=args.start_rank
    )


if __name__ == "__main__":
    main()
