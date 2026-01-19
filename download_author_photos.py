"""
Author Photo Downloader
=======================
Downloads author photos from Open Library using photo IDs and author OLIDs.

Respects rate limits: 100 requests per 5 minutes (1 request per 3 seconds)

Usage:
    python download_author_photos.py
    python download_author_photos.py --input output/master_authors.csv --limit 1000
    python download_author_photos.py --delay 5

Author: Data Engineering Team
"""

import argparse
import csv
import json
import ssl
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# SSL context that doesn't verify certificates (for Open Library)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


# =============================================================================
# CONFIGURATION
# =============================================================================

PHOTO_ID_URL = "https://covers.openlibrary.org/a/id"
AUTHOR_OLID_URL = "https://covers.openlibrary.org/a/olid"
PHOTO_SIZE = "-L"  # -S (small), -M (medium), -L (large)
DEFAULT_DELAY = 3.0  # Seconds between requests
DEFAULT_INPUT = Path("./output/master_authors.csv")
DEFAULT_OUTPUT_DIR = Path("./output/author_photos")
TIMEOUT = 30  # Request timeout in seconds
MIN_FILE_SIZE = 1000  # Bytes - blank placeholder images are smaller
MIN_GOOD_PHOTO_SIZE = 5000  # 5KB - photos smaller than this are likely placeholders


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
            'User-Agent': 'OpenLibraryAuthorPhotoDownloader/1.0 (Educational Project)'
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


def get_existing_photos(output_dir: Path) -> Set[str]:
    """
    Get set of identifiers that already have photos downloaded.
    Returns set of filenames without extension.
    """
    existing = set()
    if output_dir.exists():
        for f in output_dir.glob("*.jpg"):
            if f.stat().st_size >= MIN_FILE_SIZE:
                existing.add(f.stem)
    return existing


def fetch_wikimedia_image(image_name: str) -> Optional[bytes]:
    """
    Fetch image from Wikimedia Commons.
    Image name should be the filename from Wikipedia/Wikidata.
    """
    if not image_name:
        return None

    try:
        # Clean up image name
        image_name = image_name.strip()
        if image_name.startswith('File:'):
            image_name = image_name[5:]

        # Wikimedia Commons thumbnail URL
        # Using 300px width for reasonable quality
        from urllib.parse import quote
        encoded_name = quote(image_name.replace(' ', '_'))

        # Try the Wikimedia API to get image URL
        api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles=File:{encoded_name}&prop=imageinfo&iiprop=url&iiurlwidth=300&format=json"

        request = Request(api_url, headers={
            'User-Agent': 'OpenLibraryAuthorPhotoDownloader/1.0 (Educational Project)'
        })

        with urlopen(request, timeout=TIMEOUT, context=SSL_CONTEXT) as response:
            data = json.loads(response.read().decode('utf-8'))

            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    continue
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo:
                    thumb_url = imageinfo[0].get('thumburl') or imageinfo[0].get('url')
                    if thumb_url:
                        img_request = Request(thumb_url, headers={
                            'User-Agent': 'OpenLibraryAuthorPhotoDownloader/1.0'
                        })
                        with urlopen(img_request, timeout=TIMEOUT, context=SSL_CONTEXT) as img_response:
                            img_data = img_response.read()
                            if len(img_data) >= MIN_FILE_SIZE:
                                return img_data
        return None

    except Exception:
        return None


def download_photo(author_id: str, photo_id: Optional[int],
                   wikipedia_image: Optional[str],
                   output_dir: Path,
                   existing_photos: Optional[Set[str]] = None) -> tuple[Optional[Path], str, int]:
    """
    Download author photo with multiple fallbacks.
    Order: Photo ID -> Author OLID -> Wikipedia/Wikimedia image
    Returns: (Path to saved file or None, method used, attempts made)
    """
    # Check if already have a good photo
    if existing_photos:
        if photo_id and str(photo_id) in existing_photos:
            photo_path = output_dir / f"{photo_id}.jpg"
            if photo_path.exists() and photo_path.stat().st_size >= MIN_GOOD_PHOTO_SIZE:
                return photo_path, 'cached', 0
        if author_id in existing_photos:
            photo_path = output_dir / f"{author_id}.jpg"
            if photo_path.exists() and photo_path.stat().st_size >= MIN_GOOD_PHOTO_SIZE:
                return photo_path, 'cached', 0

    attempts = 0
    best_data = None
    best_size = 0
    best_method = 'none'
    best_id = None

    # Try photo ID first (usually better quality)
    if photo_id:
        attempts += 1
        url = f"{PHOTO_ID_URL}/{photo_id}{PHOTO_SIZE}.jpg"
        data = fetch_image(url)
        if data and len(data) > best_size:
            best_data = data
            best_size = len(data)
            best_method = f'photo_id#{attempts}'
            best_id = str(photo_id)
            # If good quality, use it
            if best_size >= MIN_GOOD_PHOTO_SIZE:
                save_path = output_dir / f"{best_id}.jpg"
                with open(save_path, 'wb') as f:
                    f.write(best_data)
                return save_path, best_method, attempts

    # Fallback to author OLID
    attempts += 1
    url = f"{AUTHOR_OLID_URL}/{author_id}{PHOTO_SIZE}.jpg"
    data = fetch_image(url)
    if data and len(data) > best_size:
        best_data = data
        best_size = len(data)
        best_method = f'olid#{attempts}'
        best_id = author_id
        if best_size >= MIN_GOOD_PHOTO_SIZE:
            save_path = output_dir / f"{best_id}.jpg"
            with open(save_path, 'wb') as f:
                f.write(best_data)
            return save_path, best_method, attempts

    # Final fallback: Wikipedia/Wikimedia image
    if wikipedia_image and best_size < MIN_GOOD_PHOTO_SIZE:
        attempts += 1
        data = fetch_wikimedia_image(wikipedia_image)
        if data and len(data) > best_size:
            best_data = data
            best_size = len(data)
            best_method = f'wikipedia#{attempts}'
            best_id = author_id

    # Save the best photo we found
    if best_data and best_size >= MIN_FILE_SIZE:
        save_path = output_dir / f"{best_id}.jpg"
        with open(save_path, 'wb') as f:
            f.write(best_data)
        return save_path, best_method, attempts

    return None, 'none', attempts


def download_photos(input_path: Path, output_dir: Path,
                    delay: float, limit: int = 0, start_row: int = 1):
    """
    Download photos for all authors in the input CSV.
    Uses Photo ID -> OLID -> Wikipedia fallback strategy.
    """
    print("=" * 70)
    print("AUTHOR PHOTO DOWNLOADER")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input:      {input_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Delay:      {delay}s between requests")
    print(f"Strategy:   Photo ID -> Author OLID -> Wikipedia")
    if limit:
        print(f"Limit:      {limit:,} photos")
    if start_row > 1:
        print(f"Start Row:  {start_row:,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing photos for fast skip
    existing_photos = get_existing_photos(output_dir)
    print(f"Existing:   {len(existing_photos):,} photos already downloaded")

    start_time = datetime.now()
    processed = 0
    row_num = 0
    by_photo_id = 0
    by_olid = 0
    by_wikipedia = 0
    cached = 0
    failed = 0
    skipped = 0
    total_attempts = 0

    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_num += 1

            if row_num < start_row:
                continue

            processed += 1

            if limit and processed > limit:
                break

            author_id = row.get('author_id', '').strip()
            photo_id_str = row.get('photo_id', '').strip()
            wikipedia_image = row.get('wikipedia_image', '').strip() or None
            name = row.get('name', 'Unknown')[:40]

            photo_id = int(photo_id_str) if photo_id_str and photo_id_str.isdigit() else None

            if not author_id:
                skipped += 1
                continue

            try:
                result_path, method, attempts = download_photo(
                    author_id, photo_id, wikipedia_image, output_dir, existing_photos
                )
                total_attempts += attempts

                if method == 'cached':
                    cached += 1
                    if cached <= 5 or cached % 50 == 0:
                        print(f"  [{processed:>6}] - Cached (skipping...)")
                elif method.startswith('photo_id'):
                    by_photo_id += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    if by_photo_id % 10 == 0 or by_photo_id <= 10:
                        print(f"  [{processed:>6}] {author_id} - {method} ({size_kb:.0f}KB) - {name}")
                elif method.startswith('olid'):
                    by_olid += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    print(f"  [{processed:>6}] {author_id} - {method} ({size_kb:.0f}KB) - {name}")
                elif method.startswith('wikipedia'):
                    by_wikipedia += 1
                    size_kb = result_path.stat().st_size / 1024 if result_path else 0
                    print(f"  [{processed:>6}] {author_id} - {method} ({size_kb:.0f}KB) - {name}")
                else:
                    failed += 1
                    if failed <= 30:
                        print(f"  [{processed:>6}] {author_id} - NOT FOUND - {name}")

            except HTTPError as e:
                if e.code == 403:
                    print(f"\n  RATE LIMITED! Waiting 60 seconds...")
                    time.sleep(60)
                    try:
                        result_path, method, attempts = download_photo(
                            author_id, photo_id, wikipedia_image, output_dir, existing_photos
                        )
                        total_attempts += attempts
                        if method.startswith('photo_id'):
                            by_photo_id += 1
                        elif method.startswith('olid'):
                            by_olid += 1
                        elif method.startswith('wikipedia'):
                            by_wikipedia += 1
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
                total_downloaded = by_photo_id + by_olid + by_wikipedia
                rate = total_downloaded / max(elapsed.total_seconds(), 1) * 3600
                print(f"\n  Progress: {processed:,} | Photo ID: {by_photo_id:,} | OLID: {by_olid:,} | "
                      f"Wiki: {by_wikipedia:,} | Failed: {failed:,} | Cached: {cached:,} | {rate:.0f}/hr\n")

            # Only delay if we actually made API requests (not cached)
            if method != 'cached':
                time.sleep(delay)

    elapsed = datetime.now() - start_time
    total_downloaded = by_photo_id + by_olid + by_wikipedia
    avg_attempts = total_attempts / max(total_downloaded + failed, 1)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time:          {elapsed}")
    print(f"Processed:             {processed:,}")
    print(f"Downloaded (Photo ID): {by_photo_id:,}")
    print(f"Downloaded (OLID):     {by_olid:,}")
    print(f"Downloaded (Wikipedia):{by_wikipedia:,}")
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
        description="Download author photos from Open Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_author_photos.py                      # Download all from master_authors
  python download_author_photos.py --limit 1000         # Download first 1000
  python download_author_photos.py --delay 5            # Slower rate
  python download_author_photos.py --start-row 5000     # Resume from row 5000
        """
    )

    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Input CSV with author_id and photo_id columns (default: {DEFAULT_INPUT})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for photos (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between requests in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of photos to download (0=all)')
    parser.add_argument('--start-row', type=int, default=1,
                        help='Start from this row number (for resuming)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run extract_authors.py first to generate the input file.")
        sys.exit(1)

    download_photos(
        input_path=args.input,
        output_dir=args.output_dir,
        delay=args.delay,
        limit=args.limit,
        start_row=args.start_row
    )


if __name__ == "__main__":
    main()
