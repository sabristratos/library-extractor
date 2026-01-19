"""
Wikidata Image Enrichment for Authors
======================================
Fetches Wikipedia/Wikimedia image filenames from Wikidata API for authors
that have a wikidata_id but no photo source.

This script enriches master_authors.csv with wikipedia_image values that
can be used by download_author_photos.py as a fallback.

Usage:
    python enrich_author_images.py
    python enrich_author_images.py --input output/master_authors.csv --limit 1000
    python enrich_author_images.py --delay 0.5

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
from typing import Dict, List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# SSL context
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path("./output/master_authors.csv")
DEFAULT_OUTPUT = Path("./output/master_authors.csv")  # In-place update
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
DEFAULT_DELAY = 0.2  # Wikidata API is generous with rate limits
TIMEOUT = 30
BATCH_SIZE = 50  # Wikidata API supports batching


# =============================================================================
# WIKIDATA API
# =============================================================================

def fetch_wikidata_images(wikidata_ids: List[str]) -> Dict[str, str]:
    """
    Fetch image filenames from Wikidata for multiple entities.
    Uses the wbgetentities API to batch requests.
    Returns: {wikidata_id: image_filename}
    """
    if not wikidata_ids:
        return {}

    results = {}

    # Wikidata API accepts up to 50 IDs at once
    ids_str = '|'.join(wikidata_ids)

    try:
        # Request P18 (image) property for all entities
        url = (
            f"{WIKIDATA_API}?"
            f"action=wbgetentities"
            f"&ids={ids_str}"
            f"&props=claims"
            f"&format=json"
        )

        request = Request(url, headers={
            'User-Agent': 'OpenLibraryAuthorEnricher/1.0 (Educational Project)'
        })

        with urlopen(request, timeout=TIMEOUT, context=SSL_CONTEXT) as response:
            data = json.loads(response.read().decode('utf-8'))

            entities = data.get('entities', {})
            for qid, entity in entities.items():
                if 'claims' not in entity:
                    continue

                claims = entity['claims']

                # P18 is the "image" property
                if 'P18' in claims:
                    p18_claims = claims['P18']
                    if p18_claims and isinstance(p18_claims, list):
                        first_claim = p18_claims[0]
                        mainsnak = first_claim.get('mainsnak', {})
                        datavalue = mainsnak.get('datavalue', {})
                        if datavalue.get('type') == 'string':
                            image_name = datavalue.get('value', '')
                            if image_name:
                                results[qid] = image_name

    except HTTPError as e:
        if e.code == 429:
            print(f"  Rate limited. Waiting 60s...")
            time.sleep(60)
        return {}
    except (URLError, json.JSONDecodeError, Exception) as e:
        return {}

    return results


def fetch_wikipedia_image_from_url(wikipedia_url: str) -> Optional[str]:
    """
    Fetch the main image from a Wikipedia page.
    Returns the image filename if found.
    """
    if not wikipedia_url:
        return None

    try:
        # Extract the page title from URL
        # e.g., https://en.wikipedia.org/wiki/J._K._Rowling -> J._K._Rowling
        if '/wiki/' in wikipedia_url:
            page_title = wikipedia_url.split('/wiki/')[-1].split('#')[0].split('?')[0]
        else:
            return None

        # Determine the language/wiki from URL
        if 'en.wikipedia.org' in wikipedia_url:
            api_base = "https://en.wikipedia.org/w/api.php"
        elif 'wikipedia.org' in wikipedia_url:
            # Extract language code
            lang = wikipedia_url.split('://')[1].split('.')[0]
            api_base = f"https://{lang}.wikipedia.org/w/api.php"
        else:
            return None

        # Use pageimages API to get the main image
        url = (
            f"{api_base}?"
            f"action=query"
            f"&titles={quote(page_title)}"
            f"&prop=pageimages"
            f"&piprop=original"
            f"&format=json"
        )

        request = Request(url, headers={
            'User-Agent': 'OpenLibraryAuthorEnricher/1.0 (Educational Project)'
        })

        with urlopen(request, timeout=TIMEOUT, context=SSL_CONTEXT) as response:
            data = json.loads(response.read().decode('utf-8'))

            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    continue
                original = page_data.get('original', {})
                source = original.get('source', '')
                if source:
                    # Extract filename from URL
                    # e.g., https://upload.wikimedia.org/wikipedia/commons/5/5d/J._K._Rowling_2010.jpg
                    if '/commons/' in source or '/wikipedia/' in source:
                        filename = source.split('/')[-1]
                        return filename

        return None

    except Exception:
        return None


# =============================================================================
# MAIN ENRICHMENT
# =============================================================================

def enrich_author_images(input_path: Path, output_path: Path,
                         delay: float, limit: int = 0):
    """
    Enrich master_authors.csv with wikipedia_image values from Wikidata API.
    """
    print("=" * 70)
    print("WIKIDATA IMAGE ENRICHMENT FOR AUTHORS")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Delay:      {delay}s between batches")
    if limit:
        print(f"Limit:      {limit:,} authors")

    # Read all authors
    print("\nReading authors...")
    authors = []
    fieldnames = None

    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            authors.append(row)

    print(f"Total authors: {len(authors):,}")

    # Find authors needing enrichment
    # Priority 1: Has wikidata_id, no photo_id, no wikipedia_image
    # Priority 2: Has wikipedia_url, no photo_id, no wikipedia_image
    needs_wikidata = []
    needs_wikipedia = []

    for i, author in enumerate(authors):
        has_photo = author.get('photo_id', '').strip()
        has_wiki_image = author.get('wikipedia_image', '').strip()
        wikidata_id = author.get('wikidata_id', '').strip()
        wikipedia_url = author.get('wikipedia_url', '').strip()

        if has_wiki_image:
            continue  # Already has image

        if wikidata_id and not has_photo:
            needs_wikidata.append((i, wikidata_id))
        elif wikipedia_url and not has_photo:
            needs_wikipedia.append((i, wikipedia_url))

    print(f"Need Wikidata lookup: {len(needs_wikidata):,}")
    print(f"Need Wikipedia lookup: {len(needs_wikipedia):,}")

    if limit:
        needs_wikidata = needs_wikidata[:limit]
        needs_wikipedia = needs_wikipedia[:max(0, limit - len(needs_wikidata))]

    # Process Wikidata lookups in batches
    print("\n" + "-" * 70)
    print("WIKIDATA API LOOKUPS")
    print("-" * 70)

    enriched_wikidata = 0
    total_batches = (len(needs_wikidata) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(needs_wikidata))
        batch = needs_wikidata[start_idx:end_idx]

        # Extract wikidata IDs for this batch
        batch_ids = [wid for _, wid in batch]
        batch_indices = {wid: idx for idx, wid in batch}

        # Fetch images from Wikidata
        images = fetch_wikidata_images(batch_ids)

        # Update authors
        for wid, image_name in images.items():
            if wid in batch_indices:
                author_idx = batch_indices[wid]
                authors[author_idx]['wikipedia_image'] = image_name
                enriched_wikidata += 1

        if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
            print(f"  Batch {batch_num + 1}/{total_batches} | Enriched: {enriched_wikidata:,}")

        time.sleep(delay)

    # Process Wikipedia URL lookups (one at a time, slower)
    print("\n" + "-" * 70)
    print("WIKIPEDIA PAGE LOOKUPS")
    print("-" * 70)

    enriched_wikipedia = 0
    for i, (author_idx, wiki_url) in enumerate(needs_wikipedia):
        image_name = fetch_wikipedia_image_from_url(wiki_url)
        if image_name:
            authors[author_idx]['wikipedia_image'] = image_name
            enriched_wikipedia += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(needs_wikipedia)} | Enriched: {enriched_wikipedia:,}")

        time.sleep(delay)

    # Write output
    print("\n" + "-" * 70)
    print("WRITING OUTPUT")
    print("-" * 70)

    # Ensure wikipedia_image column exists
    if 'wikipedia_image' not in fieldnames:
        fieldnames = list(fieldnames) + ['wikipedia_image']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for author in authors:
            writer.writerow(author)

    print(f"Written: {len(authors):,} authors")

    # Summary
    total_enriched = enriched_wikidata + enriched_wikipedia
    total_with_image = sum(1 for a in authors if a.get('wikipedia_image', '').strip())

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print(f"Enriched via Wikidata:  {enriched_wikidata:,}")
    print(f"Enriched via Wikipedia: {enriched_wikipedia:,}")
    print(f"Total Enriched:         {total_enriched:,}")
    print(f"Total with Image:       {total_with_image:,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich author data with Wikidata/Wikipedia images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enrich_author_images.py                      # Enrich all authors
  python enrich_author_images.py --limit 1000         # Enrich first 1000 needing images
  python enrich_author_images.py --delay 0.5          # Slower rate
        """
    )

    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Input CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV (default: same as input, in-place update)')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between API calls in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of authors to enrich (0=all)')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run extract_authors.py first to generate the input file.")
        sys.exit(1)

    enrich_author_images(
        input_path=args.input,
        output_path=args.output,
        delay=args.delay,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
