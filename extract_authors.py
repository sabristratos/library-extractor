"""
Author Extraction & Enrichment Pipeline
========================================
Creates enriched author profiles and pivot tables from Open Library dumps.

Outputs:
    - master_authors.csv: Rich author profiles with Wikidata enrichment
    - author_works.csv: Pivot table linking authors to works
    - work_subjects.csv: Pivot table linking works to subjects
    - author_subjects.csv: Pivot table linking authors to subjects/genres

Usage:
    python extract_authors.py
    python extract_authors.py --all-authors  # Process all authors, not just top books
    python extract_authors.py --top-n 1000   # Only top 1000 authors by work count

Author: Data Engineering Team
"""

import argparse
import csv
import gzip
import html
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("./output")
TOP_BOOKS_FILE = OUTPUT_DIR / "top_100k_books.csv"
PROGRESS_INTERVAL = 1000000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WikidataEnrichment:
    """Wikidata enrichment data for an author."""
    wikidata_id: Optional[str] = None
    wikipedia_url: Optional[str] = None
    wikipedia_image: Optional[str] = None  # For photo fallback
    description: Optional[str] = None
    nationality: Optional[str] = None
    viaf_id: Optional[str] = None
    isni: Optional[str] = None
    imdb_id: Optional[str] = None
    goodreads_id: Optional[str] = None
    librarything_id: Optional[str] = None


@dataclass
class AuthorRecord:
    """Complete author record with all enrichments."""
    author_id: str
    name: str = ""
    alternate_names: List[str] = field(default_factory=list)
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    bio: Optional[str] = None
    photo_id: Optional[int] = None
    personal_website: Optional[str] = None
    work_count: int = 0
    wikidata: Optional[WikidataEnrichment] = None

    @property
    def is_alive(self) -> Optional[bool]:
        """Determine if author is likely alive (has birth but no death)."""
        if self.birth_year and not self.death_year:
            # If born less than 120 years ago and no death date, likely alive
            current_year = datetime.now().year
            if current_year - self.birth_year < 120:
                return True
        if self.death_year:
            return False
        return None  # Unknown


@dataclass
class AuthorWorkLink:
    """Link between author and work."""
    author_id: str
    work_id: str
    role: str = "author"
    position: int = 1


@dataclass
class WorkSubjectLink:
    """Link between work and subject/genre."""
    work_id: str
    subject: str


@dataclass
class AuthorSubjectLink:
    """Aggregated link between author and subject/genre."""
    author_id: str
    subject: str
    work_count: int = 1  # Number of works with this subject


# =============================================================================
# UTILITIES
# =============================================================================

def find_dump_file(pattern: str, directory: Path = Path(".")) -> Optional[Path]:
    """Find dump file, preferring .gz versions."""
    for ext in ['.txt.gz', '.gz', '.txt', '']:
        candidates = list(directory.glob(f"{pattern}*{ext}"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def open_file(path: Path, mode: str = 'rt'):
    """Open file, handling gzip transparently."""
    if str(path).endswith('.gz'):
        return gzip.open(path, mode, encoding='utf-8', errors='replace')
    return open(path, mode, encoding='utf-8', errors='replace')


def extract_id(key: str, prefix: str) -> str:
    """Extract ID from OL key format."""
    if key.startswith(prefix):
        return key.replace(prefix, '')
    return key


def clean_text(text: str, max_length: int = 1000) -> str:
    """Clean and truncate text."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_length]


def strip_html_and_markdown(text: str) -> str:
    """Remove HTML tags and common markdown from text."""
    if not text:
        return ""
    # Decode HTML entities
    text = html.unescape(str(text))
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove markdown bold/italic
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_year(date_str: str) -> Optional[int]:
    """Extract year from various date formats."""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    # Try to find a 4-digit year (1800-2099)
    match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', date_str)
    if match:
        year = int(match.group(1))
        # Sanity check
        if 1000 <= year <= datetime.now().year + 1:
            return year

    # Handle "c. 1920" or "circa 1920"
    match = re.search(r'(?:c\.?|circa)\s*(1[89]\d{2}|20\d{2})', date_str, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Handle decade like "1920s"
    match = re.search(r'(1[89]\d{2}|20\d{2})s', date_str)
    if match:
        return int(match.group(1))

    return None


def normalize_subject(subject: str) -> Optional[str]:
    """Normalize subject string for consistency."""
    if not subject:
        return None
    subject = str(subject).strip().lower()
    # Remove common noise
    subject = re.sub(r'\s*\([^)]*\)', '', subject)  # Remove parentheticals
    subject = re.sub(r'\s+', ' ', subject).strip()
    # Skip if too short or too long
    if len(subject) < 2 or len(subject) > 100:
        return None
    # Skip numeric-only subjects
    if subject.isdigit():
        return None
    return subject.title()


def progress(count: int, interval: int = PROGRESS_INTERVAL) -> bool:
    """Check if we should print progress."""
    return count % interval == 0


# =============================================================================
# STEP 1: LOAD TOP WORKS
# =============================================================================

def load_top_work_ids(top_books_path: Path) -> Set[str]:
    """Load work IDs from top books CSV."""
    print("\n" + "=" * 70)
    print("STEP 1: LOADING TOP WORK IDs")
    print("=" * 70)
    print(f"Input: {top_books_path}")

    work_ids = set()

    if not top_books_path.exists():
        print(f"WARNING: {top_books_path} not found. Will process ALL authors.")
        return work_ids

    with open(top_books_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            work_id = row.get('work_id', '').strip()
            if work_id:
                work_ids.add(work_id)

    print(f"Loaded {len(work_ids):,} work IDs")
    return work_ids


# =============================================================================
# STEP 2: EXTRACT AUTHOR-WORK LINKS FROM WORKS DUMP
# =============================================================================

def extract_author_links(works_path: Path, target_work_ids: Set[str],
                         process_all: bool = False) -> Tuple[List[AuthorWorkLink], Set[str], List[WorkSubjectLink], Dict[str, Dict[str, int]]]:
    """
    Extract author-work relationships and subjects from works dump.
    Returns: (author_work_links, author_ids, work_subject_links, {author_id: {subject: count}})
    """
    print("\n" + "=" * 70)
    print("STEP 2: EXTRACTING AUTHOR-WORK LINKS & SUBJECTS")
    print("=" * 70)
    print(f"Input: {works_path}")
    if not process_all:
        print(f"Filtering to: {len(target_work_ids):,} works")

    author_work_links: List[AuthorWorkLink] = []
    work_subject_links: List[WorkSubjectLink] = []
    author_ids: Set[str] = set()
    author_subjects: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processed = 0
    works_found = 0

    with open_file(works_path) as f:
        for line in f:
            processed += 1
            if progress(processed):
                print(f"  Processed: {processed:,} | Works: {works_found:,} | Authors: {len(author_ids):,} | Work-Subjects: {len(work_subject_links):,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                work_id = extract_id(key, '/works/')

                # Skip if not in target list (unless processing all)
                if not process_all and work_id not in target_work_ids:
                    continue

                works_found += 1
                data = json.loads(parts[4])

                # Extract subjects for this work
                work_subjects_set = set()
                for subject in data.get('subjects', [])[:20]:  # Limit subjects per work
                    normalized = normalize_subject(subject)
                    if normalized:
                        work_subjects_set.add(normalized)
                        work_subject_links.append(WorkSubjectLink(
                            work_id=work_id,
                            subject=normalized
                        ))

                # Extract authors with roles
                work_author_ids = []
                authors = data.get('authors', [])
                for position, author_entry in enumerate(authors, 1):
                    author_ref = None
                    role = "author"

                    if isinstance(author_entry, dict):
                        # Can be {"author": {"key": "/authors/..."}} or {"key": "/authors/..."}
                        if 'author' in author_entry:
                            author_obj = author_entry['author']
                            if isinstance(author_obj, dict):
                                author_ref = author_obj.get('key', '')
                            elif isinstance(author_obj, str):
                                author_ref = author_obj
                        elif 'key' in author_entry:
                            author_ref = author_entry.get('key', '')

                        # Check for role/type
                        role = author_entry.get('type', {})
                        if isinstance(role, dict):
                            role = role.get('key', 'author').split('/')[-1]
                        elif not isinstance(role, str):
                            role = "author"

                    elif isinstance(author_entry, str):
                        author_ref = author_entry

                    if author_ref:
                        author_id = extract_id(author_ref, '/authors/')
                        if author_id:
                            author_ids.add(author_id)
                            work_author_ids.append(author_id)
                            author_work_links.append(AuthorWorkLink(
                                author_id=author_id,
                                work_id=work_id,
                                role=role.lower() if role else "author",
                                position=position
                            ))

                # Aggregate subjects per author
                for author_id in work_author_ids:
                    for subject in work_subjects_set:
                        author_subjects[author_id][subject] += 1

            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    print(f"\nWorks processed: {works_found:,}")
    print(f"Unique authors: {len(author_ids):,}")
    print(f"Author-work links: {len(author_work_links):,}")
    print(f"Work-subject links: {len(work_subject_links):,}")
    print(f"Authors with subjects: {len(author_subjects):,}")

    return author_work_links, author_ids, work_subject_links, dict(author_subjects)


# =============================================================================
# STEP 3: EXTRACT ENRICHMENT FROM AUTHOR DATA
# =============================================================================

def extract_wikidata_from_author(data: dict) -> WikidataEnrichment:
    """
    Extract Wikidata enrichment fields from an Open Library author record.
    Uses correct field paths for OL author JSON structure.
    """
    enrichment = WikidataEnrichment()

    # Remote IDs contain wikidata, viaf, isni, etc.
    remote_ids = data.get('remote_ids', {})
    if isinstance(remote_ids, dict):
        # Wikidata ID (e.g., "Q42")
        wikidata = remote_ids.get('wikidata')
        if wikidata:
            enrichment.wikidata_id = str(wikidata)

        # VIAF
        viaf = remote_ids.get('viaf')
        if viaf:
            enrichment.viaf_id = str(viaf)

        # ISNI
        isni = remote_ids.get('isni')
        if isni:
            enrichment.isni = str(isni)

        # IMDB
        imdb = remote_ids.get('imdb')
        if imdb:
            enrichment.imdb_id = str(imdb)

        # Goodreads
        goodreads = remote_ids.get('goodreads')
        if goodreads:
            enrichment.goodreads_id = str(goodreads)

        # LibraryThing
        librarything = remote_ids.get('librarything')
        if librarything:
            enrichment.librarything_id = str(librarything)

    # Also check legacy 'identifiers' field (older records)
    identifiers = data.get('identifiers', {})
    if isinstance(identifiers, dict):
        if not enrichment.viaf_id:
            viaf = identifiers.get('viaf', [])
            if viaf and isinstance(viaf, list):
                enrichment.viaf_id = str(viaf[0])
        if not enrichment.isni:
            isni = identifiers.get('isni', [])
            if isni and isinstance(isni, list):
                enrichment.isni = str(isni[0])
        if not enrichment.goodreads_id:
            goodreads = identifiers.get('goodreads', [])
            if goodreads and isinstance(goodreads, list):
                enrichment.goodreads_id = str(goodreads[0])
        if not enrichment.librarything_id:
            librarything = identifiers.get('librarything', [])
            if librarything and isinstance(librarything, list):
                enrichment.librarything_id = str(librarything[0])

    # Wikipedia URL - found in links array
    links = data.get('links', [])
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                url = link.get('url', '')
                title = link.get('title', '').lower()
                # Look for Wikipedia links
                if 'wikipedia' in url.lower() or 'wikipedia' in title:
                    enrichment.wikipedia_url = url
                    # Try to extract image name from Wikipedia (filename after /wiki/File:)
                    break

    # Also check direct wikipedia field (some records have it)
    if not enrichment.wikipedia_url:
        wikipedia = data.get('wikipedia', '')
        if wikipedia:
            enrichment.wikipedia_url = str(wikipedia)

    # Description (bio summary from Wikidata)
    description = data.get('description', '')
    if description:
        enrichment.description = clean_text(str(description), 500)

    # Nationality - check multiple possible fields
    nationality = data.get('nationality', '') or data.get('country', '')
    if nationality:
        enrichment.nationality = clean_text(str(nationality), 100)

    # Wikipedia image for photo fallback
    # Check multiple possible locations
    wiki_image = data.get('wikipedia_image', '') or data.get('image', '')
    if wiki_image:
        enrichment.wikipedia_image = str(wiki_image)

    return enrichment


# =============================================================================
# STEP 3: PROCESS AUTHORS DUMP (Single Pass with Enrichment)
# =============================================================================

def extract_authors(authors_path: Path, target_author_ids: Set[str],
                    work_counts: Dict[str, int],
                    process_all: bool = False) -> List[AuthorRecord]:
    """
    Extract and enrich author records from authors dump in a single pass.
    Extracts both core author data AND Wikidata enrichment from the same records.
    Returns: List of AuthorRecord
    """
    print("\n" + "=" * 70)
    print("STEP 3: EXTRACTING AUTHOR DATA (Single Pass with Enrichment)")
    print("=" * 70)
    print(f"Input: {authors_path}")
    if not process_all:
        print(f"Filtering to: {len(target_author_ids):,} authors")

    authors: List[AuthorRecord] = []
    processed = 0
    found = 0

    with open_file(authors_path) as f:
        for line in f:
            processed += 1
            if progress(processed):
                print(f"  Processed: {processed:,} | Found: {found:,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                author_id = extract_id(key, '/authors/')

                # Skip if not in target list
                if not process_all and author_id not in target_author_ids:
                    continue

                data = json.loads(parts[4])
                found += 1

                # Create author record
                record = AuthorRecord(author_id=author_id)

                # Name
                name = data.get('name', '')
                if name:
                    record.name = clean_text(name, 200)

                # Alternate names
                alt_names = data.get('alternate_names', [])
                if alt_names and isinstance(alt_names, list):
                    record.alternate_names = [clean_text(n, 100) for n in alt_names[:10] if n]

                # Birth date
                birth = data.get('birth_date', '')
                if birth:
                    record.birth_date = clean_text(str(birth), 50)
                    record.birth_year = parse_year(birth)

                # Death date
                death = data.get('death_date', '')
                if death:
                    record.death_date = clean_text(str(death), 50)
                    record.death_year = parse_year(death)

                # Bio - strip HTML/markdown
                bio = data.get('bio', '')
                if isinstance(bio, dict):
                    bio = bio.get('value', '')
                if bio:
                    bio_clean = strip_html_and_markdown(str(bio))
                    record.bio = clean_text(bio_clean, 2000)

                # Photo ID
                photos = data.get('photos', [])
                if photos and isinstance(photos, list):
                    for photo in photos:
                        if isinstance(photo, int) and photo > 0:
                            record.photo_id = photo
                            break

                # Personal website (from links - skip Wikipedia links)
                links = data.get('links', [])
                if links and isinstance(links, list):
                    for link in links:
                        if isinstance(link, dict):
                            url = link.get('url', '')
                            title = link.get('title', '').lower()
                            # Skip Wikipedia links for personal website
                            if 'wikipedia' in url.lower() or 'wikipedia' in title:
                                continue
                            if url and ('official' in title or 'personal' in title or 'website' in title):
                                record.personal_website = url
                                break
                    # If no official site, take first non-Wikipedia link
                    if not record.personal_website and links:
                        for link in links:
                            if isinstance(link, dict):
                                url = link.get('url', '')
                                if url and 'wikipedia' not in url.lower():
                                    record.personal_website = url
                                    break

                # Work count
                record.work_count = work_counts.get(author_id, 0)

                # Wikidata enrichment - extracted from same author record
                record.wikidata = extract_wikidata_from_author(data)

                authors.append(record)

            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    # Calculate stats
    with_wikidata = sum(1 for a in authors if a.wikidata and a.wikidata.wikidata_id)
    with_wikipedia = sum(1 for a in authors if a.wikidata and a.wikidata.wikipedia_url)
    with_remote_ids = sum(1 for a in authors if a.wikidata and (
        a.wikidata.viaf_id or a.wikidata.isni or a.wikidata.goodreads_id
    ))

    print(f"\nAuthors found: {found:,}")
    print(f"With photos: {sum(1 for a in authors if a.photo_id):,}")
    print(f"With bio: {sum(1 for a in authors if a.bio):,}")
    print(f"With Wikidata ID: {with_wikidata:,}")
    print(f"With Wikipedia URL: {with_wikipedia:,}")
    print(f"With Remote IDs: {with_remote_ids:,}")

    return authors


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def write_author_works(links: List[AuthorWorkLink], output_path: Path):
    """Write author-work pivot table."""
    print("\n" + "=" * 70)
    print("WRITING AUTHOR-WORKS PIVOT TABLE")
    print("=" * 70)
    print(f"Output: {output_path}")

    columns = ['author_id', 'work_id', 'role', 'position']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for link in links:
            writer.writerow({
                'author_id': link.author_id,
                'work_id': link.work_id,
                'role': link.role,
                'position': link.position
            })

    print(f"Written: {len(links):,} rows")


def write_work_subjects(links: List[WorkSubjectLink], output_path: Path):
    """Write work-subject pivot table."""
    print("\n" + "=" * 70)
    print("WRITING WORK-SUBJECTS PIVOT TABLE")
    print("=" * 70)
    print(f"Output: {output_path}")

    columns = ['work_id', 'subject']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for link in links:
            writer.writerow({
                'work_id': link.work_id,
                'subject': link.subject
            })

    print(f"Written: {len(links):,} rows")

    # Show top subjects
    subject_counts: Dict[str, int] = defaultdict(int)
    for link in links:
        subject_counts[link.subject] += 1
    top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Subjects:")
    for subject, count in top_subjects:
        print(f"  {count:>6,} - {subject}")


def write_author_subjects(author_subjects: Dict[str, Dict[str, int]], output_path: Path, min_count: int = 2):
    """Write author-subject pivot table (aggregated)."""
    print("\n" + "=" * 70)
    print("WRITING AUTHOR-SUBJECTS PIVOT TABLE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Min work count: {min_count}")

    columns = ['author_id', 'subject', 'work_count']

    rows_written = 0
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for author_id, subjects in author_subjects.items():
            for subject, count in subjects.items():
                if count >= min_count:  # Only include if author has multiple works in subject
                    writer.writerow({
                        'author_id': author_id,
                        'subject': subject,
                        'work_count': count
                    })
                    rows_written += 1

    print(f"Written: {rows_written:,} rows")


def write_authors(authors: List[AuthorRecord], output_path: Path):
    """Write enriched authors CSV."""
    print("\n" + "=" * 70)
    print("WRITING MASTER AUTHORS CSV")
    print("=" * 70)
    print(f"Output: {output_path}")

    columns = [
        'author_id', 'name', 'alternate_names',
        'birth_date', 'birth_year', 'death_date', 'death_year', 'is_alive',
        'bio', 'photo_id', 'personal_website', 'work_count',
        'nationality', 'wikidata_id', 'wikipedia_url', 'wikipedia_image', 'description',
        'viaf_id', 'isni', 'imdb_id', 'goodreads_id', 'librarything_id'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for author in authors:
            # Determine is_alive status
            is_alive_str = ''
            if author.is_alive is True:
                is_alive_str = 'true'
            elif author.is_alive is False:
                is_alive_str = 'false'

            row = {
                'author_id': author.author_id,
                'name': author.name,
                'alternate_names': '|'.join(author.alternate_names) if author.alternate_names else '',
                'birth_date': author.birth_date or '',
                'birth_year': author.birth_year or '',
                'death_date': author.death_date or '',
                'death_year': author.death_year or '',
                'is_alive': is_alive_str,
                'bio': author.bio or '',
                'photo_id': author.photo_id or '',
                'personal_website': author.personal_website or '',
                'work_count': author.work_count,
            }

            # Wikidata fields
            if author.wikidata:
                row['nationality'] = author.wikidata.nationality or ''
                row['wikidata_id'] = author.wikidata.wikidata_id or ''
                row['wikipedia_url'] = author.wikidata.wikipedia_url or ''
                row['wikipedia_image'] = author.wikidata.wikipedia_image or ''
                row['description'] = author.wikidata.description or ''
                row['viaf_id'] = author.wikidata.viaf_id or ''
                row['isni'] = author.wikidata.isni or ''
                row['imdb_id'] = author.wikidata.imdb_id or ''
                row['goodreads_id'] = author.wikidata.goodreads_id or ''
                row['librarything_id'] = author.wikidata.librarything_id or ''
            else:
                row['nationality'] = ''
                row['wikidata_id'] = ''
                row['wikipedia_url'] = ''
                row['wikipedia_image'] = ''
                row['description'] = ''
                row['viaf_id'] = ''
                row['isni'] = ''
                row['imdb_id'] = ''
                row['goodreads_id'] = ''
                row['librarything_id'] = ''

            writer.writerow(row)

    print(f"Written: {len(authors):,} authors")

    # Stats
    with_photo = sum(1 for a in authors if a.photo_id)
    with_birth = sum(1 for a in authors if a.birth_year)
    alive = sum(1 for a in authors if a.is_alive is True)
    with_wiki = sum(1 for a in authors if a.wikidata and a.wikidata.wikidata_id)

    print(f"With photo ID: {with_photo:,}")
    print(f"With birth year: {with_birth:,}")
    print(f"Likely alive: {alive:,}")
    print(f"With Wikidata: {with_wiki:,}")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract and enrich author data from Open Library dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_authors.py                  # Authors from top 100k books
  python extract_authors.py --all-authors    # ALL authors (very large!)
  python extract_authors.py --top-n 1000     # Only output top 1000 by work count
        """
    )

    parser.add_argument('--all-authors', action='store_true',
                        help='Process all authors, not just those in top books')
    parser.add_argument('--top-n', type=int, default=0,
                        help='Only output top N authors by work count (0=all)')
    parser.add_argument('--input-dir', type=Path, default=Path('.'),
                        help='Directory containing dump files')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='Output directory')

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("AUTHOR EXTRACTION & ENRICHMENT PIPELINE")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode:       {'All authors' if args.all_authors else 'Top books authors only'}")
    if args.top_n:
        print(f"Top N:      {args.top_n:,} authors")
    print(f"Output Dir: {args.output_dir}")

    start_time = datetime.now()

    # Find input files
    works_path = find_dump_file("ol_dump_works", args.input_dir)
    authors_path = find_dump_file("ol_dump_authors", args.input_dir)

    print(f"\nInput Files:")
    print(f"  Works:    {works_path or 'NOT FOUND'}")
    print(f"  Authors:  {authors_path or 'NOT FOUND'}")

    if not works_path or not authors_path:
        print("\nERROR: Required dump files not found!")
        sys.exit(1)

    # Step 1: Load top work IDs (unless processing all)
    target_work_ids = set()
    if not args.all_authors:
        target_work_ids = load_top_work_ids(TOP_BOOKS_FILE)
        if not target_work_ids:
            print("\nWARNING: No work IDs loaded. Processing all authors.")
            args.all_authors = True

    # Step 2: Extract author-work links and subjects
    author_work_links, author_ids, work_subject_links, author_subjects = extract_author_links(
        works_path, target_work_ids, process_all=args.all_authors
    )

    # Calculate work counts per author
    work_counts: Dict[str, int] = defaultdict(int)
    for link in author_work_links:
        work_counts[link.author_id] += 1

    # Step 3: Extract authors (with enrichment in single pass)
    authors = extract_authors(
        authors_path, author_ids, work_counts,
        process_all=args.all_authors
    )

    # Sort authors by work count (most prolific first)
    authors.sort(key=lambda a: a.work_count, reverse=True)

    # Apply top-n filter if specified
    if args.top_n and args.top_n < len(authors):
        print(f"\nFiltering to top {args.top_n:,} authors...")
        top_author_ids = {a.author_id for a in authors[:args.top_n]}
        authors = authors[:args.top_n]

        # Filter author-work links to only include top authors
        author_work_links = [l for l in author_work_links if l.author_id in top_author_ids]

        # Get work IDs that belong to top authors
        top_author_work_ids = {l.work_id for l in author_work_links}

        # Filter work_subjects to only include works by top authors
        work_subject_links = [l for l in work_subject_links if l.work_id in top_author_work_ids]

        # Filter author_subjects to only include top authors
        author_subjects = {k: v for k, v in author_subjects.items() if k in top_author_ids}

        print(f"  Filtered author_works: {len(author_work_links):,} links")
        print(f"  Filtered work_subjects: {len(work_subject_links):,} links")

    # Write outputs
    write_author_works(author_work_links, args.output_dir / "author_works.csv")
    write_work_subjects(work_subject_links, args.output_dir / "work_subjects.csv")
    write_author_subjects(author_subjects, args.output_dir / "author_subjects.csv")
    write_authors(authors, args.output_dir / "master_authors.csv")

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time: {elapsed}")
    print(f"\nOutput Files:")
    print(f"  {args.output_dir / 'master_authors.csv'}")
    print(f"  {args.output_dir / 'author_works.csv'}")
    print(f"  {args.output_dir / 'work_subjects.csv'}")
    print(f"  {args.output_dir / 'author_subjects.csv'}")

    # Preview top authors
    print("\nTop 10 Authors by Work Count:")
    print("-" * 70)
    for i, author in enumerate(authors[:10], 1):
        wiki = "W" if author.wikidata and author.wikidata.wikidata_id else " "
        photo = "P" if author.photo_id else " "
        alive = "A" if author.is_alive else " "
        print(f"  {i:>3}. {author.name[:35]:<37} {author.work_count:>4} works [{photo}{wiki}{alive}]")


if __name__ == "__main__":
    main()
