"""
Top 100,000 Books Extraction Pipeline
======================================
Generates a curated dataset of the most popular books from Open Library
for use in recommendation engines and frontend displays.

Usage:
    python extract_top_books.py
    python extract_top_books.py --top 50000 --output top_50k_books.csv

Author: Data Engineering Team
"""

import argparse
import csv
import gzip
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
DEFAULT_TOP_N = 100000
PROGRESS_INTERVAL = 1000000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WorkStats:
    interactions: int = 0
    rating_sum: float = 0.0
    rating_count: int = 0
    want_to_read: int = 0
    currently_reading: int = 0
    already_read: int = 0

    @property
    def avg_rating(self) -> Optional[float]:
        if self.rating_count == 0:
            return None
        return round(self.rating_sum / self.rating_count, 2)

    @property
    def popularity_score(self) -> int:
        return self.interactions + (self.rating_count * 2)


@dataclass
class WorkMetadata:
    title: Optional[str] = None
    subtitle: Optional[str] = None
    description: Optional[str] = None
    first_publish_year: Optional[int] = None
    series_name: Optional[str] = None
    series_position: Optional[str] = None
    author_ids: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)


@dataclass
class BookRecord:
    work_id: str
    title: Optional[str] = None
    isbn: Optional[str] = None
    has_cover: bool = False
    rating_avg: Optional[float] = None
    rating_count: int = 0
    interaction_count: int = 0
    popularity_score: int = 0


# =============================================================================
# UTILITIES
# =============================================================================

def find_dump_file(pattern: str, directory: Path = Path(".")) -> Optional[Path]:
    # Prefer .gz files (more reliable, less likely to be locked)
    for ext in ['.txt.gz', '.gz', '.txt', '']:
        candidates = list(directory.glob(f"{pattern}*{ext}"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def open_file(path: Path, mode: str = 'rt'):
    if str(path).endswith('.gz'):
        return gzip.open(path, mode, encoding='utf-8', errors='replace')
    return open(path, mode, encoding='utf-8', errors='replace')


def extract_work_id(key: str) -> str:
    if key.startswith('/works/'):
        return key.replace('/works/', '')
    return key


def progress(count: int, interval: int = PROGRESS_INTERVAL) -> bool:
    return count % interval == 0


# =============================================================================
# STEP A: AGGREGATION (Popularity Engine)
# =============================================================================

def aggregate_popularity(ratings_path: Optional[Path],
                         reading_log_path: Optional[Path]) -> Dict[str, WorkStats]:
    """
    Aggregate ratings and reading log data to calculate popularity scores.
    Returns: {work_id: WorkStats}
    """
    print("\n" + "=" * 70)
    print("STEP A: AGGREGATING POPULARITY DATA")
    print("=" * 70)

    work_stats: Dict[str, WorkStats] = defaultdict(WorkStats)
    total_processed = 0

    # Process Ratings
    if ratings_path and ratings_path.exists():
        print(f"\nProcessing Ratings: {ratings_path}")
        processed = 0

        with open_file(ratings_path) as f:
            for line in f:
                processed += 1
                if progress(processed):
                    print(f"  Ratings: {processed:,} lines processed")

                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue

                    work_id = extract_work_id(parts[0])
                    rating = float(parts[2])

                    if 1 <= rating <= 5:
                        work_stats[work_id].rating_sum += rating
                        work_stats[work_id].rating_count += 1
                        work_stats[work_id].interactions += 1

                except (ValueError, IndexError):
                    continue

        print(f"  Ratings Complete: {processed:,} lines")
        total_processed += processed
    else:
        print("\nWARNING: Ratings dump not found")

    # Process Reading Log
    if reading_log_path and reading_log_path.exists():
        print(f"\nProcessing Reading Log: {reading_log_path}")
        processed = 0
        status_counts = defaultdict(int)

        with open_file(reading_log_path) as f:
            for line in f:
                processed += 1
                if progress(processed):
                    print(f"  Reading Log: {processed:,} lines processed")

                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue

                    work_id = extract_work_id(parts[0])
                    status = parts[2].lower() if len(parts) > 2 else ''

                    work_stats[work_id].interactions += 1
                    status_counts[status] += 1

                    # Track specific reading statuses
                    if status in ('want to read', 'want-to-read', 'wanttoread'):
                        work_stats[work_id].want_to_read += 1
                    elif status in ('currently reading', 'currently-reading', 'currentlyreading'):
                        work_stats[work_id].currently_reading += 1
                    elif status in ('already read', 'already-read', 'alreadyread', 'read'):
                        work_stats[work_id].already_read += 1

                except (ValueError, IndexError):
                    continue

        print(f"  Reading Log Complete: {processed:,} lines")
        print(f"  Status Distribution: {dict(status_counts)}")
        total_processed += processed
    else:
        print("\nWARNING: Reading log dump not found")

    print(f"\nTotal Works with Activity: {len(work_stats):,}")
    print(f"Total Lines Processed: {total_processed:,}")

    return work_stats


def select_top_works(work_stats: Dict[str, WorkStats], top_n: int,
                     sort_by: str = 'rating_count') -> List[Tuple[str, WorkStats]]:
    """
    Select top N works by specified criteria.
    sort_by: 'rating_count' (most rated), 'popularity' (interactions + ratings), 'avg_rating'
    Returns: List of (work_id, stats) tuples sorted by criteria
    """
    print(f"\nSelecting Top {top_n:,} Works by {sort_by}...")

    if sort_by == 'rating_count':
        sorted_works = sorted(
            work_stats.items(),
            key=lambda x: (x[1].rating_count, x[1].avg_rating or 0, x[1].interactions),
            reverse=True
        )
    elif sort_by == 'avg_rating':
        min_votes = 100
        filtered = {k: v for k, v in work_stats.items() if v.rating_count >= min_votes}
        sorted_works = sorted(
            filtered.items(),
            key=lambda x: (x[1].avg_rating or 0, x[1].rating_count),
            reverse=True
        )
    else:
        sorted_works = sorted(
            work_stats.items(),
            key=lambda x: (x[1].popularity_score, x[1].rating_count, x[1].avg_rating or 0),
            reverse=True
        )

    top_works = sorted_works[:top_n]

    if top_works:
        print(f"  Rating Count Range: {top_works[-1][1].rating_count:,} - {top_works[0][1].rating_count:,}")
        if top_works[0][1].avg_rating:
            print(f"  Avg Rating Range: {top_works[-1][1].avg_rating:.2f} - {top_works[0][1].avg_rating:.2f}")

    return top_works


# =============================================================================
# STEP B: METADATA ENRICHMENT (Titles, Descriptions, Series)
# =============================================================================

def clean_description(text: str, max_length: int = 2500) -> str:
    """Clean and truncate description text."""
    if not text:
        return ""
    # Handle dict format (some descriptions are {"type": "/type/text", "value": "..."})
    if isinstance(text, dict):
        text = text.get('value', '')
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Clean whitespace
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common prefixes like "----------" or "***"
    text = re.sub(r'^[-*=_]{3,}\s*', '', text)
    return text[:max_length]


def extract_first_publish_year(data: dict) -> Optional[int]:
    """Extract first publication year from work data."""
    # Try first_publish_date field
    first_date = data.get('first_publish_date', '')
    if first_date:
        match = re.search(r'\b(1[5-9]\d{2}|20\d{2})\b', str(first_date))
        if match:
            return int(match.group())

    # Try first_publish_year field (sometimes present)
    first_year = data.get('first_publish_year')
    if first_year:
        try:
            year = int(first_year)
            if 1000 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass

    # Fallback: created timestamp
    created = data.get('created', {})
    if isinstance(created, dict):
        created_val = created.get('value', '')
        if created_val:
            match = re.search(r'\b(19\d{2}|20\d{2})\b', str(created_val))
            if match:
                return int(match.group())

    return None


def extract_series_info(data: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract series name and position from work data.
    Series info can be in:
    1. Dedicated 'series' field (rare)
    2. Subjects array with 'series:' prefix (common) - e.g., "series:Harry_Potter"
    3. Links with series references
    Returns: (series_name, series_position)
    """
    # Method 1: Check 'series' field (list of series references)
    series_list = data.get('series', [])
    if series_list and isinstance(series_list, list):
        for series_entry in series_list:
            if isinstance(series_entry, dict):
                series_name = series_entry.get('name', '') or series_entry.get('title', '')
                series_pos = series_entry.get('position', '') or series_entry.get('number', '')
                if series_name:
                    return clean_description(series_name, 200), str(series_pos) if series_pos else None
            elif isinstance(series_entry, str):
                return clean_description(series_entry, 200), None

    # Method 2: Check subjects array for "series:SeriesName" pattern
    subjects = data.get('subjects', [])
    if subjects and isinstance(subjects, list):
        for subj in subjects:
            if isinstance(subj, str) and subj.lower().startswith('series:'):
                # Extract series name after "series:" prefix
                series_name = subj[7:].strip()  # Remove "series:" prefix
                # Clean up underscores and formatting
                series_name = series_name.replace('_', ' ')
                series_name = re.sub(r'\s+', ' ', series_name).strip()
                if series_name:
                    return series_name, None

    # Method 3: Check links for series info
    links = data.get('links', [])
    if links and isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                title = link.get('title', '').lower()
                if 'series' in title:
                    # Try to extract series name from link title
                    # e.g., "Harry Potter Series" -> "Harry Potter"
                    series_name = link.get('title', '')
                    series_name = re.sub(r'\s*series\s*', '', series_name, flags=re.IGNORECASE).strip()
                    if series_name:
                        return series_name, None

    return None, None


def enrich_metadata(works_path: Path, top_work_ids: Set[str]) -> Dict[str, WorkMetadata]:
    """
    Extract metadata (title, description, first_publish_year, series) for top works.
    Returns: {work_id: WorkMetadata}
    """
    print("\n" + "=" * 70)
    print("STEP B: ENRICHING METADATA (Titles, Descriptions, Series)")
    print("=" * 70)
    print(f"Input: {works_path}")
    print(f"Looking for: {len(top_work_ids):,} work IDs")

    metadata: Dict[str, WorkMetadata] = {}
    processed = 0
    found = 0

    with open_file(works_path) as f:
        for line in f:
            processed += 1
            if progress(processed):
                print(f"  Processed: {processed:,} | Found: {found:,}/{len(top_work_ids):,}")

            if found >= len(top_work_ids):
                print(f"  Early exit: All {len(top_work_ids):,} works found")
                break

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                work_id = extract_work_id(key)

                if work_id not in top_work_ids:
                    continue

                data = json.loads(parts[4])
                found += 1

                work_meta = WorkMetadata()

                # Title
                title = data.get('title', '')
                if title:
                    title = re.sub(r'[\n\r\t]+', ' ', str(title))
                    title = re.sub(r'\s+', ' ', title).strip()
                    work_meta.title = title[:500]

                # Subtitle
                subtitle = data.get('subtitle', '')
                if subtitle:
                    subtitle = re.sub(r'[\n\r\t]+', ' ', str(subtitle))
                    subtitle = re.sub(r'\s+', ' ', subtitle).strip()
                    work_meta.subtitle = subtitle[:300]

                # Description
                description = data.get('description', '')
                if description:
                    work_meta.description = clean_description(description, 2500)

                # First publish year
                work_meta.first_publish_year = extract_first_publish_year(data)

                # Series info
                series_name, series_pos = extract_series_info(data)
                work_meta.series_name = series_name
                work_meta.series_position = series_pos

                # Author IDs
                authors = data.get('authors', [])
                author_ids = []
                for author_entry in authors[:10]:  # Limit to 10 authors
                    author_ref = None
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
                    elif isinstance(author_entry, str):
                        author_ref = author_entry

                    if author_ref:
                        # Extract just the ID part (e.g., "OL123A" from "/authors/OL123A")
                        aid = author_ref.replace('/authors/', '')
                        if aid:
                            author_ids.append(aid)

                work_meta.author_ids = author_ids

                # Subjects (top 5)
                subjects_raw = data.get('subjects', [])
                subjects = []
                for subj in subjects_raw[:10]:  # Check first 10, keep 5 good ones
                    if isinstance(subj, str) and len(subj) > 1 and len(subj) < 100:
                        # Clean subject
                        subj_clean = re.sub(r'\s+', ' ', subj).strip()
                        if subj_clean and not subj_clean.isdigit():
                            subjects.append(subj_clean)
                            if len(subjects) >= 5:
                                break

                work_meta.subjects = subjects

                metadata[work_id] = work_meta

            except (json.JSONDecodeError, IndexError):
                continue

    # Stats
    with_desc = sum(1 for m in metadata.values() if m.description)
    with_year = sum(1 for m in metadata.values() if m.first_publish_year)
    with_series = sum(1 for m in metadata.values() if m.series_name)
    with_subtitle = sum(1 for m in metadata.values() if m.subtitle)
    with_authors = sum(1 for m in metadata.values() if m.author_ids)
    with_subjects = sum(1 for m in metadata.values() if m.subjects)

    print(f"\nWorks Found:       {found:,}/{len(top_work_ids):,} ({100*found/len(top_work_ids):.1f}%)")
    print(f"With Description:  {with_desc:,} ({100*with_desc/max(found,1):.1f}%)")
    print(f"With Subtitle:     {with_subtitle:,} ({100*with_subtitle/max(found,1):.1f}%)")
    print(f"With Publish Year: {with_year:,} ({100*with_year/max(found,1):.1f}%)")
    print(f"With Series:       {with_series:,} ({100*with_series/max(found,1):.1f}%)")
    print(f"With Authors:      {with_authors:,} ({100*with_authors/max(found,1):.1f}%)")
    print(f"With Subjects:     {with_subjects:,} ({100*with_subjects/max(found,1):.1f}%)")

    # Collect all unique author IDs for name resolution
    all_author_ids: Set[str] = set()
    for m in metadata.values():
        all_author_ids.update(m.author_ids)
    print(f"Unique Authors:    {len(all_author_ids):,}")

    return metadata, all_author_ids


# =============================================================================
# STEP C: ISBN RESOLUTION (For Covers)
# =============================================================================

@dataclass
class EditionCandidate:
    """A single edition candidate with ISBN/cover info."""
    isbn: Optional[str] = None
    cover_id: Optional[int] = None
    has_cover: bool = False
    is_english: bool = False
    publisher_tier: int = 1  # 0=quality, 1=normal, 2=selfpub
    publish_year: Optional[int] = None  # Publication year for recency scoring

    @property
    def quality_score(self) -> int:
        """Higher = better. Used for ranking editions."""
        score = 0
        if self.is_english:
            score += 100
        if self.has_cover:
            score += 50
        if self.isbn:
            score += 10
        score -= self.publisher_tier * 20  # Quality=0, Normal=-20, Selfpub=-40
        # Favor newer editions (post-2000 get bonus, post-2010 even more)
        if self.publish_year:
            if self.publish_year >= 2015:
                score += 30
            elif self.publish_year >= 2010:
                score += 20
            elif self.publish_year >= 2000:
                score += 10
        return score


@dataclass
class WorkEditions:
    """Collection of edition candidates for a single work."""
    candidates: List[EditionCandidate] = field(default_factory=list)
    max_candidates: int = 10  # Keep top 10 candidates per work

    def add_candidate(self, candidate: EditionCandidate):
        """Add a candidate, keeping only top N by quality score."""
        if not candidate.isbn and not candidate.cover_id:
            return

        # Check for duplicates
        for existing in self.candidates:
            if candidate.isbn and candidate.isbn == existing.isbn:
                # Update if better quality
                if candidate.quality_score > existing.quality_score:
                    existing.has_cover = candidate.has_cover or existing.has_cover
                    existing.cover_id = candidate.cover_id or existing.cover_id
                    existing.is_english = candidate.is_english or existing.is_english
                    existing.publisher_tier = min(candidate.publisher_tier, existing.publisher_tier)
                return
            if candidate.cover_id and candidate.cover_id == existing.cover_id:
                if candidate.quality_score > existing.quality_score:
                    existing.isbn = candidate.isbn or existing.isbn
                    existing.is_english = candidate.is_english or existing.is_english
                    existing.publisher_tier = min(candidate.publisher_tier, existing.publisher_tier)
                return

        self.candidates.append(candidate)

        # Keep only top N candidates
        if len(self.candidates) > self.max_candidates:
            self.candidates.sort(key=lambda c: c.quality_score, reverse=True)
            self.candidates = self.candidates[:self.max_candidates]

    @property
    def isbns(self) -> List[str]:
        """Return list of ISBNs sorted by quality."""
        sorted_candidates = sorted(self.candidates, key=lambda c: c.quality_score, reverse=True)
        return [c.isbn for c in sorted_candidates if c.isbn]

    @property
    def cover_ids(self) -> List[int]:
        """Return list of cover IDs sorted by quality."""
        sorted_candidates = sorted(self.candidates, key=lambda c: c.quality_score, reverse=True)
        return [c.cover_id for c in sorted_candidates if c.cover_id]

    @property
    def best_isbn(self) -> Optional[str]:
        isbns = self.isbns
        return isbns[0] if isbns else None

    @property
    def best_cover_id(self) -> Optional[int]:
        cover_ids = self.cover_ids
        return cover_ids[0] if cover_ids else None


ENGLISH_CODES = {'eng', 'en', 'english', 'en-us', 'en-gb', 'en-au', 'en-ca'}

# Major publishers with reliable cover images (prioritized)
QUALITY_PUBLISHERS = {
    'penguin', 'random house', 'penguin random house', 'vintage', 'knopf', 'doubleday',
    'harpercollins', 'harper collins', 'harper', 'william morrow', 'avon',
    'simon & schuster', 'simon and schuster', 'scribner', 'atria', 'pocket books',
    'macmillan', 'st. martin', "st martin's", 'tor', 'farrar', 'henry holt', 'flatiron',
    'hachette', 'little, brown', 'little brown', 'grand central', 'orbit',
    'scholastic', 'disney', 'marvel', 'dc comics',
    'oxford university press', 'cambridge university press',
    'bloomsbury', 'faber', 'hodder', 'pan macmillan',
    'viking', 'dutton', 'putnam', 'berkley', 'ace', 'roc', 'daw',
    'del rey', 'bantam', 'ballantine', 'crown', 'clarkson potter',
    'anchor', 'pantheon', 'everyman', 'modern library',
    'houghton mifflin', 'harcourt', 'little simon',
    'chronicle books', 'workman', 'quirk books',
    'abrams', 'thames & hudson', 'phaidon', 'taschen',
    'wiley', 'mcgraw-hill', 'mcgraw hill', 'pearson', 'cengage',
    'norton', 'w.w. norton', 'basic books', 'princeton university press',
    'yale university press', 'harvard university press', 'mit press',
}

# Self-publishing / POD platforms (deprioritized - often poor covers)
SELFPUB_PUBLISHERS = {
    'independently published', 'independent', 'self-published', 'self published',
    'createspace', 'kindle direct', 'kdp', 'amazon publishing',
    'lulu', 'lulu.com', 'lulu press', 'blurb', 'smashwords',
    'xlibris', 'authorhouse', 'iuniverse', 'trafford', 'westbow',
    'balboa press', 'archway', 'partridge', 'palibrio',
    'grin verlag', 'grin publishing', 'scholar\'s choice', 'scholars choice',
    'franklin classics', 'creative media partners', 'wentworth press',
    'forgotten books', 'kessinger', 'nabu press', 'bibliobazaar', 'bibliolife',
    'general books', 'hardpress', 'lightning source', 'book on demand',
    '[s.n.]', 's.n.', 's.n', '[publisher not identified]', 'unknown',
}


def get_publisher_tier(publisher: Optional[str]) -> int:
    """
    Return publisher tier (lower = better).
    0 = Quality publisher, 1 = Normal, 2 = Self-pub/POD
    """
    if not publisher:
        return 1
    pub_lower = publisher.lower()

    for quality in QUALITY_PUBLISHERS:
        if quality in pub_lower:
            return 0

    for selfpub in SELFPUB_PUBLISHERS:
        if selfpub in pub_lower:
            return 2

    return 1


def extract_language(data: dict) -> Optional[str]:
    """Extract language code from edition data."""
    languages = data.get('languages', [])
    if not languages:
        return None
    first_lang = languages[0]
    if isinstance(first_lang, dict):
        key = first_lang.get('key', '')
        return key.replace('/languages/', '').lower()
    elif isinstance(first_lang, str):
        return first_lang.replace('/languages/', '').lower()
    return None


def extract_publish_year(data: dict) -> Optional[int]:
    """Extract publication year from edition data."""
    # Try publish_date first (most common)
    publish_date = data.get('publish_date', '')
    if publish_date:
        # Try to find a 4-digit year in the string
        match = re.search(r'\b(19|20)\d{2}\b', str(publish_date))
        if match:
            return int(match.group())

    # Try copyright_date
    copyright_date = data.get('copyright_date', '')
    if copyright_date:
        match = re.search(r'\b(19|20)\d{2}\b', str(copyright_date))
        if match:
            return int(match.group())

    return None


def resolve_isbns(editions_path: Path, top_work_ids: Set[str],
                  english_only: bool = True, max_candidates: int = 10) -> Dict[str, WorkEditions]:
    """
    Find multiple ISBNs and cover_ids for each top work.
    Collects up to max_candidates editions per work, ranked by quality.
    Returns: {work_id: WorkEditions}
    """
    print("\n" + "=" * 70)
    print("STEP C: RESOLVING ISBNs AND COVER IDs")
    print("=" * 70)
    print(f"Input: {editions_path}")
    print(f"Looking for: {len(top_work_ids):,} work IDs")
    print(f"Max Candidates per Work: {max_candidates}")
    print(f"Prioritizing: English + Cover + Quality Publishers")

    editions: Dict[str, WorkEditions] = {}
    processed = 0

    with open_file(editions_path) as f:
        for line in f:
            processed += 1
            if progress(processed):
                found = sum(1 for e in editions.values() if e.candidates)
                total_candidates = sum(len(e.candidates) for e in editions.values())
                print(f"  Processed: {processed:,} | Works Found: {found:,} | Total Candidates: {total_candidates:,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                data = json.loads(parts[4])

                works = data.get('works', [])
                work_id = None
                for w in works:
                    if isinstance(w, dict):
                        wkey = w.get('key', '')
                        work_id = extract_work_id(wkey)
                    elif isinstance(w, str):
                        work_id = extract_work_id(w)
                    break

                if not work_id or work_id not in top_work_ids:
                    continue

                lang = extract_language(data)
                is_english = lang in ENGLISH_CODES if lang else False

                isbn13_list = data.get('isbn_13', [])
                isbn10_list = data.get('isbn_10', [])
                isbn = isbn13_list[0] if isbn13_list else (isbn10_list[0] if isbn10_list else None)

                covers = data.get('covers', [])
                cover_id = None
                has_cover = False
                if covers:
                    for c in covers:
                        if isinstance(c, int) and c > 0:
                            cover_id = c
                            has_cover = True
                            break

                publishers = data.get('publishers', [])
                publisher = publishers[0] if publishers else None
                publisher_tier = get_publisher_tier(publisher)

                publish_year = extract_publish_year(data)

                candidate = EditionCandidate(
                    isbn=isbn,
                    cover_id=cover_id,
                    has_cover=has_cover,
                    is_english=is_english,
                    publisher_tier=publisher_tier,
                    publish_year=publish_year
                )

                if work_id not in editions:
                    editions[work_id] = WorkEditions(max_candidates=max_candidates)

                editions[work_id].add_candidate(candidate)

            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    # Fill missing works with empty WorkEditions
    for work_id in top_work_ids:
        if work_id not in editions:
            editions[work_id] = WorkEditions(max_candidates=max_candidates)

    # Stats
    works_with_editions = sum(1 for e in editions.values() if e.candidates)
    total_candidates = sum(len(e.candidates) for e in editions.values())
    with_isbn = sum(1 for e in editions.values() if e.best_isbn)
    with_cover_id = sum(1 for e in editions.values() if e.best_cover_id)
    multi_isbn = sum(1 for e in editions.values() if len(e.isbns) > 1)
    avg_candidates = total_candidates / max(works_with_editions, 1)

    print(f"\nWorks Found:       {works_with_editions:,}/{len(top_work_ids):,} ({100*works_with_editions/len(top_work_ids):.1f}%)")
    print(f"Total Candidates:  {total_candidates:,} (avg {avg_candidates:.1f}/work)")
    print(f"With ISBN:         {with_isbn:,} ({100*with_isbn/max(works_with_editions,1):.1f}%)")
    print(f"With Cover ID:     {with_cover_id:,} ({100*with_cover_id/max(works_with_editions,1):.1f}%)")
    print(f"Multiple ISBNs:    {multi_isbn:,} ({100*multi_isbn/max(works_with_editions,1):.1f}%)")

    return editions


# =============================================================================
# STEP D: AUTHOR NAME RESOLUTION
# =============================================================================

def resolve_author_names(authors_path: Path, author_ids: Set[str]) -> Dict[str, str]:
    """
    Resolve author IDs to names from the authors dump.
    Returns: {author_id: author_name}
    """
    print("\n" + "=" * 70)
    print("STEP D: RESOLVING AUTHOR NAMES")
    print("=" * 70)
    print(f"Input: {authors_path}")
    print(f"Looking for: {len(author_ids):,} author IDs")

    names: Dict[str, str] = {}
    processed = 0
    found = 0

    with open_file(authors_path) as f:
        for line in f:
            processed += 1
            if progress(processed):
                print(f"  Processed: {processed:,} | Found: {found:,}/{len(author_ids):,}")

            if found >= len(author_ids):
                print(f"  Early exit: All {len(author_ids):,} authors found")
                break

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                author_id = key.replace('/authors/', '')

                if author_id not in author_ids:
                    continue

                data = json.loads(parts[4])
                name = data.get('name', '')

                if name:
                    name = re.sub(r'[\n\r\t]+', ' ', str(name))
                    name = re.sub(r'\s+', ' ', name).strip()
                    names[author_id] = name[:200]
                    found += 1

            except (json.JSONDecodeError, IndexError):
                continue

    print(f"\nAuthors Found: {found:,}/{len(author_ids):,} ({100*found/max(len(author_ids),1):.1f}%)")
    return names


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_output(top_works: List[Tuple[str, WorkStats]],
                    metadata: Dict[str, WorkMetadata],
                    editions: Dict[str, WorkEditions],
                    author_names: Dict[str, str],
                    output_path: Path):
    """
    Generate the final CSV output.
    ISBNs, cover_ids, author_ids, authors, and subjects are pipe-separated.
    """
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT")
    print("=" * 70)
    print(f"Output: {output_path}")

    columns = [
        'rank', 'work_id', 'title', 'subtitle', 'description',
        'authors', 'author_ids', 'subjects',
        'first_publish_year', 'series_name', 'series_position',
        'isbns', 'cover_ids',
        'rating_avg', 'rating_count',
        'want_to_read', 'currently_reading', 'already_read', 'interaction_count'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        written = 0
        missing_titles = 0
        missing_isbns = 0
        missing_covers = 0
        missing_desc = 0
        missing_authors = 0
        total_isbns = 0
        total_covers = 0

        for rank, (work_id, stats) in enumerate(top_works, 1):
            work_meta = metadata.get(work_id, WorkMetadata())
            work_editions = editions.get(work_id)

            isbns_list = work_editions.isbns if work_editions else []
            cover_ids_list = work_editions.cover_ids if work_editions else []

            isbns_str = '|'.join(isbns_list) if isbns_list else ''
            cover_ids_str = '|'.join(str(c) for c in cover_ids_list) if cover_ids_list else ''

            # Resolve author names from IDs
            author_ids_list = work_meta.author_ids if work_meta.author_ids else []
            author_names_list = [author_names.get(aid, '') for aid in author_ids_list]
            # Filter out empty names but keep order
            author_names_list = [n for n in author_names_list if n]

            author_ids_str = '|'.join(author_ids_list) if author_ids_list else ''
            authors_str = '|'.join(author_names_list) if author_names_list else ''
            subjects_str = '|'.join(work_meta.subjects) if work_meta.subjects else ''

            if not work_meta.title:
                missing_titles += 1
            if not work_meta.description:
                missing_desc += 1
            if not isbns_list:
                missing_isbns += 1
            if not cover_ids_list:
                missing_covers += 1
            if not author_names_list:
                missing_authors += 1

            total_isbns += len(isbns_list)
            total_covers += len(cover_ids_list)

            writer.writerow({
                'rank': rank,
                'work_id': work_id,
                'title': work_meta.title or '',
                'subtitle': work_meta.subtitle or '',
                'description': work_meta.description or '',
                'authors': authors_str,
                'author_ids': author_ids_str,
                'subjects': subjects_str,
                'first_publish_year': work_meta.first_publish_year or '',
                'series_name': work_meta.series_name or '',
                'series_position': work_meta.series_position or '',
                'isbns': isbns_str,
                'cover_ids': cover_ids_str,
                'rating_avg': stats.avg_rating or '',
                'rating_count': stats.rating_count,
                'want_to_read': stats.want_to_read,
                'currently_reading': stats.currently_reading,
                'already_read': stats.already_read,
                'interaction_count': stats.interactions,
            })
            written += 1

    avg_isbns = total_isbns / max(written - missing_isbns, 1)
    avg_covers = total_covers / max(written - missing_covers, 1)

    print(f"\nRecords Written:   {written:,}")
    print(f"Missing Titles:    {missing_titles:,}")
    print(f"Missing Desc:      {missing_desc:,}")
    print(f"Missing Authors:   {missing_authors:,}")
    print(f"Missing ISBNs:     {missing_isbns:,}")
    print(f"Missing Cover IDs: {missing_covers:,}")
    print(f"Total ISBNs:       {total_isbns:,} (avg {avg_isbns:.1f}/work)")
    print(f"Total Cover IDs:   {total_covers:,} (avg {avg_covers:.1f}/work)")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File Size: {size_mb:.2f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract Top N Most Popular Books from Open Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_top_books.py                     # Default: Top 100,000
  python extract_top_books.py --top 50000         # Top 50,000
  python extract_top_books.py --output best.csv   # Custom output name
        """
    )

    parser.add_argument('--top', type=int, default=DEFAULT_TOP_N,
                        help=f'Number of top books to extract (default: {DEFAULT_TOP_N:,})')
    parser.add_argument('--sort-by', choices=['rating_count', 'popularity', 'avg_rating'],
                        default='rating_count',
                        help='Sort criteria: rating_count (most rated), popularity, avg_rating')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV path (default: output/top_Nk_books.csv)')
    parser.add_argument('--input-dir', type=Path, default=Path('.'),
                        help='Input directory containing dump files')

    args = parser.parse_args()

    if args.output is None:
        k = args.top // 1000
        args.output = OUTPUT_DIR / f"top_{k}k_books.csv"

    args.output.parent.mkdir(exist_ok=True)

    print("=" * 70)
    print("TOP BOOKS EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target:     Top {args.top:,} books")
    print(f"Sort By:    {args.sort_by}")
    print(f"Output:     {args.output}")

    start_time = datetime.now()

    # Find input files
    ratings_path = find_dump_file("ol_dump_ratings", args.input_dir)
    reading_log_path = find_dump_file("ol_dump_reading", args.input_dir)
    works_path = find_dump_file("ol_dump_works", args.input_dir)
    editions_path = find_dump_file("ol_dump_editions", args.input_dir)
    authors_path = find_dump_file("ol_dump_authors", args.input_dir)

    print(f"\nInput Files:")
    print(f"  Ratings:     {ratings_path or 'NOT FOUND'}")
    print(f"  Reading Log: {reading_log_path or 'NOT FOUND'}")
    print(f"  Works:       {works_path or 'NOT FOUND'}")
    print(f"  Editions:    {editions_path or 'NOT FOUND'}")
    print(f"  Authors:     {authors_path or 'NOT FOUND'}")

    if not works_path or not editions_path:
        print("\nERROR: Required dump files not found!")
        sys.exit(1)

    # Step A: Aggregate popularity
    work_stats = aggregate_popularity(ratings_path, reading_log_path)

    if not work_stats:
        print("\nERROR: No popularity data found. Check ratings/reading log files.")
        sys.exit(1)

    # Select top N
    top_works = select_top_works(work_stats, args.top, sort_by=args.sort_by)
    top_work_ids = {w[0] for w in top_works}

    print(f"\nSelected {len(top_works):,} works for enrichment")

    # Step B: Enrich metadata (titles, descriptions, series, authors, subjects)
    metadata, all_author_ids = enrich_metadata(works_path, top_work_ids)

    # Step C: Resolve ISBNs
    isbns = resolve_isbns(editions_path, top_work_ids)

    # Step D: Resolve author names
    author_names: Dict[str, str] = {}
    if authors_path and all_author_ids:
        author_names = resolve_author_names(authors_path, all_author_ids)
    else:
        print("\nWARNING: Authors dump not found. Author names will be empty.")

    # Generate output
    generate_output(top_works, metadata, isbns, author_names, args.output)

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time: {elapsed}")
    print(f"Output File:  {args.output}")

    # Preview
    print("\nTop 10 Preview:")
    print("-" * 100)
    with open(args.output, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            title = row['title'][:28] + '..' if row['title'] and len(row['title']) > 28 else row['title']
            author = row.get('authors', '').split('|')[0][:20] if row.get('authors') else ''
            year = row.get('first_publish_year', '')
            # Flags: D=description, S=series, A=authors, G=subjects(genres)
            flags = ""
            flags += "D" if row.get('description') else "-"
            flags += "S" if row.get('series_name') else "-"
            flags += "A" if row.get('authors') else "-"
            flags += "G" if row.get('subjects') else "-"
            print(f"  {row['rank']:>5}. {title:<30} by {author:<20} {year:>4} ({row['rating_count']:>5}) [{flags}]")


if __name__ == "__main__":
    main()
