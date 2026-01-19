"""
Open Library Data Pattern Analyzer
===================================
High-performance EDA script for probing Open Library dumps.
Identifies patterns for Enum definitions, Quality Thresholds, and Normalization Rules.

Usage:
    python analyze_library_patterns.py --works ol_dump_works.txt.gz --editions ol_dump_editions.txt.gz --authors ol_dump_authors.txt.gz --limit 1000000
"""

import argparse
import gzip
import json
import re
import sys
import statistics
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Any, List
from datetime import datetime


@dataclass
class EditionsStats:
    """Collector for Editions dump analysis."""
    total: int = 0
    formats: Counter = field(default_factory=Counter)
    publishers: Counter = field(default_factory=Counter)
    page_counts: List[int] = field(default_factory=list)

    has_isbn13: int = 0
    has_isbn10: int = 0
    has_covers: int = 0
    has_publish_date: int = 0
    has_physical_format: int = 0
    has_publishers: int = 0
    has_pages: int = 0

    languages: Counter = field(default_factory=Counter)

    page_sample_limit: int = 100000

    def record_page_count(self, pages: int):
        if len(self.page_counts) < self.page_sample_limit:
            self.page_counts.append(pages)


@dataclass
class WorksStats:
    """Collector for Works dump analysis."""
    total: int = 0

    has_title: int = 0
    has_subtitle: int = 0
    has_description: int = 0
    has_covers: int = 0
    has_subjects: int = 0
    has_authors: int = 0
    has_series_field: int = 0

    description_is_string: int = 0
    description_is_dict: int = 0

    series_is_string: int = 0
    series_is_list: int = 0

    title_has_series_pattern: int = 0

    subject_counts: List[int] = field(default_factory=list)
    author_counts: List[int] = field(default_factory=list)

    sample_limit: int = 100000

    SERIES_PATTERN = re.compile(
        r'(?:'
        r'\(([^)]+?),?\s*#\s*\d+\)|'
        r'\(([^)]+?)\s+(?:Book|Vol\.?|Part|Tome)\s+\d+\)|'
        r',\s*(?:tome|vol\.?|volume|book|part|#|no\.?)\s*\d+$'
        r')',
        re.IGNORECASE
    )


@dataclass
class AuthorsStats:
    """Collector for Authors dump analysis."""
    total: int = 0

    has_name: int = 0
    has_bio: int = 0
    has_birth_date: int = 0

    name_only: int = 0
    sparse: int = 0
    rich: int = 0
    has_death_date: int = 0
    has_wikidata: int = 0
    has_wikipedia: int = 0
    has_photos: int = 0
    has_links: int = 0
    has_alternate_names: int = 0

    bio_is_string: int = 0
    bio_is_dict: int = 0

    birth_date_formats: Counter = field(default_factory=Counter)

    name_lengths: List[int] = field(default_factory=list)
    sample_limit: int = 100000


class StreamReader:
    """O(1) memory streaming reader for GZIP TSV files."""

    def __init__(self, file_path: Path, limit: int = 0):
        self.file_path = file_path
        self.limit = limit

    def records(self):
        count = 0
        with gzip.open(self.file_path, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                if self.limit > 0 and count >= self.limit:
                    break
                try:
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        key = parts[1].strip()
                        data = json.loads(parts[4])
                        yield key, data
                        count += 1
                except (json.JSONDecodeError, IndexError):
                    continue


def analyze_editions(file_path: Path, limit: int) -> EditionsStats:
    """Analyze the Editions dump for format, publisher, and page patterns."""
    stats = EditionsStats()
    reader = StreamReader(file_path, limit)

    print(f"\n[Editions] Analyzing {file_path.name}...")

    for key, data in reader.records():
        stats.total += 1

        if stats.total % 100000 == 0:
            print(f"  Processed: {stats.total:,}")

        physical_format = data.get('physical_format')
        if physical_format:
            stats.has_physical_format += 1
            if isinstance(physical_format, str):
                stats.formats[physical_format.strip().lower()] += 1

        publishers = data.get('publishers', [])
        if publishers and isinstance(publishers, list):
            stats.has_publishers += 1
            for pub in publishers[:3]:
                if isinstance(pub, str):
                    stats.publishers[pub.strip()] += 1

        pages = data.get('number_of_pages')
        if pages:
            stats.has_pages += 1
            try:
                page_int = int(pages)
                if 1 <= page_int <= 50000:
                    stats.record_page_count(page_int)
            except (ValueError, TypeError):
                pass

        isbn13 = data.get('isbn_13')
        if isbn13 and isinstance(isbn13, list) and len(isbn13) > 0:
            stats.has_isbn13 += 1

        isbn10 = data.get('isbn_10')
        if isbn10 and isinstance(isbn10, list) and len(isbn10) > 0:
            stats.has_isbn10 += 1

        covers = data.get('covers')
        if covers and isinstance(covers, list) and len(covers) > 0:
            stats.has_covers += 1

        pub_date = data.get('publish_date')
        if pub_date:
            stats.has_publish_date += 1

        langs = data.get('languages', [])
        if isinstance(langs, list):
            for lang in langs:
                if isinstance(lang, dict):
                    lang_key = lang.get('key', '')
                    if lang_key:
                        stats.languages[lang_key.replace('/languages/', '')] += 1

    return stats


def analyze_works(file_path: Path, limit: int) -> WorksStats:
    """Analyze the Works dump for metadata patterns."""
    stats = WorksStats()
    reader = StreamReader(file_path, limit)

    print(f"\n[Works] Analyzing {file_path.name}...")

    for key, data in reader.records():
        stats.total += 1

        if stats.total % 100000 == 0:
            print(f"  Processed: {stats.total:,}")

        title = data.get('title')
        if title and isinstance(title, str) and title.strip():
            stats.has_title += 1
            if stats.SERIES_PATTERN.search(title):
                stats.title_has_series_pattern += 1

        subtitle = data.get('subtitle')
        if subtitle:
            stats.has_subtitle += 1

        description = data.get('description')
        if description:
            stats.has_description += 1
            if isinstance(description, str):
                stats.description_is_string += 1
            elif isinstance(description, dict):
                stats.description_is_dict += 1

        covers = data.get('covers')
        if covers and isinstance(covers, list) and len(covers) > 0:
            stats.has_covers += 1

        subjects = data.get('subjects', [])
        if subjects and isinstance(subjects, list) and len(subjects) > 0:
            stats.has_subjects += 1
            if len(stats.subject_counts) < stats.sample_limit:
                stats.subject_counts.append(len(subjects))

        authors = data.get('authors', [])
        if authors and isinstance(authors, list) and len(authors) > 0:
            stats.has_authors += 1
            if len(stats.author_counts) < stats.sample_limit:
                stats.author_counts.append(len(authors))

        series = data.get('series')
        if series:
            stats.has_series_field += 1
            if isinstance(series, str):
                stats.series_is_string += 1
            elif isinstance(series, list):
                stats.series_is_list += 1

    return stats


def analyze_authors(file_path: Path, limit: int) -> AuthorsStats:
    """Analyze the Authors dump for metadata density."""
    stats = AuthorsStats()
    reader = StreamReader(file_path, limit)

    print(f"\n[Authors] Analyzing {file_path.name}...")

    for key, data in reader.records():
        stats.total += 1

        if stats.total % 100000 == 0:
            print(f"  Processed: {stats.total:,}")

        name = data.get('name')
        if name and isinstance(name, str) and name.strip():
            stats.has_name += 1
            if len(stats.name_lengths) < stats.sample_limit:
                stats.name_lengths.append(len(name))

        bio = data.get('bio')
        if bio:
            stats.has_bio += 1
            if isinstance(bio, str):
                stats.bio_is_string += 1
            elif isinstance(bio, dict):
                stats.bio_is_dict += 1

        birth_date = data.get('birth_date')
        if birth_date:
            stats.has_birth_date += 1
            if isinstance(birth_date, str):
                bd = birth_date.strip()
                if re.match(r'^\d{4}$', bd):
                    stats.birth_date_formats['YYYY'] += 1
                elif re.match(r'^\d{4}-\d{2}-\d{2}', bd):
                    stats.birth_date_formats['YYYY-MM-DD'] += 1
                elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}', bd):
                    stats.birth_date_formats['MM/DD/YYYY'] += 1
                elif re.match(r'^[A-Za-z]+ \d{1,2}, \d{4}', bd):
                    stats.birth_date_formats['Month DD, YYYY'] += 1
                elif re.match(r'^\d{1,2} [A-Za-z]+ \d{4}', bd):
                    stats.birth_date_formats['DD Month YYYY'] += 1
                else:
                    stats.birth_date_formats['Other'] += 1

        death_date = data.get('death_date')
        if death_date:
            stats.has_death_date += 1

        remote_ids = data.get('remote_ids', {})
        if isinstance(remote_ids, dict):
            if remote_ids.get('wikidata'):
                stats.has_wikidata += 1

        if data.get('wikipedia'):
            stats.has_wikipedia += 1

        photos = data.get('photos')
        if photos and isinstance(photos, list) and len(photos) > 0:
            stats.has_photos += 1

        links = data.get('links')
        if links and isinstance(links, list) and len(links) > 0:
            stats.has_links += 1

        alt_names = data.get('alternate_names')
        if alt_names and isinstance(alt_names, list) and len(alt_names) > 0:
            stats.has_alternate_names += 1

        metadata_score = sum([
            bool(bio),
            bool(birth_date),
            bool(death_date),
            bool(remote_ids.get('wikidata') if isinstance(remote_ids, dict) else False),
            bool(data.get('wikipedia')),
            bool(photos and isinstance(photos, list) and len(photos) > 0),
            bool(alt_names and isinstance(alt_names, list) and len(alt_names) > 0),
        ])

        if metadata_score == 0:
            stats.name_only += 1
        elif metadata_score <= 2:
            stats.sparse += 1
        else:
            stats.rich += 1

    return stats


def pct(part: int, total: int) -> str:
    """Calculate percentage string."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def calculate_percentiles(data: List[int]) -> dict:
    """Calculate min, max, median, p90, p95, p99."""
    if not data:
        return {'min': 0, 'max': 0, 'median': 0, 'p90': 0, 'p95': 0, 'p99': 0}

    sorted_data = sorted(data)
    n = len(sorted_data)

    return {
        'min': sorted_data[0],
        'max': sorted_data[-1],
        'median': sorted_data[n // 2],
        'p90': sorted_data[int(n * 0.90)],
        'p95': sorted_data[int(n * 0.95)],
        'p99': sorted_data[int(n * 0.99)] if n > 100 else sorted_data[-1]
    }


def print_report(editions: Optional[EditionsStats], works: Optional[WorksStats], authors: Optional[AuthorsStats]):
    """Generate the structured analysis report."""

    print("\n")
    print("=" * 80)
    print(" " * 20 + "OPEN LIBRARY DATA PATTERN ANALYSIS")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if editions:
        print("\n")
        print("-" * 80)
        print("EDITIONS DUMP ANALYSIS")
        print("-" * 80)
        print(f"Total Records Sampled: {editions.total:,}")

        print("\n+- COMPLETENESS ---------------------------------------------+")
        print(f"|  ISBN-13:         {pct(editions.has_isbn13, editions.total):>8}  ({editions.has_isbn13:,} records)")
        print(f"|  ISBN-10:         {pct(editions.has_isbn10, editions.total):>8}  ({editions.has_isbn10:,} records)")
        print(f"|  Covers:          {pct(editions.has_covers, editions.total):>8}  ({editions.has_covers:,} records)")
        print(f"|  Publish Date:    {pct(editions.has_publish_date, editions.total):>8}  ({editions.has_publish_date:,} records)")
        print(f"|  Physical Format: {pct(editions.has_physical_format, editions.total):>8}  ({editions.has_physical_format:,} records)")
        print(f"|  Publishers:      {pct(editions.has_publishers, editions.total):>8}  ({editions.has_publishers:,} records)")
        print(f"|  Page Count:      {pct(editions.has_pages, editions.total):>8}  ({editions.has_pages:,} records)")
        print("+------------------------------------------------------------+")

        print("\n+- PHYSICAL FORMAT DISTRIBUTION (Top 50) --------------------+")
        print("|  Raw String                                    Count    %   |")
        print("|  ---------------------------------------------------------  |")
        for fmt, count in editions.formats.most_common(50):
            pct_val = (count / editions.total) * 100
            print(f"|  {fmt[:42]:<42} {count:>8,}  {pct_val:>5.1f}%  |")
        print("+------------------------------------------------------------+")

        print("\n+- FORMAT MAPPING RECOMMENDATIONS --------------------------+")
        paperback_terms = ['paperback', 'mass market', 'trade paperback', 'softcover', 'paper']
        hardcover_terms = ['hardcover', 'hardback', 'library binding', 'cloth']
        ebook_terms = ['ebook', 'e-book', 'kindle', 'digital', 'electronic', 'pdf', 'epub']
        audio_terms = ['audio', 'audiobook', 'cd', 'mp3', 'cassette']

        paperback_count = sum(c for f, c in editions.formats.items() if any(t in f for t in paperback_terms))
        hardcover_count = sum(c for f, c in editions.formats.items() if any(t in f for t in hardcover_terms))
        ebook_count = sum(c for f, c in editions.formats.items() if any(t in f for t in ebook_terms))
        audio_count = sum(c for f, c in editions.formats.items() if any(t in f for t in audio_terms))

        print(f"|  PAPERBACK matches: {paperback_count:>10,}  ({pct(paperback_count, editions.total):>6})")
        print(f"|  HARDCOVER matches: {hardcover_count:>10,}  ({pct(hardcover_count, editions.total):>6})")
        print(f"|  EBOOK matches:     {ebook_count:>10,}  ({pct(ebook_count, editions.total):>6})")
        print(f"|  AUDIO matches:     {audio_count:>10,}  ({pct(audio_count, editions.total):>6})")
        print("+------------------------------------------------------------+")

        print("\n+- TOP 50 PUBLISHERS ----------------------------------------+")
        print("|  Publisher Name                                Count    %   |")
        print("|  ---------------------------------------------------------  |")
        for pub, count in editions.publishers.most_common(50):
            pct_val = (count / editions.total) * 100
            print(f"|  {pub[:42]:<42} {count:>8,}  {pct_val:>5.2f}%  |")
        print("+------------------------------------------------------------+")

        if editions.page_counts:
            page_stats = calculate_percentiles(editions.page_counts)
            print("\n+- PAGE COUNT DISTRIBUTION ---------------------------------+")
            print(f"|  Sample Size:  {len(editions.page_counts):,}")
            print(f"|  Min:          {page_stats['min']:,}")
            print(f"|  Median:       {page_stats['median']:,}")
            print(f"|  90th %ile:    {page_stats['p90']:,}")
            print(f"|  95th %ile:    {page_stats['p95']:,}")
            print(f"|  99th %ile:    {page_stats['p99']:,}")
            print(f"|  Max:          {page_stats['max']:,}")
            print("|")
            print("|  RECOMMENDATION: Sanity Window")
            print(f"|    Min pages: 20 (below this = pamphlet/noise)")
            print(f"|    Max pages: {page_stats['p99']} (99th percentile)")
            print("+------------------------------------------------------------+")

        print("\n+- TOP 20 LANGUAGES -----------------------------------------+")
        for lang, count in editions.languages.most_common(20):
            pct_val = (count / editions.total) * 100
            print(f"|  {lang:<10} {count:>10,}  ({pct_val:>5.1f}%)")
        print("+------------------------------------------------------------+")

    if works:
        print("\n")
        print("-" * 80)
        print("WORKS DUMP ANALYSIS")
        print("-" * 80)
        print(f"Total Records Sampled: {works.total:,}")

        print("\n+- COMPLETENESS ---------------------------------------------+")
        print(f"|  Title:        {pct(works.has_title, works.total):>8}  ({works.has_title:,} records)")
        print(f"|  Subtitle:     {pct(works.has_subtitle, works.total):>8}  ({works.has_subtitle:,} records)")
        print(f"|  Description:  {pct(works.has_description, works.total):>8}  ({works.has_description:,} records)")
        print(f"|  Covers:       {pct(works.has_covers, works.total):>8}  ({works.has_covers:,} records)")
        print(f"|  Subjects:     {pct(works.has_subjects, works.total):>8}  ({works.has_subjects:,} records)")
        print(f"|  Authors:      {pct(works.has_authors, works.total):>8}  ({works.has_authors:,} records)")
        print("+------------------------------------------------------------+")

        print("\n+- SERIES PATTERN ANALYSIS ----------------------------------+")
        print(f"|  Explicit 'series' field present:  {pct(works.has_series_field, works.total):>8}  ({works.has_series_field:,})")
        if works.has_series_field > 0:
            print(f"|    -> String type:  {pct(works.series_is_string, works.has_series_field):>8}  ({works.series_is_string:,})")
            print(f"|    -> List type:    {pct(works.series_is_list, works.has_series_field):>8}  ({works.series_is_list:,})")
        print(f"|")
        print(f"|  Title matches series regex:       {pct(works.title_has_series_pattern, works.total):>8}  ({works.title_has_series_pattern:,})")
        print(f"|")
        print(f"|  INSIGHT: {works.title_has_series_pattern - works.has_series_field:,} works have series info")
        print(f"|           in title but NO explicit series field!")
        print("+------------------------------------------------------------+")

        print("\n+- DESCRIPTION POLYMORPHISM ---------------------------------+")
        if works.has_description > 0:
            print(f"|  Total with description:  {works.has_description:,}")
            print(f"|    -> Simple String:  {pct(works.description_is_string, works.has_description):>8}  ({works.description_is_string:,})")
            print(f"|    -> Dict (/type/text): {pct(works.description_is_dict, works.has_description):>8}  ({works.description_is_dict:,})")
        else:
            print("|  No descriptions found in sample.")
        print("+------------------------------------------------------------+")

        if works.subject_counts:
            subj_stats = calculate_percentiles(works.subject_counts)
            print("\n+- SUBJECTS PER WORK ----------------------------------------+")
            print(f"|  Median:     {subj_stats['median']:,} subjects")
            print(f"|  90th %ile:  {subj_stats['p90']:,} subjects")
            print(f"|  Max:        {subj_stats['max']:,} subjects")
            print("+------------------------------------------------------------+")

        if works.author_counts:
            auth_stats = calculate_percentiles(works.author_counts)
            print("\n+- AUTHORS PER WORK -----------------------------------------+")
            print(f"|  Median:     {auth_stats['median']:,} authors")
            print(f"|  90th %ile:  {auth_stats['p90']:,} authors")
            print(f"|  Max:        {auth_stats['max']:,} authors")
            print("+------------------------------------------------------------+")

    if authors:
        print("\n")
        print("-" * 80)
        print("AUTHORS DUMP ANALYSIS")
        print("-" * 80)
        print(f"Total Records Sampled: {authors.total:,}")

        print("\n+- METADATA DENSITY -----------------------------------------+")
        print(f"|  Has Name:            {pct(authors.has_name, authors.total):>8}  ({authors.has_name:,})")
        print(f"|  Has Bio:             {pct(authors.has_bio, authors.total):>8}  ({authors.has_bio:,})")
        print(f"|  Has Birth Date:      {pct(authors.has_birth_date, authors.total):>8}  ({authors.has_birth_date:,})")
        print(f"|  Has Death Date:      {pct(authors.has_death_date, authors.total):>8}  ({authors.has_death_date:,})")
        print(f"|  Has Wikidata ID:     {pct(authors.has_wikidata, authors.total):>8}  ({authors.has_wikidata:,})")
        print(f"|  Has Wikipedia Link:  {pct(authors.has_wikipedia, authors.total):>8}  ({authors.has_wikipedia:,})")
        print(f"|  Has Photos:          {pct(authors.has_photos, authors.total):>8}  ({authors.has_photos:,})")
        print(f"|  Has External Links:  {pct(authors.has_links, authors.total):>8}  ({authors.has_links:,})")
        print(f"|  Has Alternate Names: {pct(authors.has_alternate_names, authors.total):>8}  ({authors.has_alternate_names:,})")
        print("+------------------------------------------------------------+")

        print("\n+- BIO POLYMORPHISM -----------------------------------------+")
        if authors.has_bio > 0:
            print(f"|  Total with bio:      {authors.has_bio:,}")
            print(f"|    -> Simple String:   {pct(authors.bio_is_string, authors.has_bio):>8}  ({authors.bio_is_string:,})")
            print(f"|    -> Dict (/type/text): {pct(authors.bio_is_dict, authors.has_bio):>8}  ({authors.bio_is_dict:,})")
        else:
            print("|  No bios found in sample.")
        print("+------------------------------------------------------------+")

        print("\n+- BIRTH DATE FORMAT DISTRIBUTION --------------------------+")
        for fmt, count in authors.birth_date_formats.most_common():
            print(f"|  {fmt:<20} {count:>8,}  ({pct(count, authors.has_birth_date):>6})")
        print("+------------------------------------------------------------+")

        if authors.name_lengths:
            name_stats = calculate_percentiles(authors.name_lengths)
            print("\n+- NAME LENGTH DISTRIBUTION ---------------------------------+")
            print(f"|  Median:     {name_stats['median']:,} chars")
            print(f"|  95th %ile:  {name_stats['p95']:,} chars")
            print(f"|  Max:        {name_stats['max']:,} chars")
            print("+------------------------------------------------------------+")

        print("\n+- AUTHOR QUALITY TIERS -------------------------------------+")
        print(f"|  NAME-ONLY (0 metadata fields):   {pct(authors.name_only, authors.total):>8}  ({authors.name_only:,})")
        print(f"|  SPARSE (1-2 metadata fields):    {pct(authors.sparse, authors.total):>8}  ({authors.sparse:,})")
        print(f"|  RICH (3+ metadata fields):       {pct(authors.rich, authors.total):>8}  ({authors.rich:,})")
        print("+------------------------------------------------------------+")

        print("\n+- DROP DECISION ANALYSIS -----------------------------------+")
        print(f"|")
        print(f"|  QUESTION: Should I drop authors with no metadata?")
        print(f"|")
        print(f"|  DATA POINT: {pct(authors.name_only, authors.total)} ({authors.name_only:,}) have ONLY a name.")
        print(f"|")
        droppable = authors.name_only
        print(f"|  IF YOU DROP NAME-ONLY AUTHORS:")
        print(f"|    -> You eliminate {droppable:,} sparse records")
        print(f"|    -> You keep {authors.total - droppable:,} authors with at least 1 metadata field")
        print(f"|    -> Potential data loss: Some indie/self-pub authors have no metadata")
        print(f"|")
        print(f"|  RECOMMENDATION:")
        if authors.name_only / authors.total > 0.7:
            print(f"|    [!]  HIGH SPARSITY ({pct(authors.name_only, authors.total)} name-only)")
            print(f"|    -> Consider keeping all, filter at query time by works linkage")
        elif authors.name_only / authors.total > 0.4:
            print(f"|    [*] MODERATE SPARSITY ({pct(authors.name_only, authors.total)} name-only)")
            print(f"|    -> Safe to drop name-only, minimal quality impact")
        else:
            print(f"|    [OK]  LOW SPARSITY ({pct(authors.name_only, authors.total)} name-only)")
            print(f"|    -> Safe to drop name-only authors")
        print(f"|")
        print(f"|  ALTERNATIVE: Keep all authors, add 'quality_tier' column for filtering")
        print("+------------------------------------------------------------+")

        print("\n+- ENRICHMENT RECOMMENDATIONS -------------------------------+")
        print(f"|  Only {pct(authors.has_wikidata, authors.total)} have Wikidata IDs.")
        print(f"|  Consider external enrichment pipeline for 'rich' authors.")
        print("+------------------------------------------------------------+")

    print("\n")
    print("=" * 80)
    print(" " * 25 + "END OF ANALYSIS REPORT")
    print("=" * 80)


def find_dump_file(pattern: str, directory: Path = Path(".")) -> Optional[Path]:
    """Find dump file matching pattern."""
    candidates = list(directory.glob(f"{pattern}*.txt.gz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Open Library data dump patterns for ETL pipeline design.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_library_patterns.py --limit 100000
  python analyze_library_patterns.py --works my_works.txt.gz --limit 500000
  python analyze_library_patterns.py --editions editions.txt.gz --authors authors.txt.gz
        """
    )

    parser.add_argument('--works', type=Path, help='Path to works dump file')
    parser.add_argument('--editions', type=Path, help='Path to editions dump file')
    parser.add_argument('--authors', type=Path, help='Path to authors dump file')
    parser.add_argument('--limit', type=int, default=1_000_000,
                        help='Maximum records to analyze per dump (default: 1,000,000)')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-detect dump files in current directory')

    args = parser.parse_args()

    works_path = args.works
    editions_path = args.editions
    authors_path = args.authors

    if args.auto or (not works_path and not editions_path and not authors_path):
        print("Auto-detecting dump files...")
        if not works_path:
            works_path = find_dump_file("ol_dump_works")
        if not editions_path:
            editions_path = find_dump_file("ol_dump_editions")
        if not authors_path:
            authors_path = find_dump_file("ol_dump_authors")

    print("\n" + "=" * 80)
    print(" " * 15 + "OPEN LIBRARY DATA PATTERN ANALYZER")
    print("=" * 80)
    print(f"Limit: {args.limit:,} records per dump")
    print(f"Works:    {works_path if works_path and works_path.exists() else 'Not found'}")
    print(f"Editions: {editions_path if editions_path and editions_path.exists() else 'Not found'}")
    print(f"Authors:  {authors_path if authors_path and authors_path.exists() else 'Not found'}")

    editions_stats = None
    works_stats = None
    authors_stats = None

    if editions_path and editions_path.exists():
        editions_stats = analyze_editions(editions_path, args.limit)

    if works_path and works_path.exists():
        works_stats = analyze_works(works_path, args.limit)

    if authors_path and authors_path.exists():
        authors_stats = analyze_authors(authors_path, args.limit)

    if not any([editions_stats, works_stats, authors_stats]):
        print("\nERROR: No dump files found or specified.")
        print("Use --auto to auto-detect, or specify paths with --works, --editions, --authors")
        sys.exit(1)

    print_report(editions_stats, works_stats, authors_stats)


if __name__ == "__main__":
    main()
