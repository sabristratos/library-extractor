"""
Open Library Production ETL Pipeline
=====================================
Processes all Open Library dumps + Wikidata enrichment into normalized CSVs.

Usage:
    python etl_pipeline.py --all                    # Run complete pipeline
    python etl_pipeline.py --step wikidata          # Run single step
    python etl_pipeline.py --step authors works     # Run multiple steps
    python etl_pipeline.py --limit 100000           # Limit records per file

Author: Data Engineering Team
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("./output")
LOG_FILE = OUTPUT_DIR / "etl_errors.log"

COMPLETENESS_THRESHOLD = 40
PAGE_MIN = 20
PAGE_MAX = 1500

GENDER_MAP = {
    'Q6581097': 'male',
    'Q6581072': 'female',
    'Q1097630': 'intersex',
    'Q1052281': 'transgender_female',
    'Q2449503': 'transgender_male',
    'Q48270': 'non-binary',
}

FORMAT_RULES = {
    'paperback': ['paperback', 'soft', 'mass market', 'perfect paperback', 'softcover'],
    'hardcover': ['hardcover', 'hardback', 'bound', 'leather', 'library binding', 'casebound'],
    'ebook': ['ebook', 'e-book', 'kindle', 'epub', 'electronic', 'digital'],
    'audiobook': ['audio', 'cd', 'cassette', 'mp3', 'sound recording'],
}

SERIES_PATTERNS = [
    re.compile(r'\(([^)]+?),?\s*#\s*\d+\)', re.IGNORECASE),
    re.compile(r'\(([^)]+?)\s+Book\s+\d+\)', re.IGNORECASE),
    re.compile(r'\(([^)]+?)\s+Vol\.?\s*\d+\)', re.IGNORECASE),
    re.compile(r'\(([^)]+?)\s+Part\s+\d+\)', re.IGNORECASE),
    re.compile(r'\(([^)]+?)\s+Tome\s+\d+\)', re.IGNORECASE),
    re.compile(r'^(.+?),\s*(?:tome|vol\.?|volume|book|part|#|no\.?)\s*\d+', re.IGNORECASE),
    re.compile(r'^(.+?)\s+(?:tome|vol\.?|volume|book|part)\s+\d+', re.IGNORECASE),
]


# =============================================================================
# UTILITIES
# =============================================================================

@dataclass
class Stats:
    processed: int = 0
    kept: int = 0
    skipped: int = 0
    errors: int = 0


class ErrorLogger:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, 'w', encoding='utf-8')
        self.counts = Counter()

    def log(self, category: str, key: str, reason: str):
        self.counts[category] += 1
        self.file.write(f"{category}|{key}|{reason}\n")

    def close(self):
        self.file.close()

    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


def fix_mojibake(text: str) -> str:
    if not text:
        return text
    try:
        fixed = text.encode('latin-1').decode('utf-8')
        if fixed != text:
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    try:
        fixed = text.encode('cp1252').decode('utf-8')
        if fixed != text:
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def clean_text(text: Any, max_length: int = None) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, dict):
        text = text.get('value') or text.get('text') or ''
    text = str(text)
    text = fix_mojibake(text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]
    return text if text else None


def extract_id(key: str) -> str:
    if key.startswith('/authors/'):
        return key.replace('/authors/', '')
    if key.startswith('/works/'):
        return key.replace('/works/', '')
    if key.startswith('/books/'):
        return key.replace('/books/', '')
    return key


# =============================================================================
# POLYMORPHISM HANDLERS (Open Library Type/Value Pattern)
# =============================================================================

class TypeGuards:
    """
    Robust handlers for Open Library's polymorphic JSON structures.

    Common patterns:
    - String OR {"type": "/type/text", "value": "..."}
    - String OR {"key": "/authors/OL123A"}
    - List[str] OR List[{"key": "..."}]
    - int OR str (for dates, page counts)
    """

    @staticmethod
    def extract_text(data: Any, max_length: int = 1000) -> Optional[str]:
        """
        Extract text from polymorphic field.
        Handles: str, dict with 'value'/'text', dict with '/type/text'
        """
        if data is None:
            return None

        if isinstance(data, str):
            return clean_text(data, max_length)

        if isinstance(data, dict):
            if data.get('type') == '/type/text':
                return clean_text(data.get('value'), max_length)
            return clean_text(
                data.get('value') or data.get('text') or data.get('content'),
                max_length
            )

        if isinstance(data, list) and data:
            return TypeGuards.extract_text(data[0], max_length)

        return None

    @staticmethod
    def extract_key(data: Any) -> Optional[str]:
        """
        Extract key/ID from polymorphic reference.
        Handles: str, dict with 'key', nested dict
        """
        if data is None:
            return None

        if isinstance(data, str):
            return extract_id(data)

        if isinstance(data, dict):
            key = data.get('key')
            if key:
                return extract_id(key)
            author = data.get('author')
            if author:
                return TypeGuards.extract_key(author)
            work = data.get('work')
            if work:
                return TypeGuards.extract_key(work)

        return None

    @staticmethod
    def extract_keys(data: Any) -> List[str]:
        """
        Extract list of keys from polymorphic list field.
        Handles: List[str], List[dict], mixed
        """
        if data is None:
            return []

        if isinstance(data, str):
            return [extract_id(data)]

        if isinstance(data, dict):
            key = TypeGuards.extract_key(data)
            return [key] if key else []

        if isinstance(data, list):
            result = []
            for item in data:
                key = TypeGuards.extract_key(item)
                if key:
                    result.append(key)
            return result

        return []

    @staticmethod
    def extract_year(data: Any) -> Optional[int]:
        """
        Extract year from polymorphic date field.
        Handles: int, str with various formats, dict with 'value'
        """
        if data is None:
            return None

        if isinstance(data, int):
            if 1000 <= data <= 2030:
                return data
            return None

        if isinstance(data, dict):
            data = data.get('value') or data.get('time') or ''

        if isinstance(data, str):
            match = re.search(r'(\d{4})', data)
            if match:
                year = int(match.group(1))
                if 1000 <= year <= 2030:
                    return year

        return None

    @staticmethod
    def extract_int(data: Any, min_val: int = None, max_val: int = None) -> Optional[int]:
        """
        Extract integer from polymorphic field with optional bounds.
        Handles: int, str, dict with 'value'
        """
        if data is None:
            return None

        if isinstance(data, dict):
            data = data.get('value')

        try:
            value = int(data)
            if min_val is not None and value < min_val:
                return None
            if max_val is not None and value > max_val:
                return None
            return value
        except (ValueError, TypeError):
            return None

    @staticmethod
    def extract_first_string(data: Any) -> Optional[str]:
        """
        Extract first string from field that may be string or list.
        Handles: str, List[str], List[dict]
        """
        if data is None:
            return None

        if isinstance(data, str):
            return clean_text(data)

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, str):
                return clean_text(first)
            if isinstance(first, dict):
                return clean_text(first.get('value') or first.get('name') or first.get('text'))

        if isinstance(data, dict):
            return clean_text(data.get('value') or data.get('name') or data.get('text'))

        return None

    @staticmethod
    def extract_string_list(data: Any, max_items: int = None) -> List[str]:
        """
        Extract list of strings from polymorphic list field.
        Handles: str, List[str], List[dict]
        """
        if data is None:
            return []

        if isinstance(data, str):
            return [clean_text(data)] if clean_text(data) else []

        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    cleaned = clean_text(item)
                    if cleaned:
                        result.append(cleaned)
                elif isinstance(item, dict):
                    cleaned = clean_text(item.get('value') or item.get('name') or item.get('text'))
                    if cleaned:
                        result.append(cleaned)
                if max_items and len(result) >= max_items:
                    break
            return result

        return []

    @staticmethod
    def extract_language(data: Any) -> Optional[str]:
        """
        Extract language code from polymorphic language field.
        Handles: str, dict with 'key', List[dict]
        """
        if data is None:
            return None

        if isinstance(data, str):
            return data.replace('/languages/', '')[:10]

        if isinstance(data, dict):
            key = data.get('key', '')
            return key.replace('/languages/', '')[:10] if key else None

        if isinstance(data, list) and data:
            return TypeGuards.extract_language(data[0])

        return None


def extract_author_refs(data: Dict) -> List[str]:
    """
    Extract author IDs from works data using TypeGuards.
    Handles all known polymorphic author reference patterns.
    """
    authors = data.get('authors', [])
    if not authors:
        return []

    result = []
    for author in authors:
        if isinstance(author, dict):
            author_ref = author.get('author')
            if author_ref:
                key = TypeGuards.extract_key(author_ref)
            else:
                key = TypeGuards.extract_key(author)
            if key:
                result.append(key)
        elif isinstance(author, str):
            result.append(extract_id(author))

    return result


def find_dump_file(pattern: str, directory: Path = Path(".")) -> Optional[Path]:
    candidates = list(directory.glob(f"{pattern}*.txt.gz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def progress(count: int, interval: int = 100000) -> bool:
    return count % interval == 0


# =============================================================================
# STEP A: WIKIDATA ENRICHER (Build In-Memory Lookup)
# =============================================================================

def extract_wikidata_value(statement: Dict) -> Optional[str]:
    if not statement:
        return None
    value = statement.get('value', {})
    if not isinstance(value, dict):
        return str(value) if value else None

    value_type = value.get('type')
    content = value.get('content')

    if value_type == 'value':
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            if 'time' in content:
                time_str = content['time']
                match = re.search(r'(\d{4})', time_str)
                if match:
                    return match.group(1)
            elif 'id' in content:
                return content['id']
            elif 'text' in content:
                return content['text']
        return str(content) if content else None
    return None


def build_wikidata_lookup(input_path: Path, logger: ErrorLogger, limit: int = 0) -> Tuple[Dict[str, Dict], Stats]:
    """
    Build in-memory lookup dict from Wikidata dump.
    Returns: {author_id: {wikidata_id, image_url, gender, country_qid, birth_year, death_year}}
    """
    print("\n" + "=" * 70)
    print("STEP A: BUILDING WIKIDATA LOOKUP")
    print("=" * 70)
    print(f"Input: {input_path}")

    stats = Stats()
    lookup = {}

    with gzip.open(input_path, 'rt', encoding='utf-8', errors='replace') as infile:
        for line in infile:
            stats.processed += 1
            if limit and stats.processed > limit:
                break
            if progress(stats.processed):
                print(f"  Processed: {stats.processed:,} | Found: {len(lookup):,}")

            try:
                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                qid = parts[0]
                json_str = parts[1]
                if json_str.startswith('"') and json_str.endswith('"'):
                    json_str = json_str[1:-1].replace('""', '"')

                data = json.loads(json_str)
                statements = data.get('statements', {})

                ol_id_statements = statements.get('P648', [])
                if not ol_id_statements:
                    continue

                ol_id = extract_wikidata_value(ol_id_statements[0])
                if not ol_id:
                    continue

                if ol_id.startswith('/authors/'):
                    ol_id = ol_id.replace('/authors/', '')
                elif not (ol_id.startswith('OL') and ol_id.endswith('A')):
                    continue

                record = {'wikidata_id': qid}

                if 'P18' in statements and statements['P18']:
                    record['image_url'] = extract_wikidata_value(statements['P18'][0])

                if 'P21' in statements and statements['P21']:
                    gender_qid = extract_wikidata_value(statements['P21'][0])
                    record['gender'] = GENDER_MAP.get(gender_qid, gender_qid)

                if 'P27' in statements and statements['P27']:
                    record['country_qid'] = extract_wikidata_value(statements['P27'][0])

                if 'P569' in statements and statements['P569']:
                    record['birth_year_wd'] = extract_wikidata_value(statements['P569'][0])

                if 'P570' in statements and statements['P570']:
                    record['death_year_wd'] = extract_wikidata_value(statements['P570'][0])

                lookup[ol_id] = record
                stats.kept += 1

            except Exception as e:
                stats.errors += 1
                logger.log('wikidata', qid if 'qid' in dir() else 'unknown', str(e))

    print(f"\nWikidata Lookup Built: {len(lookup):,} authors indexed")
    return lookup, stats


# =============================================================================
# STEP B: AUTHORS PROCESSOR (With Wikidata Enrichment)
# =============================================================================

def process_authors(input_path: Path, output_path: Path, wikidata_lookup: Dict[str, Dict],
                    logger: ErrorLogger, limit: int = 0) -> Stats:
    print("\n" + "=" * 70)
    print("STEP B: AUTHORS PROCESSING (WITH ENRICHMENT)")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Wikidata Lookup: {len(wikidata_lookup):,} entries")

    stats = Stats()
    enriched_count = 0

    columns = [
        'author_id', 'name', 'bio', 'birth_date', 'death_date',
        'wikidata_id', 'gender', 'country_qid', 'image_url',
        'birth_year_wd', 'death_year_wd'
    ]

    with gzip.open(input_path, 'rt', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8-sig') as outfile:

        writer = csv.DictWriter(outfile, fieldnames=columns, extrasaction='ignore',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for line in infile:
            stats.processed += 1
            if limit and stats.processed > limit:
                break
            if progress(stats.processed):
                print(f"  Processed: {stats.processed:,} | Kept: {stats.kept:,} | Enriched: {enriched_count:,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                data = json.loads(parts[4])

                author_id = extract_id(key)
                name = TypeGuards.extract_text(data.get('name'), max_length=500)

                if not name:
                    stats.skipped += 1
                    logger.log('authors', key, 'no_name')
                    continue

                bio = TypeGuards.extract_text(data.get('bio'), max_length=1000)
                birth_date = TypeGuards.extract_first_string(data.get('birth_date'))
                death_date = TypeGuards.extract_first_string(data.get('death_date'))

                record = {
                    'author_id': author_id,
                    'name': name,
                    'bio': bio,
                    'birth_date': birth_date,
                    'death_date': death_date,
                }

                wd = wikidata_lookup.get(author_id)
                if wd:
                    record['wikidata_id'] = wd.get('wikidata_id')
                    record['gender'] = wd.get('gender')
                    record['country_qid'] = wd.get('country_qid')
                    record['image_url'] = wd.get('image_url')
                    record['birth_year_wd'] = wd.get('birth_year_wd')
                    record['death_year_wd'] = wd.get('death_year_wd')
                    enriched_count += 1

                writer.writerow(record)
                stats.kept += 1

            except Exception as e:
                stats.errors += 1
                logger.log('authors', key if 'key' in dir() else 'unknown', str(e))

    print(f"\nAuthors Complete: {stats.kept:,} authors | {enriched_count:,} enriched with Wikidata")
    return stats


# =============================================================================
# STEP C: WORKS PROCESSOR
# =============================================================================

def extract_series(data: Dict) -> Optional[str]:
    series_field = data.get('series')
    if series_field:
        if isinstance(series_field, list) and series_field:
            return clean_text(series_field[0], max_length=200)
        elif isinstance(series_field, str):
            return clean_text(series_field, max_length=200)

    title = data.get('title', '')
    for pattern in SERIES_PATTERNS:
        match = pattern.search(title)
        if match:
            series_name = match.group(1).strip()
            if len(series_name) >= 3 and len(series_name) <= 100:
                return clean_text(series_name, max_length=200)

    return None


def calculate_completeness(data: Dict) -> int:
    score = 0
    weights = {
        'title': 25,
        'authors': 25,
        'description': 20,
        'covers': 15,
        'first_publish_date': 10,
        'subjects': 5,
    }

    if TypeGuards.extract_text(data.get('title')):
        score += weights['title']

    author_refs = extract_author_refs(data)
    if author_refs:
        score += weights['authors']

    desc = TypeGuards.extract_text(data.get('description'))
    if desc and len(desc) > 20:
        score += weights['description']

    covers = data.get('covers', [])
    if covers and any(c and c > 0 for c in covers if isinstance(c, int)):
        score += weights['covers']

    if TypeGuards.extract_year(data.get('first_publish_date')):
        score += weights['first_publish_date']

    subjects = TypeGuards.extract_string_list(data.get('subjects'))
    if subjects:
        score += weights['subjects']

    return score


def process_works(input_path: Path, works_output: Path, authors_output: Path,
                  tags_output: Path, logger: ErrorLogger, limit: int = 0) -> Tuple[Stats, Set[str]]:
    print("\n" + "=" * 70)
    print("STEP C: WORKS PROCESSING")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {works_output}")
    print(f"        {authors_output}")
    print(f"        {tags_output}")

    stats = Stats()
    valid_work_ids = set()

    works_columns = ['work_id', 'title', 'subtitle', 'first_publish_year', 'series_name',
                     'cover_id', 'description', 'completeness_score']
    authors_columns = ['work_id', 'author_id', 'ordinal']
    tags_columns = ['work_id', 'tag']

    with gzip.open(input_path, 'rt', encoding='utf-8', errors='replace') as infile, \
         open(works_output, 'w', newline='', encoding='utf-8-sig') as works_file, \
         open(authors_output, 'w', newline='', encoding='utf-8-sig') as authors_file, \
         open(tags_output, 'w', newline='', encoding='utf-8-sig') as tags_file:

        works_writer = csv.DictWriter(works_file, fieldnames=works_columns,
                                      extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        authors_writer = csv.DictWriter(authors_file, fieldnames=authors_columns,
                                        extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        tags_writer = csv.DictWriter(tags_file, fieldnames=tags_columns,
                                     extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)

        works_writer.writeheader()
        authors_writer.writeheader()
        tags_writer.writeheader()

        for line in infile:
            stats.processed += 1
            if limit and stats.processed > limit:
                break
            if progress(stats.processed):
                print(f"  Processed: {stats.processed:,} | Kept: {stats.kept:,} | Skipped: {stats.skipped:,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                data = json.loads(parts[4])

                work_id = extract_id(key)
                title = TypeGuards.extract_text(data.get('title'), max_length=500)

                if not title:
                    stats.skipped += 1
                    logger.log('works', key, 'no_title')
                    continue

                completeness = calculate_completeness(data)
                if completeness < COMPLETENESS_THRESHOLD:
                    stats.skipped += 1
                    logger.log('works', key, f'low_completeness:{completeness}')
                    continue

                subtitle = TypeGuards.extract_text(data.get('subtitle'), max_length=300)
                series_name = extract_series(data)
                description = TypeGuards.extract_text(data.get('description'), max_length=2000)

                first_publish_year = TypeGuards.extract_year(data.get('first_publish_date'))

                covers = data.get('covers', [])
                cover_id = TypeGuards.extract_int(covers[0] if covers else None, min_val=1)

                works_writer.writerow({
                    'work_id': work_id,
                    'title': title,
                    'subtitle': subtitle,
                    'first_publish_year': first_publish_year,
                    'series_name': series_name,
                    'cover_id': cover_id,
                    'description': description,
                    'completeness_score': completeness,
                })

                valid_work_ids.add(work_id)
                stats.kept += 1

                author_refs = extract_author_refs(data)
                for ordinal, author_id in enumerate(author_refs, 1):
                    authors_writer.writerow({
                        'work_id': work_id,
                        'author_id': author_id,
                        'ordinal': ordinal,
                    })

                tags = set()
                for subject_field in ['subjects', 'subject_places', 'subject_people', 'subject_times']:
                    subject_list = TypeGuards.extract_string_list(data.get(subject_field), max_items=20)
                    for subj in subject_list:
                        if subj and len(subj) >= 2:
                            tags.add(subj.lower())

                for tag in list(tags)[:10]:
                    tags_writer.writerow({
                        'work_id': work_id,
                        'tag': tag,
                    })

            except Exception as e:
                stats.errors += 1
                logger.log('works', key if 'key' in dir() else 'unknown', str(e))

    print(f"\nWorks Complete: {stats.kept:,} works | {stats.skipped:,} filtered")
    return stats, valid_work_ids


# =============================================================================
# STEP D: EDITIONS PROCESSOR
# =============================================================================

def classify_format(format_str: Optional[str]) -> Optional[str]:
    if not format_str:
        return None
    format_lower = format_str.lower()
    for format_type, keywords in FORMAT_RULES.items():
        for keyword in keywords:
            if keyword in format_lower:
                return format_type
    return None


def process_editions(input_path: Path, editions_output: Path,
                     valid_work_ids: Set[str], logger: ErrorLogger,
                     limit: int = 0) -> Tuple[Stats, Dict[str, Dict]]:
    """
    Process editions and return page stats for works enrichment.
    Returns: (stats, {work_id: {median_pages, edition_count, formats}})
    """
    print("\n" + "=" * 70)
    print("STEP D: EDITIONS PROCESSING")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Valid Works Filter: {len(valid_work_ids):,} work IDs")
    print(f"Output: {editions_output}")

    stats = Stats()
    work_pages: Dict[str, List[int]] = defaultdict(list)
    work_formats: Dict[str, Counter] = defaultdict(Counter)
    work_edition_count: Dict[str, int] = defaultdict(int)

    editions_columns = ['edition_id', 'work_id', 'isbn13', 'isbn10', 'title',
                        'format', 'pages', 'publisher', 'publish_year', 'language']

    with gzip.open(input_path, 'rt', encoding='utf-8', errors='replace') as infile, \
         open(editions_output, 'w', newline='', encoding='utf-8-sig') as editions_file:

        writer = csv.DictWriter(editions_file, fieldnames=editions_columns,
                                extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for line in infile:
            stats.processed += 1
            if limit and stats.processed > limit:
                break
            if progress(stats.processed):
                print(f"  Processed: {stats.processed:,} | Kept: {stats.kept:,} | Skipped: {stats.skipped:,}")

            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                key = parts[1].strip()
                data = json.loads(parts[4])

                work_refs = TypeGuards.extract_keys(data.get('works'))
                work_id = work_refs[0] if work_refs else None

                if not work_id or work_id not in valid_work_ids:
                    stats.skipped += 1
                    continue

                isbn13 = TypeGuards.extract_first_string(data.get('isbn_13'))
                isbn10 = TypeGuards.extract_first_string(data.get('isbn_10'))

                if not isbn13 and not isbn10:
                    stats.skipped += 1
                    logger.log('editions', key, 'no_isbn')
                    continue

                edition_id = extract_id(key)
                title = TypeGuards.extract_text(data.get('title'), max_length=500)

                physical_format = TypeGuards.extract_first_string(data.get('physical_format'))
                format_type = classify_format(physical_format)

                pages = TypeGuards.extract_int(data.get('number_of_pages'), min_val=PAGE_MIN, max_val=PAGE_MAX)
                if pages:
                    work_pages[work_id].append(pages)

                work_edition_count[work_id] += 1
                if format_type:
                    work_formats[work_id][format_type] += 1

                publisher = TypeGuards.extract_first_string(data.get('publishers'))
                if publisher:
                    publisher = clean_text(publisher, max_length=200)

                publish_year = TypeGuards.extract_year(data.get('publish_date'))

                language = TypeGuards.extract_language(data.get('languages'))

                writer.writerow({
                    'edition_id': edition_id,
                    'work_id': work_id,
                    'isbn13': isbn13,
                    'isbn10': isbn10,
                    'title': title,
                    'format': format_type,
                    'pages': pages,
                    'publisher': publisher,
                    'publish_year': publish_year,
                    'language': language,
                })
                stats.kept += 1

            except Exception as e:
                stats.errors += 1
                logger.log('editions', key if 'key' in dir() else 'unknown', str(e))

    print(f"\nEditions Complete: {stats.kept:,} editions")
    print(f"Building page stats for {len(work_pages):,} works...")

    page_stats = {}
    for work_id in work_edition_count:
        pages_list = work_pages.get(work_id, [])
        formats = work_formats.get(work_id, Counter())
        primary_format = formats.most_common(1)[0][0] if formats else None

        page_stats[work_id] = {
            'median_pages': int(median(pages_list)) if pages_list else None,
            'edition_count': work_edition_count[work_id],
            'primary_format': primary_format,
        }

    print(f"Page Stats Built: {len(page_stats):,} works")
    return stats, page_stats


# =============================================================================
# STEP E: ENRICH WORKS WITH EDITION DATA
# =============================================================================

def enrich_works_file(works_path: Path, page_stats: Dict[str, Dict], output_path: Path):
    """
    Read works CSV and add edition enrichment columns.
    """
    print("\n" + "=" * 70)
    print("STEP E: ENRICHING WORKS WITH EDITION DATA")
    print("=" * 70)
    print(f"Input:  {works_path}")
    print(f"Stats:  {len(page_stats):,} works with edition data")
    print(f"Output: {output_path}")

    enriched_count = 0

    with open(works_path, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        original_columns = reader.fieldnames

        new_columns = list(original_columns) + ['median_pages', 'edition_count', 'primary_format']

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=new_columns, extrasaction='ignore',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for row in reader:
                work_id = row.get('work_id')
                stats = page_stats.get(work_id, {})

                row['median_pages'] = stats.get('median_pages')
                row['edition_count'] = stats.get('edition_count')
                row['primary_format'] = stats.get('primary_format')

                if stats:
                    enriched_count += 1

                writer.writerow(row)

    print(f"Works Enriched: {enriched_count:,} works with edition data")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Open Library Production ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python etl_pipeline.py --all                    Run complete pipeline
  python etl_pipeline.py --step wikidata          Run Wikidata enrichment only
  python etl_pipeline.py --step authors works     Run Authors and Works
  python etl_pipeline.py --all --limit 100000     Test run with 100k limit
        """
    )

    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--step', nargs='+', choices=['wikidata', 'authors', 'works', 'editions'],
                        help='Run specific step(s)')
    parser.add_argument('--limit', type=int, default=0, help='Limit records per file (0 = all)')
    parser.add_argument('--input-dir', type=Path, default=Path('.'), help='Input directory')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='Output directory')

    args = parser.parse_args()

    if not args.all and not args.step:
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(exist_ok=True)

    steps_to_run = set()
    if args.all:
        steps_to_run = {'wikidata', 'authors', 'works', 'editions'}
    else:
        steps_to_run = set(args.step)

    print("=" * 70)
    print("OPEN LIBRARY PRODUCTION ETL PIPELINE")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps:      {', '.join(sorted(steps_to_run))}")
    print(f"Input Dir:  {args.input_dir}")
    print(f"Output Dir: {args.output_dir}")
    if args.limit:
        print(f"Limit:      {args.limit:,} records per file")

    logger = ErrorLogger(args.output_dir / "etl_errors.log")
    all_stats = {}
    valid_work_ids = set()
    wikidata_lookup = {}
    page_stats = {}

    start_time = datetime.now()

    # =========================================================================
    # STEP A: Build Wikidata Lookup (in-memory for author enrichment)
    # =========================================================================
    if 'wikidata' in steps_to_run or 'authors' in steps_to_run:
        wikidata_path = find_dump_file("ol_dump_wikidata", args.input_dir)
        if wikidata_path:
            wikidata_lookup, stats = build_wikidata_lookup(
                wikidata_path,
                logger,
                args.limit
            )
            all_stats['wikidata'] = stats
        else:
            print("\nWARNING: Wikidata dump not found, authors will not be enriched")

    # =========================================================================
    # STEP B: Authors (with Wikidata enrichment merged in)
    # =========================================================================
    if 'authors' in steps_to_run:
        authors_path = find_dump_file("ol_dump_authors", args.input_dir)
        if authors_path:
            all_stats['authors'] = process_authors(
                authors_path,
                args.output_dir / "master_authors.csv",
                wikidata_lookup,
                logger,
                args.limit
            )
        else:
            print("\nERROR: Authors dump not found!")
            sys.exit(1)

    # =========================================================================
    # STEP C: Works (initial pass - will be enriched after editions)
    # =========================================================================
    if 'works' in steps_to_run:
        works_path = find_dump_file("ol_dump_works", args.input_dir)
        if works_path:
            works_temp_path = args.output_dir / "master_works_temp.csv"
            stats, valid_work_ids = process_works(
                works_path,
                works_temp_path,
                args.output_dir / "book_authors.csv",
                args.output_dir / "book_tags.csv",
                logger,
                args.limit
            )
            all_stats['works'] = stats

            valid_ids_path = args.output_dir / "valid_work_ids.txt"
            with open(valid_ids_path, 'w', encoding='utf-8') as f:
                for wid in valid_work_ids:
                    f.write(f"{wid}\n")
            print(f"Saved {len(valid_work_ids):,} valid work IDs to {valid_ids_path}")
        else:
            print("\nERROR: Works dump not found!")
            sys.exit(1)

    # =========================================================================
    # STEP D: Editions (collect page stats for works enrichment)
    # =========================================================================
    if 'editions' in steps_to_run:
        editions_path = find_dump_file("ol_dump_editions", args.input_dir)
        if editions_path:
            if not valid_work_ids:
                valid_ids_path = args.output_dir / "valid_work_ids.txt"
                if valid_ids_path.exists():
                    print(f"\nLoading valid work IDs from {valid_ids_path}...")
                    with open(valid_ids_path, 'r', encoding='utf-8') as f:
                        valid_work_ids = set(line.strip() for line in f if line.strip())
                    print(f"Loaded {len(valid_work_ids):,} valid work IDs")
                else:
                    print("\nERROR: No valid_work_ids available. Run 'works' step first.")
                    sys.exit(1)

            stats, page_stats = process_editions(
                editions_path,
                args.output_dir / "master_editions.csv",
                valid_work_ids,
                logger,
                args.limit
            )
            all_stats['editions'] = stats

    # =========================================================================
    # STEP E: Enrich Works with Edition data (page counts, formats)
    # =========================================================================
    if 'works' in steps_to_run and 'editions' in steps_to_run:
        works_temp_path = args.output_dir / "master_works_temp.csv"
        works_final_path = args.output_dir / "master_works.csv"

        if works_final_path.exists():
            works_final_path.unlink()

        enrich_works_file(works_temp_path, page_stats, works_final_path)

        works_temp_path.unlink()
        print(f"Removed temporary file: {works_temp_path}")
    elif 'works' in steps_to_run:
        works_temp_path = args.output_dir / "master_works_temp.csv"
        works_final_path = args.output_dir / "master_works.csv"
        if works_temp_path.exists():
            if works_final_path.exists():
                works_final_path.unlink()
            works_temp_path.rename(works_final_path)

    logger.close()
    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time: {elapsed}")
    print()

    print("STEP SUMMARY:")
    print("-" * 50)
    for step, stats in all_stats.items():
        print(f"  {step.upper():12} | Processed: {stats.processed:>12,} | Kept: {stats.kept:>12,} | Errors: {stats.errors:>6,}")

    print()
    print("ERROR SUMMARY:")
    print("-" * 50)
    for category, count in logger.summary().items():
        print(f"  {category}: {count:,}")

    print()
    print("OUTPUT FILES:")
    print("-" * 50)
    for csv_file in sorted(args.output_dir.glob("*.csv")):
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"  {csv_file.name}: {size_mb:.2f} MB")

    print()
    print("FINAL OUTPUT SCHEMA:")
    print("-" * 50)
    print("""
  master_authors.csv:
    author_id, name, bio, birth_date, death_date,
    wikidata_id, gender, country_qid, image_url, birth_year_wd, death_year_wd

  master_works.csv:
    work_id, title, subtitle, first_publish_year, series_name, cover_id,
    description, completeness_score, median_pages, edition_count, primary_format

  master_editions.csv:
    edition_id, work_id, isbn13, isbn10, title, format, pages,
    publisher, publish_year, language

  book_authors.csv:
    work_id, author_id, ordinal

  book_tags.csv:
    work_id, tag
""")


if __name__ == "__main__":
    main()
