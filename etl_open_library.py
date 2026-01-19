"""
Open Library Data Dump ETL Pipeline
====================================
Production-grade stream processor for transforming Open Library GZIP dumps
into normalized CSV files suitable for SQL ingestion.

Output Files:
    - master_authors.csv
    - master_works.csv
    - works_authors_pivot.csv
    - works_search_tags.csv
    - etl_errors.log
"""

import gzip
import json
import csv
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Generator
from dataclasses import dataclass


@dataclass
class ETLStats:
    """Track processing statistics."""
    processed: int = 0
    kept: int = 0
    discarded: int = 0
    errors: int = 0

    def kept_percentage(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.kept / self.processed) * 100


def fix_mojibake(text: str) -> str:
    """
    Attempt to fix double-encoded UTF-8 (mojibake).
    Common pattern: UTF-8 bytes incorrectly decoded as Latin-1, then re-encoded.
    Example: "é" -> "Ã©" or combining chars like "Ì„" (U+0304)
    """
    if not text:
        return text
    try:
        fixed = text.encode('latin-1').decode('utf-8')
        if fixed != text and not any(ord(c) > 0x10FFFF for c in fixed):
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    try:
        fixed = text.encode('cp1252').decode('utf-8')
        if fixed != text and not any(ord(c) > 0x10FFFF for c in fixed):
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def clean_text(text: Optional[str], max_length: int = None) -> Optional[str]:
    """
    Remove tabs and newlines from strings for CSV safety.
    Attempts to fix mojibake (double-encoded UTF-8).
    Optionally truncate to max_length.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    text = fix_mojibake(text)
    cleaned = re.sub(r'[\n\r\t]+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return None
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


class TypeGuards:
    """
    Polymorphic type handlers for Open Library's loosely-typed JSON.
    Each method normalizes different data shapes into consistent output.
    """

    @staticmethod
    def extract_text(value: Any) -> Optional[str]:
        """
        Handle text fields that can be string OR {"type": "/type/text", "value": "..."}.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return str(value.get("value", "")).strip() or None
        return None

    @staticmethod
    def extract_datetime(value: Any) -> Optional[str]:
        """
        Handle dates that can be string OR {"type": "/type/datetime", "value": "..."}.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return str(value.get("value", "")).strip() or None
        return None

    @staticmethod
    def extract_author_key(entry: Any) -> Optional[str]:
        """
        Extract author key from nested structure: entry['author']['key'].
        Handles both {"author": {"key": "/authors/OL123A"}} and {"author": "/authors/OL123A"}.
        """
        if not isinstance(entry, dict):
            return None
        author = entry.get("author")
        if author is None:
            return None
        if isinstance(author, dict):
            key = author.get("key")
        elif isinstance(author, str):
            key = author
        else:
            return None
        if key and key.startswith("/authors/"):
            return key.replace("/authors/", "")
        return None

    SERIES_PATTERNS = [
        re.compile(r'\(([^)]+?),?\s*#\s*\d+\)', re.IGNORECASE),
        re.compile(r'\(([^)]+?)\s+Book\s+\d+\)', re.IGNORECASE),
        re.compile(r'\(([^)]+?)\s+Vol\.?\s*\d+\)', re.IGNORECASE),
        re.compile(r'\(([^)]+?)\s+Part\s+\d+\)', re.IGNORECASE),
        re.compile(r'\(([^)]+?)\s+Tome\s+\d+\)', re.IGNORECASE),
        re.compile(r'^(.+?),\s*(?:tome|vol\.?|volume|book|part|#|no\.?)\s*\d+', re.IGNORECASE),
        re.compile(r'^(.+?)\s+(?:tome|vol\.?|volume|book|part)\s+\d+$', re.IGNORECASE),
    ]

    @staticmethod
    def extract_series_from_field(value: Any) -> Optional[str]:
        """
        Handle series field that can be string OR list of strings. Return first value.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, list) and len(value) > 0:
            first = value[0]
            if isinstance(first, str):
                return first.strip() or None
        return None

    @classmethod
    def extract_series_hybrid(cls, data: dict) -> Optional[str]:
        """
        Hybrid series extraction:
        1. Try explicit 'series' field in JSON
        2. Fallback: regex patterns on title to detect series indicators
        """
        series_field = cls.extract_series_from_field(data.get("series"))
        if series_field:
            return series_field

        title = data.get("title", "")
        if not title or not isinstance(title, str):
            return None

        for pattern in cls.SERIES_PATTERNS:
            match = pattern.search(title)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def extract_first_cover(covers: Any) -> Optional[int]:
        """
        Extract first valid cover ID from covers array.
        """
        if not isinstance(covers, list) or len(covers) == 0:
            return None
        for cover in covers:
            if isinstance(cover, int) and cover > 0:
                return cover
        return None


class DateNormalizer:
    """
    Normalize various date formats to YYYY-MM-DD or YYYY.
    """

    DATE_PATTERNS = [
        (r'^(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        (r'^(\d{4})-(\d{2})$', lambda m: f"{m.group(1)}-{m.group(2)}-01"),
        (r'^(\d{4})$', lambda m: m.group(1)),
        (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        (r'(\d{4})', lambda m: m.group(1)),
    ]

    @classmethod
    def normalize(cls, date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        date_str = date_str.strip()
        for pattern, formatter in cls.DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                return formatter(match)
        return None


class ErrorLogger:
    """
    Log parsing errors to a separate file without crashing the pipeline.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.error_count = 0

    def log(self, record_key: str, error_type: str, details: str):
        self.error_count += 1
        timestamp = datetime.now().isoformat()
        self.log_file.write(f"{timestamp}\t{record_key}\t{error_type}\t{details}\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class StreamReader:
    """
    Stream processor for GZIP TSV files.
    Yields parsed records one at a time without loading full file into RAM.
    """

    def __init__(self, file_path: Path, error_logger: ErrorLogger):
        self.file_path = file_path
        self.error_logger = error_logger

    def records(self) -> Generator[tuple, None, None]:
        """
        Yield (type, key, json_data) tuples from the dump file.
        """
        with gzip.open(self.file_path, 'rt', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.split('\t')
                    if len(parts) < 5:
                        self.error_logger.log(
                            f"line:{line_num}",
                            "MALFORMED_TSV",
                            f"Expected 5 columns, got {len(parts)}"
                        )
                        continue

                    record_type = parts[0].strip()
                    record_key = parts[1].strip()
                    json_blob = parts[4].strip()

                    try:
                        data = json.loads(json_blob)
                    except json.JSONDecodeError as e:
                        self.error_logger.log(record_key, "JSON_PARSE_ERROR", str(e))
                        continue

                    yield record_type, record_key, data

                except Exception as e:
                    self.error_logger.log(f"line:{line_num}", "UNEXPECTED_ERROR", str(e))
                    continue


class AuthorProcessor:
    """
    Process authors dump into master_authors.csv.
    """

    COLUMNS = ['author_id', 'name', 'birth_date', 'death_date', 'wikidata_id', 'bio_snippet']

    def __init__(self, output_path: Path, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.stats = ETLStats()
        self.csv_file = open(output_path, 'w', newline='', encoding='utf-8-sig')
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.COLUMNS,
            extrasaction='ignore',
            quoting=csv.QUOTE_MINIMAL
        )
        self.writer.writeheader()

    def process_record(self, key: str, data: dict) -> bool:
        """
        Process a single author record. Returns True if kept, False if discarded.
        """
        self.stats.processed += 1

        author_id = key.replace("/authors/", "")
        name = data.get("name")

        if not name or not isinstance(name, str) or not name.strip():
            self.stats.discarded += 1
            return False

        birth_date = TypeGuards.extract_datetime(data.get("birth_date"))
        if birth_date:
            birth_date = fix_mojibake(birth_date.strip())

        death_date = TypeGuards.extract_datetime(data.get("death_date"))
        if death_date:
            death_date = fix_mojibake(death_date.strip())

        wikidata_id = None
        remote_ids = data.get("remote_ids")
        if isinstance(remote_ids, dict):
            wd = remote_ids.get("wikidata")
            if isinstance(wd, str) and wd.startswith("Q"):
                wikidata_id = wd

        bio_raw = TypeGuards.extract_text(data.get("bio"))
        bio_snippet = clean_text(bio_raw, max_length=1000)

        row = {
            'author_id': author_id,
            'name': clean_text(name.strip()),
            'birth_date': birth_date,
            'death_date': death_date,
            'wikidata_id': wikidata_id,
            'bio_snippet': bio_snippet
        }

        self.writer.writerow(row)
        self.stats.kept += 1
        return True

    def close(self):
        self.csv_file.close()


class WorksProcessor:
    """
    Process works dump into master_works.csv, works_authors_pivot.csv, and works_search_tags.csv.
    """

    WORKS_COLUMNS = ['work_id', 'title', 'subtitle', 'publish_date', 'series_name', 'cover_id', 'completeness_score']
    PIVOT_COLUMNS = ['work_id', 'author_id', 'ordinal']
    TAGS_COLUMNS = ['work_id', 'tag']

    SCORE_THRESHOLD = 30
    MAX_TAGS_PER_WORK = 10

    def __init__(self, works_path: Path, pivot_path: Path, tags_path: Path, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.stats = ETLStats()

        self.works_file = open(works_path, 'w', newline='', encoding='utf-8-sig')
        self.works_writer = csv.DictWriter(
            self.works_file,
            fieldnames=self.WORKS_COLUMNS,
            quoting=csv.QUOTE_MINIMAL
        )
        self.works_writer.writeheader()

        self.pivot_file = open(pivot_path, 'w', newline='', encoding='utf-8-sig')
        self.pivot_writer = csv.DictWriter(
            self.pivot_file,
            fieldnames=self.PIVOT_COLUMNS,
            quoting=csv.QUOTE_MINIMAL
        )
        self.pivot_writer.writeheader()

        self.tags_file = open(tags_path, 'w', newline='', encoding='utf-8-sig')
        self.tags_writer = csv.DictWriter(
            self.tags_file,
            fieldnames=self.TAGS_COLUMNS,
            quoting=csv.QUOTE_MINIMAL
        )
        self.tags_writer.writeheader()

    def calculate_completeness(self, data: dict, has_description: bool, has_cover: bool,
                                has_authors: bool, has_subjects: bool, has_publish_date: bool) -> int:
        """
        Calculate completeness score (0-100) based on metadata presence.
        """
        score = 0
        score += 10  # Has title (guaranteed at this point)
        if has_description:
            score += 20
        if has_cover:
            score += 20
        if has_authors:
            score += 20
        if has_subjects:
            score += 10
        if has_publish_date:
            score += 10
        return score

    def collect_unique_tags(self, data: dict) -> list:
        """
        Aggregate subjects, subject_places, subject_people, subject_times.
        Return unique tags, limited to top 10.
        """
        seen = set()
        unique_tags = []

        tag_fields = ['subjects', 'subject_places', 'subject_people', 'subject_times']

        for field in tag_fields:
            values = data.get(field, [])
            if not isinstance(values, list):
                continue
            for tag in values:
                if isinstance(tag, str):
                    tag_clean = clean_text(tag)
                    if tag_clean and tag_clean.lower() not in seen:
                        seen.add(tag_clean.lower())
                        unique_tags.append(tag_clean)
                        if len(unique_tags) >= self.MAX_TAGS_PER_WORK:
                            return unique_tags

        return unique_tags

    def process_record(self, key: str, data: dict) -> bool:
        """
        Process a single work record. Returns True if kept, False if discarded.
        """
        self.stats.processed += 1

        work_id = key.replace("/works/", "")

        title = data.get("title")
        if not title or not isinstance(title, str):
            self.stats.discarded += 1
            return False
        title = title.strip()
        if not title or title.lower() == "untitled":
            self.stats.discarded += 1
            return False

        description = TypeGuards.extract_text(data.get("description"))
        has_description = bool(description)

        cover_id = TypeGuards.extract_first_cover(data.get("covers"))
        has_cover = cover_id is not None

        authors_list = data.get("authors", [])
        has_authors = isinstance(authors_list, list) and len(authors_list) > 0

        subjects = data.get("subjects", [])
        has_subjects = isinstance(subjects, list) and len(subjects) > 0

        raw_date = data.get("first_publish_date") or TypeGuards.extract_datetime(data.get("created"))
        publish_date = DateNormalizer.normalize(raw_date)
        has_publish_date = bool(publish_date)

        score = self.calculate_completeness(
            data, has_description, has_cover, has_authors, has_subjects, has_publish_date
        )

        if score < self.SCORE_THRESHOLD:
            self.stats.discarded += 1
            return False

        subtitle = data.get("subtitle")
        if isinstance(subtitle, str):
            subtitle = clean_text(subtitle.strip())
        else:
            subtitle = None

        series_name = TypeGuards.extract_series_hybrid(data)

        works_row = {
            'work_id': work_id,
            'title': clean_text(title),
            'subtitle': subtitle,
            'publish_date': publish_date,
            'series_name': clean_text(series_name),
            'cover_id': cover_id,
            'completeness_score': score
        }
        self.works_writer.writerow(works_row)

        if has_authors:
            for ordinal, author_entry in enumerate(authors_list):
                author_id = TypeGuards.extract_author_key(author_entry)
                if author_id:
                    self.pivot_writer.writerow({
                        'work_id': work_id,
                        'author_id': author_id,
                        'ordinal': ordinal
                    })

        unique_tags = self.collect_unique_tags(data)
        for tag in unique_tags:
            self.tags_writer.writerow({
                'work_id': work_id,
                'tag': tag
            })

        self.stats.kept += 1
        return True

    def close(self):
        self.works_file.close()
        self.pivot_file.close()
        self.tags_file.close()


class ProgressReporter:
    """
    Print status updates at defined intervals.
    """

    def __init__(self, report_interval: int = 100_000):
        self.report_interval = report_interval
        self.last_report = 0

    def report(self, stats: ETLStats, prefix: str = ""):
        if stats.processed - self.last_report >= self.report_interval:
            count_fmt = self.format_count(stats.processed)
            kept_pct = stats.kept_percentage()
            print(f"{prefix}Processed: {count_fmt} | Kept: {kept_pct:.1f}%")
            self.last_report = stats.processed

    @staticmethod
    def format_count(n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)


def process_authors(input_path: Path, output_dir: Path, error_logger: ErrorLogger) -> ETLStats:
    """
    Process the authors dump file.
    """
    print("\n" + "="*60)
    print("PHASE 1: Processing Authors Dump")
    print("="*60)

    reader = StreamReader(input_path, error_logger)
    processor = AuthorProcessor(output_dir / "master_authors.csv", error_logger)
    progress = ProgressReporter()

    for record_type, record_key, data in reader.records():
        try:
            processor.process_record(record_key, data)
            progress.report(processor.stats, prefix="[Authors] ")
        except Exception as e:
            error_logger.log(record_key, "PROCESSING_ERROR", str(e))
            processor.stats.errors += 1

    processor.close()

    print(f"\n[Authors] COMPLETE:")
    print(f"  - Processed: {processor.stats.processed:,}")
    print(f"  - Kept: {processor.stats.kept:,} ({processor.stats.kept_percentage():.1f}%)")
    print(f"  - Discarded: {processor.stats.discarded:,}")
    print(f"  - Errors: {processor.stats.errors:,}")

    return processor.stats


def process_works(input_path: Path, output_dir: Path, error_logger: ErrorLogger) -> ETLStats:
    """
    Process the works dump file.
    """
    print("\n" + "="*60)
    print("PHASE 2: Processing Works Dump")
    print("="*60)

    reader = StreamReader(input_path, error_logger)
    processor = WorksProcessor(
        output_dir / "master_works.csv",
        output_dir / "works_authors_pivot.csv",
        output_dir / "works_search_tags.csv",
        error_logger
    )
    progress = ProgressReporter()

    for record_type, record_key, data in reader.records():
        try:
            processor.process_record(record_key, data)
            progress.report(processor.stats, prefix="[Works] ")
        except Exception as e:
            error_logger.log(record_key, "PROCESSING_ERROR", str(e))
            processor.stats.errors += 1

    processor.close()

    print(f"\n[Works] COMPLETE:")
    print(f"  - Processed: {processor.stats.processed:,}")
    print(f"  - Kept: {processor.stats.kept:,} ({processor.stats.kept_percentage():.1f}%)")
    print(f"  - Discarded: {processor.stats.discarded:,}")
    print(f"  - Errors: {processor.stats.errors:,}")

    return processor.stats


def find_dump_file(pattern: str, input_dir: Path) -> Optional[Path]:
    """
    Find dump file matching pattern. Supports both dated (ol_dump_authors_YYYY-MM-DD.txt.gz)
    and non-dated (ol_dump_authors.txt.gz) filenames.
    """
    import glob
    candidates = list(input_dir.glob(f"{pattern}*.txt.gz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    print("="*60)
    print("Open Library ETL Pipeline")
    print("="*60)

    input_dir = Path(".")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    authors_file = find_dump_file("ol_dump_authors", input_dir)
    works_file = find_dump_file("ol_dump_works", input_dir)

    missing_files = []
    if not authors_file:
        missing_files.append("ol_dump_authors*.txt.gz")
    if not works_file:
        missing_files.append("ol_dump_works*.txt.gz")

    if missing_files:
        print(f"\nERROR: Missing input files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure dump files are in the current directory.")
        print("Supported patterns: ol_dump_authors*.txt.gz, ol_dump_works*.txt.gz")
        sys.exit(1)

    print(f"\nUsing files:")
    print(f"  - Authors: {authors_file.name}")
    print(f"  - Works: {works_file.name}")

    error_logger = ErrorLogger(output_dir / "etl_errors.log")

    start_time = datetime.now()

    try:
        author_stats = process_authors(authors_file, output_dir, error_logger)
        works_stats = process_works(works_file, output_dir, error_logger)
    finally:
        error_logger.close()

    elapsed = datetime.now() - start_time

    print("\n" + "="*60)
    print("ETL PIPELINE COMPLETE")
    print("="*60)
    print(f"\nElapsed Time: {elapsed}")
    print(f"Total Errors Logged: {error_logger.error_count:,}")
    print(f"\nOutput Files:")
    for f in output_dir.glob("*.csv"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB")
    print(f"  - etl_errors.log")


if __name__ == "__main__":
    main()
