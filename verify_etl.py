"""
Verification script - processes first 100 records from each dump file.
Outputs results to verify_results.txt to avoid console encoding issues.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from etl_open_library import (
    TypeGuards, DateNormalizer, clean_text,
    AuthorProcessor, WorksProcessor, ErrorLogger
)

SAMPLE_SIZE = 1000

def main():
    authors_file = Path("ol_dump_authors_2025-12-31.txt.gz")
    works_file = Path("ol_dump_works_2025-12-31.txt.gz")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    results = []

    if not authors_file.exists():
        results.append(f"ERROR: {authors_file} not found")
        return
    if not works_file.exists():
        results.append(f"ERROR: {works_file} not found")
        return

    results.append("=" * 60)
    results.append("RAW DATA PEEK")
    results.append("=" * 60)

    with gzip.open(authors_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.split('\t')
            data = json.loads(parts[4]) if len(parts) >= 5 else {}
            results.append(f"Author {i+1}: key={parts[1]}, json_keys={list(data.keys())[:8]}")

    with gzip.open(works_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.split('\t')
            data = json.loads(parts[4]) if len(parts) >= 5 else {}
            results.append(f"Work {i+1}: key={parts[1]}, json_keys={list(data.keys())[:8]}")

    results.append("")
    results.append("=" * 60)
    results.append(f"PROCESSING AUTHORS (first {SAMPLE_SIZE})")
    results.append("=" * 60)

    error_logger = ErrorLogger(output_dir / "v5_errors.log")
    author_proc = AuthorProcessor(output_dir / "v5_authors.csv", error_logger)

    with gzip.open(authors_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                key = parts[1].strip()
                data = json.loads(parts[4])
                author_proc.process_record(key, data)
            except Exception as e:
                results.append(f"  Error on record {i}: {e}")

    author_proc.close()
    error_logger.close()

    results.append(f"  Processed: {author_proc.stats.processed}")
    results.append(f"  Kept: {author_proc.stats.kept}")
    results.append(f"  Discarded: {author_proc.stats.discarded}")

    results.append("")
    results.append("=" * 60)
    results.append(f"PROCESSING WORKS (first {SAMPLE_SIZE})")
    results.append("=" * 60)

    error_logger2 = ErrorLogger(output_dir / "v5_errors2.log")
    works_proc = WorksProcessor(
        output_dir / "v5_works.csv",
        output_dir / "v5_pivot.csv",
        output_dir / "v5_tags.csv",
        error_logger2
    )

    with gzip.open(works_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            try:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                key = parts[1].strip()
                data = json.loads(parts[4])
                works_proc.process_record(key, data)
            except Exception as e:
                results.append(f"  Error on record {i}: {e}")

    works_proc.close()
    error_logger2.close()

    results.append(f"  Processed: {works_proc.stats.processed}")
    results.append(f"  Kept: {works_proc.stats.kept}")
    results.append(f"  Discarded: {works_proc.stats.discarded}")
    results.append(f"  Discard rate: {100 - works_proc.stats.kept_percentage():.1f}%")

    results.append("")
    results.append("=" * 60)
    results.append("VERIFICATION COMPLETE")
    results.append("=" * 60)
    results.append("Output files created in ./output/test_*.csv")

    with open(output_dir / "verify_results.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print("Verification complete. Results in output/verify_results.txt")

if __name__ == "__main__":
    main()
