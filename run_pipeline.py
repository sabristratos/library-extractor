"""
Open Library Data Pipeline Orchestrator
========================================
Runs all extraction and download scripts in the correct order.

Order:
    1. extract_top_books.py    - Extract top 100k books with metadata
    2. extract_authors.py      - Extract authors for top books
    3. enrich_author_images.py - Fetch Wikidata images for authors
    4. download_author_photos.py - Download author photos
    5. download_covers.py      - Download book covers (LAST - slowest)

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --skip-covers      # Skip cover downloads
    python run_pipeline.py --only-extract     # Only run extraction (no downloads)
    python run_pipeline.py --resume-from 3    # Resume from step 3

Author: Data Engineering Team
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# PIPELINE STEPS
# =============================================================================

STEPS = [
    {
        'num': 1,
        'name': 'Extract Top Books',
        'script': 'extract_top_books.py',
        'args': [],
        'category': 'extract',
        'description': 'Extracts top 100k books with ratings, metadata, authors, subjects',
    },
    {
        'num': 2,
        'name': 'Extract Authors',
        'script': 'extract_authors.py',
        'args': [],
        'category': 'extract',
        'description': 'Extracts author profiles and creates pivot tables',
    },
    {
        'num': 3,
        'name': 'Classify Books (SLM)',
        'script': 'classify_books.py',
        'args': [],
        'category': 'enrich',
        'description': 'Classifies books by mood, audience, themes using local SLM',
        'optional': True,
        'requires_model': True,
    },
    {
        'num': 4,
        'name': 'Enrich Author Images',
        'script': 'enrich_author_images.py',
        'args': [],
        'category': 'enrich',
        'description': 'Fetches Wikidata images for authors missing photos',
    },
    {
        'num': 5,
        'name': 'Download Author Photos',
        'script': 'download_author_photos.py',
        'args': [],
        'category': 'download',
        'description': 'Downloads author photos (OL + Wikipedia fallback)',
    },
    {
        'num': 6,
        'name': 'Download Book Covers',
        'script': 'download_covers.py',
        'args': [],
        'category': 'download',
        'description': 'Downloads book covers (OL + Google Books fallback)',
    },
]


# =============================================================================
# RUNNER
# =============================================================================

def run_step(step: dict, dry_run: bool = False) -> bool:
    """
    Run a single pipeline step.
    Returns True if successful, False otherwise.
    """
    script_path = Path(__file__).parent / step['script']

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + step['args']

    print(f"\n{'=' * 70}")
    print(f"STEP {step['num']}: {step['name'].upper()}")
    print(f"{'=' * 70}")
    print(f"Script:      {step['script']}")
    print(f"Description: {step['description']}")
    print(f"Command:     {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN - Skipping execution]")
        return True

    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 70)
        print(f"Completed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Step failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n  INTERRUPTED by user")
        return False


def run_pipeline(steps_to_run: list, dry_run: bool = False) -> int:
    """
    Run the pipeline with specified steps.
    Returns number of failed steps.
    """
    print("=" * 70)
    print("OPEN LIBRARY DATA PIPELINE")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps:      {len(steps_to_run)}")
    print(f"Dry Run:    {dry_run}")

    print("\nPipeline Overview:")
    for step in steps_to_run:
        print(f"  {step['num']}. {step['name']}")

    start_time = datetime.now()
    completed = 0
    failed = 0

    for step in steps_to_run:
        success = run_step(step, dry_run=dry_run)
        if success:
            completed += 1
        else:
            failed += 1
            print(f"\n  Pipeline stopped at step {step['num']}")
            break

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Time:  {elapsed}")
    print(f"Completed:   {completed}/{len(steps_to_run)} steps")
    if failed:
        print(f"Failed:      {failed} step(s)")
    else:
        print("Status:      SUCCESS")

    return failed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the Open Library data extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. extract_top_books.py      - Extract top 100k books
  2. extract_authors.py        - Extract author data
  3. classify_books.py         - [OPTIONAL] SLM classification (mood, audience, etc.)
  4. enrich_author_images.py   - Fetch Wikidata images
  5. download_author_photos.py - Download author photos
  6. download_covers.py        - Download book covers

Examples:
  python run_pipeline.py                    # Run full pipeline (skips step 3)
  python run_pipeline.py --with-classify    # Include SLM classification
  python run_pipeline.py --skip-covers      # Skip cover downloads (step 6)
  python run_pipeline.py --only-extract     # Only extraction (steps 1-2)
  python run_pipeline.py --resume-from 4    # Resume from step 4
  python run_pipeline.py --steps 1,2,3      # Run specific steps
  python run_pipeline.py --dry-run          # Show what would run
        """
    )

    parser.add_argument('--skip-covers', action='store_true',
                        help='Skip book cover downloads (step 6)')
    parser.add_argument('--skip-photos', action='store_true',
                        help='Skip author photo downloads (step 5)')
    parser.add_argument('--skip-classify', action='store_true',
                        help='Skip SLM book classification (step 3) - requires local model')
    parser.add_argument('--with-classify', action='store_true',
                        help='Include SLM book classification (step 3) - requires local model')
    parser.add_argument('--only-extract', action='store_true',
                        help='Only run extraction steps (1-2)')
    parser.add_argument('--only-download', action='store_true',
                        help='Only run download steps (4-5)')
    parser.add_argument('--resume-from', type=int, default=1,
                        help='Resume from step N (1-5)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated list of step numbers (e.g., "1,2,5")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')

    args = parser.parse_args()

    # Determine which steps to run
    steps_to_run = []

    if args.steps:
        # Run specific steps
        step_nums = [int(s.strip()) for s in args.steps.split(',')]
        steps_to_run = [s for s in STEPS if s['num'] in step_nums]
    elif args.only_extract:
        steps_to_run = [s for s in STEPS if s['category'] == 'extract']
    elif args.only_download:
        steps_to_run = [s for s in STEPS if s['category'] == 'download']
    else:
        # Full pipeline with optional skips
        # By default, skip optional steps (like classification) unless explicitly included
        steps_to_run = [s for s in STEPS if not s.get('optional', False)]

        # Include classification if requested
        if args.with_classify:
            classify_step = next((s for s in STEPS if s['script'] == 'classify_books.py'), None)
            if classify_step:
                # Insert at correct position
                steps_to_run = sorted(
                    steps_to_run + [classify_step],
                    key=lambda x: x['num']
                )

        if args.skip_covers:
            steps_to_run = [s for s in steps_to_run if s['script'] != 'download_covers.py']
        if args.skip_photos:
            steps_to_run = [s for s in steps_to_run if s['script'] != 'download_author_photos.py']
        if args.skip_classify:
            steps_to_run = [s for s in steps_to_run if s['script'] != 'classify_books.py']

    # Apply resume-from filter
    if args.resume_from > 1:
        steps_to_run = [s for s in steps_to_run if s['num'] >= args.resume_from]

    if not steps_to_run:
        print("ERROR: No steps to run. Check your arguments.")
        sys.exit(1)

    # Run the pipeline
    failed = run_pipeline(steps_to_run, dry_run=args.dry_run)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
