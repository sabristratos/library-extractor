# Open Library Data Extractor

A production-grade ETL pipeline for processing Open Library data dumps into normalized CSV files suitable for database ingestion.

## Overview

This project processes the Open Library bulk data dumps and transforms them into clean, normalized CSV files. It handles:

- **Authors**: Extracts author metadata with optional Wikidata enrichment (gender, nationality, images)
- **Works**: Processes book works with completeness scoring and quality filtering
- **Editions**: Extracts ISBN, format, publisher, and page count information
- **Tags/Subjects**: Normalizes subject data for search indexing

## Features

- Stream processing for memory-efficient handling of multi-GB dump files
- Polymorphic type handling for Open Library's loosely-typed JSON
- Mojibake detection and UTF-8 encoding fixes
- Completeness scoring to filter low-quality records
- Series detection from both explicit fields and title patterns
- Wikidata integration for author enrichment
- Parallel-friendly architecture

## Requirements

- Python 3.10+
- No external dependencies (uses only standard library)

## Installation

```bash
git clone https://github.com/sabristratos/library-extractor.git
cd library-extractor
```

## Data Sources

Download the Open Library bulk data dumps from:
https://openlibrary.org/developers/dumps

Required files:
- `ol_dump_authors_YYYY-MM-DD.txt.gz`
- `ol_dump_works_YYYY-MM-DD.txt.gz`
- `ol_dump_editions_YYYY-MM-DD.txt.gz`
- `ol_dump_wikidata_YYYY-MM-DD.txt.gz` (optional, for author enrichment)
- `ol_dump_ratings_YYYY-MM-DD.txt.gz` (optional)
- `ol_dump_reading-log_YYYY-MM-DD.txt.gz` (optional)

## Usage

### Full Pipeline

```bash
python etl_pipeline.py --all
```

### Individual Steps

```bash
python etl_pipeline.py --step wikidata          # Build Wikidata lookup
python etl_pipeline.py --step authors           # Process authors
python etl_pipeline.py --step works             # Process works
python etl_pipeline.py --step editions          # Process editions
python etl_pipeline.py --step authors works     # Multiple steps
```

### With Record Limit (for testing)

```bash
python etl_pipeline.py --all --limit 100000
```

### Legacy ETL (simpler, authors + works only)

```bash
python etl_open_library.py
```

## Output Files

All output is written to the `./output/` directory:

| File | Description |
|------|-------------|
| `master_authors.csv` | Author records with Wikidata enrichment |
| `master_works.csv` | Work records with completeness scores |
| `master_editions.csv` | Edition records with ISBN and format |
| `book_authors.csv` | Work-to-author relationship pivot table |
| `book_tags.csv` | Work-to-subject/tag relationship table |
| `etl_errors.log` | Processing errors for debugging |

### Schema

**master_authors.csv**
```
author_id, name, bio, birth_date, death_date,
wikidata_id, gender, country_qid, image_url, birth_year_wd, death_year_wd
```

**master_works.csv**
```
work_id, title, subtitle, first_publish_year, series_name, cover_id,
description, completeness_score, median_pages, edition_count, primary_format
```

**master_editions.csv**
```
edition_id, work_id, isbn13, isbn10, title, format, pages,
publisher, publish_year, language
```

## Analysis Tools

### Pattern Analyzer

Analyze dump files to understand data patterns and quality:

```bash
python analyze_library_patterns.py --limit 1000000
```

### Wikidata Enrichment (standalone)

```bash
python enrich_authors_wikidata.py --output ./output/author_enrichment.csv
```

## Configuration

Key thresholds are defined at the top of `etl_pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `COMPLETENESS_THRESHOLD` | 40 | Minimum score for works (0-100) |
| `PAGE_MIN` | 20 | Minimum valid page count |
| `PAGE_MAX` | 1500 | Maximum valid page count |

## Quality Filtering

Works are scored on metadata completeness:
- Title: 25 points
- Authors: 25 points
- Description: 20 points
- Cover image: 15 points
- Publish date: 10 points
- Subjects: 5 points

Works scoring below the threshold (default 40) are filtered out.

## License

MIT License
