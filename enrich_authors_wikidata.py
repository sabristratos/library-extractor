"""
Wikidata Author Enrichment Pipeline
====================================
Extracts author metadata from Wikidata dump and creates an enrichment CSV
that can be joined with master_authors.csv via author_id (Open Library ID).

Usage:
    python enrich_authors_wikidata.py --wikidata ol_dump_wikidata.txt.gz --output author_enrichment.csv
"""

import argparse
import gzip
import json
import csv
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
from collections import Counter


WIKIDATA_PROPERTIES = {
    'P648': 'open_library_id',
    'P569': 'birth_date',
    'P570': 'death_date',
    'P21': 'gender',
    'P27': 'country',
    'P106': 'occupation',
    'P18': 'image_file',
    'P19': 'birth_place',
    'P20': 'death_place',
    'P214': 'viaf_id',
    'P227': 'gnd_id',
    'P244': 'loc_id',
    'P213': 'isni',
    'P1412': 'language',
}

GENDER_MAP = {
    'Q6581097': 'male',
    'Q6581072': 'female',
    'Q1097630': 'intersex',
    'Q1052281': 'transgender_female',
    'Q2449503': 'transgender_male',
    'Q48270': 'non-binary',
}


def extract_value(statement: Dict) -> Optional[str]:
    """
    Extract the actual value from a Wikidata statement.
    Handles different value types (string, time, wikibase-item, etc.)
    """
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
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', time_str)
                if match:
                    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                match = re.search(r'(\d{4})', time_str)
                if match:
                    return match.group(1)
            elif 'id' in content:
                return content['id']
            elif 'text' in content:
                return content['text']
        return str(content) if content else None

    return None


def extract_label(data: Dict, lang: str = 'en') -> Optional[str]:
    """Extract label in preferred language, fallback to 'mul' or first available."""
    labels = data.get('labels', {})
    if lang in labels:
        return labels[lang]
    if 'mul' in labels:
        return labels['mul']
    if labels:
        return next(iter(labels.values()))
    return None


def extract_description(data: Dict, lang: str = 'en') -> Optional[str]:
    """Extract description in preferred language."""
    descriptions = data.get('descriptions', {})
    if lang in descriptions:
        return descriptions[lang]
    if descriptions:
        return next(iter(descriptions.values()))
    return None


def clean_text(text: Optional[str], max_length: int = None) -> Optional[str]:
    """Clean text for CSV output."""
    if not text:
        return None
    text = re.sub(r'[\n\r\t]+', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]
    return text or None


def parse_wikidata_record(line: str) -> Optional[Dict]:
    """
    Parse a single Wikidata dump line and extract enrichment fields.
    Returns dict with author_id (OL ID) as key field for joining.
    """
    try:
        parts = line.split('\t')
        if len(parts) < 2:
            return None

        qid = parts[0]
        json_str = parts[1]

        if json_str.startswith('"') and json_str.endswith('"'):
            json_str = json_str[1:-1].replace('""', '"')

        data = json.loads(json_str)
        statements = data.get('statements', {})

        ol_id_statements = statements.get('P648', [])
        if not ol_id_statements:
            return None

        ol_id = extract_value(ol_id_statements[0])
        if not ol_id:
            return None

        if ol_id.startswith('/authors/'):
            ol_id = ol_id.replace('/authors/', '')
        elif ol_id.startswith('OL') and ol_id.endswith('A'):
            pass
        else:
            return None

        result = {
            'author_id': ol_id,
            'wikidata_id': qid,
            'name_wikidata': clean_text(extract_label(data, 'en')),
            'description': clean_text(extract_description(data, 'en'), max_length=500),
        }

        if 'P569' in statements and statements['P569']:
            result['birth_date_wd'] = extract_value(statements['P569'][0])

        if 'P570' in statements and statements['P570']:
            result['death_date_wd'] = extract_value(statements['P570'][0])

        if 'P21' in statements and statements['P21']:
            gender_qid = extract_value(statements['P21'][0])
            result['gender'] = GENDER_MAP.get(gender_qid, gender_qid)

        if 'P27' in statements and statements['P27']:
            result['country_qid'] = extract_value(statements['P27'][0])

        if 'P106' in statements and statements['P106']:
            occupations = []
            for occ in statements['P106'][:3]:
                occ_qid = extract_value(occ)
                if occ_qid:
                    occupations.append(occ_qid)
            if occupations:
                result['occupation_qids'] = '|'.join(occupations)

        if 'P18' in statements and statements['P18']:
            result['image_file'] = extract_value(statements['P18'][0])

        if 'P19' in statements and statements['P19']:
            result['birth_place_qid'] = extract_value(statements['P19'][0])

        if 'P20' in statements and statements['P20']:
            result['death_place_qid'] = extract_value(statements['P20'][0])

        if 'P214' in statements and statements['P214']:
            result['viaf_id'] = extract_value(statements['P214'][0])

        if 'P227' in statements and statements['P227']:
            result['gnd_id'] = extract_value(statements['P227'][0])

        if 'P244' in statements and statements['P244']:
            result['loc_id'] = extract_value(statements['P244'][0])

        if 'P213' in statements and statements['P213']:
            result['isni'] = extract_value(statements['P213'][0])

        return result

    except Exception:
        return None


def find_dump_file(pattern: str, directory: Path = Path(".")) -> Optional[Path]:
    """Find dump file matching pattern."""
    candidates = list(directory.glob(f"{pattern}*.txt.gz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Extract author enrichment data from Wikidata dump."
    )
    parser.add_argument('--wikidata', type=Path, help='Path to Wikidata dump file')
    parser.add_argument('--output', type=Path, default=Path('./output/author_enrichment.csv'),
                        help='Output CSV path')
    parser.add_argument('--limit', type=int, default=0, help='Limit records (0 = all)')

    args = parser.parse_args()

    wikidata_path = args.wikidata
    if not wikidata_path:
        wikidata_path = find_dump_file("ol_dump_wikidata")

    if not wikidata_path or not wikidata_path.exists():
        print("ERROR: Wikidata dump file not found.")
        print("Use --wikidata to specify path, or ensure ol_dump_wikidata*.txt.gz is in current directory.")
        sys.exit(1)

    args.output.parent.mkdir(exist_ok=True)

    print("=" * 70)
    print("WIKIDATA AUTHOR ENRICHMENT PIPELINE")
    print("=" * 70)
    print(f"Input:  {wikidata_path}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit:  {args.limit:,} records")

    columns = [
        'author_id',
        'wikidata_id',
        'name_wikidata',
        'description',
        'birth_date_wd',
        'death_date_wd',
        'gender',
        'country_qid',
        'occupation_qids',
        'birth_place_qid',
        'death_place_qid',
        'image_file',
        'viaf_id',
        'gnd_id',
        'loc_id',
        'isni',
    ]

    start_time = datetime.now()
    processed = 0
    written = 0
    errors = 0

    stats = Counter()

    with gzip.open(wikidata_path, 'rt', encoding='utf-8', errors='replace') as infile, \
         open(args.output, 'w', newline='', encoding='utf-8-sig') as outfile:

        writer = csv.DictWriter(outfile, fieldnames=columns, extrasaction='ignore',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for line in infile:
            processed += 1

            if args.limit and processed > args.limit:
                break

            if processed % 100000 == 0:
                print(f"  Processed: {processed:,} | Written: {written:,}")

            try:
                record = parse_wikidata_record(line)
                if record:
                    writer.writerow(record)
                    written += 1

                    if record.get('birth_date_wd'):
                        stats['has_birth'] += 1
                    if record.get('death_date_wd'):
                        stats['has_death'] += 1
                    if record.get('gender'):
                        stats['has_gender'] += 1
                    if record.get('description'):
                        stats['has_description'] += 1
                    if record.get('image_file'):
                        stats['has_image'] += 1
                    if record.get('viaf_id'):
                        stats['has_viaf'] += 1

            except Exception:
                errors += 1

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print(f"Elapsed Time:     {elapsed}")
    print(f"Records Scanned:  {processed:,}")
    print(f"Records Written:  {written:,}")
    print(f"Errors:           {errors:,}")

    if written > 0:
        print(f"\nENRICHMENT COVERAGE:")
        print(f"  Birth Date:    {stats['has_birth']:>8,} ({stats['has_birth']/written*100:.1f}%)")
        print(f"  Death Date:    {stats['has_death']:>8,} ({stats['has_death']/written*100:.1f}%)")
        print(f"  Gender:        {stats['has_gender']:>8,} ({stats['has_gender']/written*100:.1f}%)")
        print(f"  Description:   {stats['has_description']:>8,} ({stats['has_description']/written*100:.1f}%)")
        print(f"  Image:         {stats['has_image']:>8,} ({stats['has_image']/written*100:.1f}%)")
        print(f"  VIAF ID:       {stats['has_viaf']:>8,} ({stats['has_viaf']/written*100:.1f}%)")

    print(f"\nOutput: {args.output}")
    if args.output.exists():
        size_mb = args.output.stat().st_size / (1024 * 1024)
        print(f"Size:   {size_mb:.2f} MB")

    print("\nTo join with master_authors.csv:")
    print("  SQL:  SELECT * FROM master_authors a LEFT JOIN author_enrichment e ON a.author_id = e.author_id")


if __name__ == "__main__":
    main()
