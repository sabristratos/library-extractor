"""
Book Classification with Local SLM
===================================
Uses a small language model (Qwen 2.5-7B, Mistral Nemo, etc.) to classify
books by mood, intensity, audience, and themes using constrained outputs.

Requires: llama-cpp-python with GPU support
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

Model download (example):
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf

Usage:
    python classify_books.py
    python classify_books.py --model ./models/qwen2.5-7b-instruct-q4_k_m.gguf
    python classify_books.py --limit 1000 --batch-size 50

Author: Data Engineering Team
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, get_args

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("ERROR: pydantic not installed. Run: pip install pydantic")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed.")
    print("Run: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    sys.exit(1)


# =============================================================================
# TAXONOMY - Strict Categories (The Model MUST pick from these)
# =============================================================================

# Mood/Tone of the book
MoodType = Literal[
    "Inspiring",      # Uplifting, motivational
    "Dark",           # Heavy, grim, serious
    "Romantic",       # Love-focused, passionate
    "Suspenseful",    # Thrilling, tension-filled
    "Humorous",       # Funny, comedic, lighthearted
    "Educational",    # Informative, instructional
    "Philosophical",  # Thought-provoking, reflective
    "Melancholic",    # Sad, bittersweet, emotional
    "Adventurous",    # Action-packed, exciting
    "Mysterious",     # Enigmatic, intriguing
    "Unknown"         # Classification failed or insufficient data
]

# Target audience
AudienceType = Literal[
    "Children",       # Ages 0-12
    "Young Adult",    # Ages 13-17
    "Adult",          # General adult audience
    "Mature",         # Adult with mature themes
    "Professional",   # Business/career focused
    "Academic",       # Scholarly/research
    "Unknown"         # Classification failed or insufficient data
]

# Pacing/Intensity
PaceType = Literal[
    "Slow",           # Contemplative, literary
    "Moderate",       # Balanced pacing
    "Fast",           # Quick, page-turner
    "Unknown"         # Classification failed or insufficient data
]


class BookClassification(BaseModel):
    """Structured classification output - model must conform to this schema."""
    mood: MoodType = Field(..., description="Primary emotional tone of the book")
    secondary_mood: Optional[MoodType] = Field(None, description="Secondary tone if applicable")
    intensity: int = Field(..., ge=1, le=10, description="Emotional intensity from 1 (light) to 10 (heavy)")
    audience: AudienceType = Field(..., description="Primary target audience")
    pace: PaceType = Field(..., description="Reading pace/pacing of the narrative")
    themes: List[str] = Field(..., min_length=1, max_length=5, description="Key themes (1-5)")
    content_warnings: List[str] = Field(default_factory=list, max_length=3, description="Content warnings if any")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path("./output/top_100k_books.csv")
DEFAULT_OUTPUT = Path("./output/top_100k_books_classified.csv")
DEFAULT_MODEL = Path("./models/qwen2.5-7b-instruct-q4_k_m.gguf")

SYSTEM_PROMPT = """You are a book classifier. Analyze the provided book title and description.
Classify the book according to the schema provided.

Guidelines:
- mood: The PRIMARY emotional tone (pick the most dominant)
- secondary_mood: Only if there's a clear secondary tone, otherwise null
- intensity: 1-3 (light/casual), 4-6 (moderate), 7-10 (heavy/intense)
- audience: Based on content complexity and themes
- pace: Based on description style and genre conventions
- themes: 1-5 specific themes (e.g., "redemption", "family", "survival")
- content_warnings: Only if clearly indicated (violence, abuse, etc.)

Handle English, Spanish, Portuguese, French, and German equally well.
Be objective and consistent. When uncertain, choose the most likely category."""


# =============================================================================
# CLASSIFIER
# =============================================================================

class BookClassifier:
    """
    Hardened SLM classifier with:
    - Explicit constraint injection into prompts
    - Robust JSON extraction (handles markdown blocks)
    - Retry logic with backoff
    - Pydantic validation
    - Increased context window
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 0.5

    def __init__(self, model_path: Path, n_gpu_layers: int = -1, n_ctx: int = 8192):
        print(f"Loading model: {model_path}")
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        )
        print("Model loaded successfully")

        self.taxonomy = {
            "moods": [m for m in get_args(MoodType) if m != "Unknown"],
            "audiences": [a for a in get_args(AudienceType) if a != "Unknown"],
            "paces": [p for p in get_args(PaceType) if p != "Unknown"],
        }

    def _clean_json_output(self, content: str) -> str:
        """Strip markdown code blocks if present."""
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return content.strip()

    def _normalize_value(self, value: str, allowed: list) -> str:
        """Fuzzy match value to allowed options (handles casing issues)."""
        if not value:
            return "Unknown"
        value_lower = value.lower().replace("_", " ").replace("-", " ")
        for option in allowed:
            if option.lower() == value_lower:
                return option
        for option in allowed:
            if option.lower().replace(" ", "") == value_lower.replace(" ", ""):
                return option
        return "Unknown"

    def _generate_with_retry(self, messages: list, schema: dict) -> dict:
        """Generate with retry logic."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.llm.create_chat_completion(
                    messages=messages,
                    response_format={
                        "type": "json_object",
                        "schema": schema,
                    },
                    temperature=0.1,
                    max_tokens=500
                )
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        raise last_error

    def classify(self, title: str, description: str, subjects: str = "") -> dict:
        """
        Classify a single book.
        Returns dict with classification or {"error": "..."} on failure.
        """
        if not description and not subjects:
            return {
                "mood": "Unknown",
                "secondary_mood": None,
                "intensity": None,
                "audience": "Unknown",
                "pace": "Unknown",
                "themes": [],
                "content_warnings": [],
                "classification_confidence": "none"
            }

        formatted_system_prompt = SYSTEM_PROMPT + f"""

ALLOWED VALUES (you MUST use these exact values):
- mood: {', '.join(self.taxonomy['moods'])}
- audience: {', '.join(self.taxonomy['audiences'])}
- pace: {', '.join(self.taxonomy['paces'])}

Respond ONLY with valid JSON matching the schema. No markdown, no explanation."""

        desc = description[:6000] if len(description) > 6000 else description

        context_parts = [f"Title: {title}"]
        if desc and desc.strip():
            context_parts.append(f"Description: {desc}")
        if subjects and subjects.strip():
            context_parts.append(f"Genres/Subjects: {subjects}")

        user_content = "\n".join(context_parts)

        try:
            output = self._generate_with_retry(
                messages=[
                    {"role": "system", "content": formatted_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                schema=BookClassification.model_json_schema()
            )

            raw_content = output['choices'][0]['message']['content']
            clean_content = self._clean_json_output(raw_content)
            result_data = json.loads(clean_content)

            try:
                validated = BookClassification(**result_data)
                result = validated.model_dump()
            except Exception:
                result = result_data
                result['mood'] = self._normalize_value(result.get('mood'), self.taxonomy['moods'])
                result['audience'] = self._normalize_value(result.get('audience'), self.taxonomy['audiences'])
                result['pace'] = self._normalize_value(result.get('pace'), self.taxonomy['paces'])
                if result.get('secondary_mood'):
                    result['secondary_mood'] = self._normalize_value(result['secondary_mood'], self.taxonomy['moods'])
                    if result['secondary_mood'] == "Unknown":
                        result['secondary_mood'] = None

            result['classification_confidence'] = "high" if description else "medium"
            return result

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}"}
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def get_existing_classifications(output_path: Path) -> set:
    """Get work_ids that have already been classified."""
    classified = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('mood'):  # Has classification
                    classified.add(row.get('work_id', ''))
    return classified


def classify_books(input_path: Path, output_path: Path, model_path: Path,
                   limit: int = 0, batch_size: int = 100, resume: bool = True):
    """
    Classify all books in the input CSV.
    """
    print("=" * 70)
    print("BOOK CLASSIFICATION WITH LOCAL SLM")
    print("=" * 70)
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Model:      {model_path}")
    if limit:
        print(f"Limit:      {limit:,} books")

    # Check model exists
    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        print("\nTo download Qwen 2.5-7B:")
        print("  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models")
        sys.exit(1)

    # Load existing classifications for resume
    already_classified = set()
    if resume and output_path.exists():
        already_classified = get_existing_classifications(output_path)
        print(f"Resuming:   {len(already_classified):,} already classified")

    # Read input
    print("\nReading input file...")
    books = []
    fieldnames = None
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            books.append(row)

    print(f"Total books: {len(books):,}")

    # Filter to unclassified
    if resume:
        to_classify = [b for b in books if b.get('work_id') not in already_classified]
    else:
        to_classify = books

    if limit:
        to_classify = to_classify[:limit]

    print(f"To classify: {len(to_classify):,}")

    if not to_classify:
        print("\nNo books to classify. Done!")
        return

    # Initialize classifier
    classifier = BookClassifier(model_path)

    # New columns to add
    new_columns = [
        'mood', 'secondary_mood', 'intensity', 'audience', 'pace',
        'themes', 'content_warnings', 'classification_confidence'
    ]

    # Ensure output columns include new fields
    output_fieldnames = list(fieldnames) + [c for c in new_columns if c not in fieldnames]

    # Process in batches
    start_time = datetime.now()
    processed = 0
    errors = 0
    results = []

    # If resuming, load existing results
    if resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

    print("\n" + "-" * 70)
    print("CLASSIFYING BOOKS")
    print("-" * 70)

    for i, book in enumerate(to_classify):
        work_id = book.get('work_id', '')
        title = book.get('title', 'Unknown')
        description = book.get('description', '')
        subjects = book.get('subjects', '')

        # Classify
        classification = classifier.classify(title, description, subjects)

        if 'error' in classification:
            errors += 1
            # Add empty classification columns
            for col in new_columns:
                book[col] = ''
            book['classification_confidence'] = 'error'
        else:
            # Merge classification into book row
            book['mood'] = classification.get('mood', '')
            book['secondary_mood'] = classification.get('secondary_mood', '') or ''
            book['intensity'] = classification.get('intensity', '')
            book['audience'] = classification.get('audience', '')
            book['pace'] = classification.get('pace', '')
            book['themes'] = '|'.join(classification.get('themes', []))
            book['content_warnings'] = '|'.join(classification.get('content_warnings', []))
            book['classification_confidence'] = classification.get('classification_confidence', '')

        results.append(book)
        processed += 1

        # Progress
        if processed % 10 == 0 or processed == len(to_classify):
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / max(elapsed, 1)
            eta = (len(to_classify) - processed) / max(rate, 0.01)
            print(f"  [{processed:>6}/{len(to_classify):,}] {title[:40]:<42} -> {book.get('mood', 'ERR'):<12} ({rate:.1f}/s, ETA: {eta/60:.0f}m)")

        # Save checkpoint every batch_size
        if processed % batch_size == 0:
            print(f"\n  Saving checkpoint ({processed:,} processed)...")
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=output_fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for row in results:
                    # Ensure all columns exist
                    for col in output_fieldnames:
                        if col not in row:
                            row[col] = ''
                    writer.writerow(row)
            print(f"  Checkpoint saved.\n")

    # Final save
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in results:
            for col in output_fieldnames:
                if col not in row:
                    row[col] = ''
            writer.writerow(row)

    elapsed = datetime.now() - start_time

    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"Elapsed:    {elapsed}")
    print(f"Processed:  {processed:,}")
    print(f"Errors:     {errors:,}")
    print(f"Rate:       {processed / max(elapsed.total_seconds(), 1):.1f} books/second")

    # Distribution stats
    mood_counts = {}
    audience_counts = {}
    for row in results:
        mood = row.get('mood', '')
        if mood:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        audience = row.get('audience', '')
        if audience:
            audience_counts[audience] = audience_counts.get(audience, 0) + 1

    if mood_counts:
        print("\nMood Distribution:")
        for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {mood:<15} {count:>6,} ({100*count/processed:.1f}%)")

    if audience_counts:
        print("\nAudience Distribution:")
        for audience, count in sorted(audience_counts.items(), key=lambda x: -x[1]):
            print(f"  {audience:<15} {count:>6,} ({100*count/processed:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify books using local SLM with constrained outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Taxonomy:
  Mood:     Inspiring, Dark, Romantic, Suspenseful, Humorous, Educational,
            Philosophical, Melancholic, Adventurous, Mysterious, Unknown
  Audience: Children, Young Adult, Adult, Mature, Professional, Academic, Unknown
  Pace:     Slow, Moderate, Fast, Unknown
  Intensity: 1 (light) to 10 (heavy)

Notes:
  - "Unknown" is returned when classification fails or data is insufficient
  - The classifier injects allowed values into prompts for better SLM compliance
  - Retry logic (3 attempts) handles transient failures

Examples:
  python classify_books.py                              # Classify all
  python classify_books.py --limit 100                  # Test with 100 books
  python classify_books.py --model ./models/mistral.gguf  # Use different model
  python classify_books.py --no-resume                  # Start fresh
        """
    )

    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Input CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output CSV (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                        help=f'GGUF model path (default: {DEFAULT_MODEL})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of books to classify (0=all)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Save checkpoint every N books (default: 100)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore existing classifications')
    parser.add_argument('--n-gpu-layers', type=int, default=-1,
                        help='GPU layers (-1=all, 0=CPU only)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run extract_top_books.py first.")
        sys.exit(1)

    classify_books(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        limit=args.limit,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
