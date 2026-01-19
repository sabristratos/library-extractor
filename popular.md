Act as a Principal Data Engineer. I need a specialized Python script (`extract_top_books.py`) to generate a "Best 100,000 Books" dataset for my recommendation engine.

### 1. Objective
Create a CSV (`top_100k_books.csv`) containing the most popular/highest-rated books from the Open Library dataset, enriched with the necessary metadata for a frontend (Title, Cover ISBN, Ratings).

### 2. Input Data Sources
- `ol_dump_ratings.txt` (Format: `WorkID \t EditionID \t Rating \t Date`)
- `ol_dump_reading_log.txt` (Format: `WorkID \t EditionID \t Status \t Date`)
- `ol_dump_works.txt.gz` (Source for Titles)
- `ol_dump_editions.txt.gz` (Source for ISBNs)

### 3. Processing Logic (The Pipeline)

**Step A: Aggregation (The "Popularity" Engine)**
- **Inputs:** Ratings Dump + Reading Log Dump.
- **Logic:**
  1. Stream both files. Key off `WorkID`.
  2. Count `interactions` (Read + Want to Read + Rated).
  3. Sum `rating_score` and count `rating_votes`.
- **Selection:**
  1. Calculate `average_rating`.
  2. Calculate `popularity_score` = `interactions + (rating_votes * 2)`.
  3. Select the **Top 100,000** Work IDs based on `popularity_score`.
  4. Store these IDs in a `Set` for O(1) lookups in Step B/C.

**Step B: Metadata Enrichment (Titles)**
- **Input:** `ol_dump_works.txt.gz`.
- **Logic:** Stream the file. If `WorkID` is in the Top 100k Set, capture the `title`.

**Step C: ISBN Resolution (For Covers)**
- **Input:** `ol_dump_editions.txt.gz`.
- **Logic:**
  1. Stream the file. If `work_key` matches the Top 100k Set:
  2. Capture the **First Valid ISBN-13** found.
  3. *Optimization:* If an edition has a `cover_id`, prefer that ISBN.
  4. Stop searching for a Work once a high-quality ISBN is found (First match wins strategy is fine for speed, but preferring covers is better).

### 4. Output Schema
**File:** `top_100k_books.csv`
**Columns:**
1. `rank` (1-100,000)
2. `work_id` (e.g., OL123W)
3. `title`
4. `isbn` (The specific ISBN to query the Cover API)
5. `rating_avg` (e.g., 4.5)
6. `rating_count` (e.g., 1050)
7. `interaction_count` (Total popularity signal)

### 5. Technical Constraints
- **Performance:** Use `gzip` and `csv` modules. Do not use Pandas (memory constraints).
- **Format Handling:** Handle the TSV structure of the ratings dump robustly (check for optional columns).
- **Progress:** Print progress every 1,000,000 lines processed.

Generate the complete script.