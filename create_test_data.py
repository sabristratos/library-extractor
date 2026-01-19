"""
Create sample GZIP test files mimicking Open Library dump format.
"""

import gzip
import json
from pathlib import Path

def create_test_files():
    Path(".").mkdir(exist_ok=True)

    authors_data = [
        {
            "key": "/authors/OL1A",
            "json": {
                "name": "George Orwell",
                "birth_date": "1903-06-25",
                "death_date": {"type": "/type/datetime", "value": "1950-01-21"},
                "bio": {"type": "/type/text", "value": "Eric Arthur Blair, known by his pen name George Orwell, was an English novelist and essayist.\nHe wrote many famous works."},
                "remote_ids": {"wikidata": "Q3335"}
            }
        },
        {
            "key": "/authors/OL2A",
            "json": {
                "name": "Jane Austen",
                "birth_date": "1775",
                "death_date": "1817",
                "bio": "English novelist known for her six major novels.",
                "remote_ids": {"wikidata": "Q36322"}
            }
        },
        {
            "key": "/authors/OL3A",
            "json": {
                "name": "Test Author No Bio",
                "birth_date": None
            }
        },
        {
            "key": "/authors/OL4A",
            "json": {}
        },
        {
            "key": "/authors/OL5A",
            "json": {
                "name": "   ",
            }
        },
    ]

    works_data = [
        {
            "key": "/works/OL1W",
            "json": {
                "title": "1984",
                "subtitle": "A Dystopian Novel",
                "description": {"type": "/type/text", "value": "A dystopian social science fiction novel and cautionary tale."},
                "authors": [
                    {"author": {"key": "/authors/OL1A"}},
                ],
                "covers": [12345, -1, 67890],
                "subjects": ["Dystopia", "Totalitarianism", "Political fiction"],
                "subject_places": ["London", "Oceania"],
                "first_publish_date": "1949",
                "series": "Classics"
            }
        },
        {
            "key": "/works/OL2W",
            "json": {
                "title": "Pride and Prejudice",
                "description": "A romantic novel of manners.",
                "authors": [
                    {"author": "/authors/OL2A"},
                ],
                "covers": [11111],
                "subjects": ["Romance", "Social class", "England", "19th century", "Marriage", "Family", "Sisters", "Courtship", "Manners", "Society", "Extra Tag 11", "Extra Tag 12"],
                "created": {"type": "/type/datetime", "value": "2008-04-01T03:28:50"}
            }
        },
        {
            "key": "/works/OL3W",
            "json": {
                "title": "Animal Farm",
                "authors": [
                    {"author": {"key": "/authors/OL1A"}},
                    {"author": {"key": "/authors/OL2A"}},
                ],
                "subjects": ["Allegory", "Satire"],
                "first_publish_date": "August 17, 1945",
                "series": ["Penguin Classics", "Modern Library"]
            }
        },
        {
            "key": "/works/OL4W",
            "json": {
                "title": "Untitled"
            }
        },
        {
            "key": "/works/OL5W",
            "json": {
                "description": "A book with no title"
            }
        },
        {
            "key": "/works/OL6W",
            "json": {
                "title": "Low Quality Book"
            }
        },
        {
            "key": "/works/OL7W",
            "json": {
                "title": "Book With\tTabs\nAnd\rNewlines",
                "description": "Description with\ttabs and\nnewlines\r\ntoo.",
                "authors": [{"author": {"key": "/authors/OL3A"}}],
                "covers": [99999]
            }
        },
    ]

    print("Creating ol_dump_authors.txt.gz...")
    with gzip.open("ol_dump_authors.txt.gz", 'wt', encoding='utf-8') as f:
        for author in authors_data:
            line = f"/type/author\t{author['key']}\t1\t2023-01-01\t{json.dumps(author['json'])}\n"
            f.write(line)

    print("Creating ol_dump_works.txt.gz...")
    with gzip.open("ol_dump_works.txt.gz", 'wt', encoding='utf-8') as f:
        for work in works_data:
            line = f"/type/work\t{work['key']}\t1\t2023-01-01\t{json.dumps(work['json'])}\n"
            f.write(line)
        f.write("malformed line with not enough columns\n")
        f.write(f"/type/work\t/works/OL99W\t1\t2023-01-01\t{{invalid json}}\n")

    print("Test files created successfully!")

if __name__ == "__main__":
    create_test_files()
