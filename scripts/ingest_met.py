"""Fetch public-domain Open Access metadata from The Met Collection API."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_common import (  # noqa: E402
    DATA_DIR,
    append_jsonl,
    clean_list,
    clean_text,
    default_parser,
    get_json,
    load_seen_ids,
    looks_useful,
    matches_focus,
    rag_text,
)


API_ROOT = "https://collectionapi.metmuseum.org/public/collection/v1"
SOURCE = "Metropolitan Museum of Art"
LICENSE = "CC0 / public domain / open access"
SEARCH_TERMS = ("painting", "drawing", "print", "sculpture", "photograph")


def fetch_ids(per_term: int) -> List[int]:
    ids: List[int] = []
    seen: set[int] = set()
    for term in SEARCH_TERMS:
        data = get_json(
            f"{API_ROOT}/search",
            {
                "q": term,
                "hasImages": "true",
                "isPublicDomain": "true",
            },
        )
        object_ids = data.get("objectIDs") if isinstance(data.get("objectIDs"), list) else []
        random.Random(term).shuffle(object_ids)
        for object_id in object_ids[:per_term]:
            if isinstance(object_id, int) and object_id not in seen:
                seen.add(object_id)
                ids.append(object_id)
    return ids


def normalize(record: Dict[str, Any]) -> Dict[str, Any] | None:
    if record.get("isPublicDomain") is not True:
        return None
    if not matches_focus(record.get("objectName"), record.get("classification"), record.get("medium")):
        return None

    object_id = record.get("objectID")
    source_url = clean_text(record.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{object_id}")
    tags = clean_list(record.get("tags"))
    description_parts = [
        record.get("artistDisplayBio"),
        record.get("period"),
        record.get("creditLine"),
        record.get("classification"),
    ]
    row = {
        "id": f"met_{object_id}",
        "source": SOURCE,
        "source_url": source_url,
        "license": LICENSE,
        "title": clean_text(record.get("title"), 300),
        "artist": clean_text(record.get("artistDisplayName") or record.get("artistAlphaSort"), 300),
        "date": clean_text(record.get("objectDate"), 160),
        "medium": clean_text(record.get("medium"), 300),
        "culture": clean_text(record.get("culture") or record.get("country"), 200),
        "department": clean_text(record.get("department"), 200),
        "description": clean_text(" | ".join(part for part in description_parts if part), 1400),
        "tags": tags,
        "image_url": clean_text(record.get("primaryImageSmall") or record.get("primaryImage"), 600),
    }
    row["rag_text"] = rag_text(row)
    return row if looks_useful(row) else None


def ingest(limit: int, delay: float, output: Path, resume: bool) -> int:
    seen = load_seen_ids(output) if resume else set()
    per_term = max(200, (limit * 3) // len(SEARCH_TERMS))
    object_ids = fetch_ids(per_term)
    written = 0

    def rows() -> Iterable[Dict[str, Any]]:
        nonlocal written
        for object_id in object_ids:
            record_id = f"met_{object_id}"
            if record_id in seen:
                continue
            try:
                record = get_json(f"{API_ROOT}/objects/{object_id}")
                row = normalize(record)
            except Exception as exc:
                print(f"[met] skipped {object_id}: {exc}", file=sys.stderr)
                row = None
            time.sleep(delay)
            if not row:
                continue
            written += 1
            yield row
            if written >= limit:
                break

    append_jsonl(output, rows())
    return written


def main() -> None:
    parser = default_parser("Ingest public-domain records from The Met Collection API.")
    args = parser.parse_args()
    output = args.output or DATA_DIR / "met_sources.jsonl"
    count = ingest(args.limit, args.delay, output, args.resume)
    print(f"Wrote {count} Met records to {output}")


if __name__ == "__main__":
    main()
