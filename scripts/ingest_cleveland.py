"""Fetch CC0 Open Access metadata from the Cleveland Museum of Art API."""

from __future__ import annotations

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


API_URL = "https://openaccess-api.clevelandart.org/api/artworks"
SOURCE = "Cleveland Museum of Art"
LICENSE = "CC0 / public domain / open access"
TYPE_QUERIES = ("Painting", "Drawing", "Print", "Sculpture", "Photograph")


def normalize(record: Dict[str, Any]) -> Dict[str, Any] | None:
    license_status = clean_text(record.get("share_license_status")).upper()
    if license_status != "CC0":
        return None
    if not matches_focus(record.get("type"), record.get("collection"), record.get("technique")):
        return None

    artwork_id = clean_text(record.get("id"), 80)
    images = record.get("images") if isinstance(record.get("images"), dict) else {}
    web_image = images.get("web") if isinstance(images.get("web"), dict) else {}
    creators = record.get("creators") if isinstance(record.get("creators"), list) else []
    artists = [clean_text(item.get("description") or item.get("name"), 160) for item in creators if isinstance(item, dict)]
    artist = "; ".join(item for item in artists if item)
    source_url = clean_text(record.get("url") or f"https://www.clevelandart.org/art/{artwork_id}", 600)

    description = clean_text(
        record.get("description")
        or record.get("wall_description")
        or record.get("tombstone")
        or record.get("fun_fact"),
        2400,
    )
    tags = clean_list(record.get("tags"))
    row = {
        "id": f"cleveland_{artwork_id}",
        "source": SOURCE,
        "source_url": source_url,
        "license": LICENSE if license_status == "CC0" else license_status,
        "title": clean_text(record.get("title"), 300),
        "artist": artist,
        "date": clean_text(record.get("creation_date"), 160),
        "medium": clean_text(record.get("technique"), 300),
        "culture": clean_text(record.get("culture"), 200),
        "department": clean_text(record.get("department") or record.get("collection"), 200),
        "description": description,
        "tags": tags,
        "image_url": clean_text(web_image.get("url"), 600),
    }
    row["rag_text"] = rag_text(row)
    return row if looks_useful(row) else None


def fetch_page(query: str, skip: int, limit: int) -> List[Dict[str, Any]]:
    data = get_json(
        API_URL,
        {
            "q": query,
            "cc0": "",
            "has_image": 1,
            "skip": skip,
            "limit": limit,
        },
    )
    return data.get("data") if isinstance(data.get("data"), list) else []


def ingest(limit: int, delay: float, output: Path, resume: bool) -> int:
    seen = load_seen_ids(output) if resume else set()
    written = 0

    def rows() -> Iterable[Dict[str, Any]]:
        nonlocal written
        page_size = 100
        for query in TYPE_QUERIES:
            skip = 0
            while written < limit:
                try:
                    records = fetch_page(query, skip, page_size)
                except Exception as exc:
                    print(f"[cleveland] skipped page {query}/{skip}: {exc}", file=sys.stderr)
                    break
                if not records:
                    break
                for record in records:
                    record_id = f"cleveland_{clean_text(record.get('id'), 80)}"
                    if record_id in seen:
                        continue
                    row = normalize(record)
                    if not row:
                        continue
                    written += 1
                    yield row
                    if written >= limit:
                        break
                skip += page_size
                time.sleep(delay)
            if written >= limit:
                break

    append_jsonl(output, rows())
    return written


def main() -> None:
    parser = default_parser("Ingest CC0 records from the Cleveland Museum of Art Open Access API.")
    args = parser.parse_args()
    output = args.output or DATA_DIR / "cleveland_sources.jsonl"
    count = ingest(args.limit, args.delay, output, args.resume)
    print(f"Wrote {count} Cleveland records to {output}")


if __name__ == "__main__":
    main()
