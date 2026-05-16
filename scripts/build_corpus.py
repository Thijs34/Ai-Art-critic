"""Validate and merge museum ingestion files into data/art_sources.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ingest_common import DATA_DIR, append_jsonl, clean_text, looks_useful


DEFAULT_INPUTS = (DATA_DIR / "met_sources.jsonl", DATA_DIR / "cleveland_sources.jsonl")
DEFAULT_OUTPUT = DATA_DIR / "art_sources.jsonl"
OPEN_LICENSE_MARKERS = ("cc0", "public domain", "open access", "unrestricted")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def is_open_license(row: Dict[str, Any]) -> bool:
    license_text = clean_text(row.get("license")).lower()
    return any(marker in license_text for marker in OPEN_LICENSE_MARKERS)


def normalize_final(row: Dict[str, Any]) -> Dict[str, Any] | None:
    required = [
        "id",
        "source",
        "source_url",
        "license",
        "title",
        "artist",
        "date",
        "medium",
        "culture",
        "department",
        "description",
        "tags",
        "image_url",
        "rag_text",
    ]
    normalized = {key: row.get(key) for key in required}
    if not isinstance(normalized.get("tags"), list):
        normalized["tags"] = []
    if not looks_useful(normalized) or not is_open_license(normalized):
        return None
    return normalized


def build(inputs: List[Path], output: Path) -> int:
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for path in inputs:
        for row in read_jsonl(path):
            normalized = normalize_final(row)
            if not normalized:
                continue
            record_id = str(normalized["id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            rows.append(normalized)

    output.unlink(missing_ok=True)
    return append_jsonl(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final open-access museum RAG corpus.")
    parser.add_argument("--input", type=Path, action="append", default=None, help="Input JSONL file. May be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final output JSONL path.")
    args = parser.parse_args()

    inputs = args.input or list(DEFAULT_INPUTS)
    count = build(inputs, args.output)
    print(f"Wrote {count} validated records to {args.output}")


if __name__ == "__main__":
    main()
