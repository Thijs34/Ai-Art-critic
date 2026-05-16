"""Shared helpers for open-access museum ingestion scripts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
USER_AGENT = "AiArtCriticOpenAccessIngest/1.0 (metadata-only; contact: local)"
ALLOWED_TYPES = ("painting", "drawing", "print", "sculpture", "photograph", "photography")


def default_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=750, help="Maximum normalized records to write.")
    parser.add_argument("--delay", type=float, default=0.12, help="Delay between API calls in seconds.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--resume", action="store_true", help="Skip records already present in the output file.")
    return parser


def get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30, retries: int = 3) -> Dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt >= retries - 1:
                raise
            time.sleep(1.0 + attempt)
    return {}


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_seen_ids(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(row.get("id") or "")
            if record_id:
                seen.add(record_id)
    return seen


def clean_text(value: Any, max_length: int = 3000) -> str:
    text = " ".join(str(value or "").split())
    return text[:max_length]


def clean_list(values: Any, max_items: int = 12) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("term") or value.get("name") or value.get("title")
        text = clean_text(value, 80)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def looks_useful(row: Dict[str, Any]) -> bool:
    if not row.get("id") or not row.get("source") or not row.get("source_url"):
        return False
    if not row.get("title") or str(row.get("title")).strip().lower() in {"untitled", "unknown"}:
        return False
    if not row.get("license"):
        return False
    rag_text = clean_text(row.get("rag_text"))
    return len(rag_text) >= 80


def matches_focus(*values: Any) -> bool:
    haystack = " ".join(clean_text(value).lower() for value in values)
    return any(term in haystack for term in ALLOWED_TYPES)


def rag_text(fields: Dict[str, Any]) -> str:
    lines = []
    for label, key in (
        ("Title", "title"),
        ("Artist", "artist"),
        ("Date", "date"),
        ("Medium", "medium"),
        ("Culture", "culture"),
        ("Department", "department"),
        ("Description", "description"),
        ("Tags", "tags"),
        ("Source", "source_url"),
        ("License", "license"),
    ):
        value = fields.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item)
        value = clean_text(value, 1600)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)
