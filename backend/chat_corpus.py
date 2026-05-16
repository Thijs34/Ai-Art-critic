"""Museum-source corpus loader for chat retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).parent.parent
ART_SOURCES_PATH = ROOT / "data" / "art_sources.jsonl"


def load_chat_corpus(path: Path = ART_SOURCES_PATH) -> List[Dict[str, str]]:
    """Load normalized open-access museum records from JSONL."""
    if not path.exists():
        return []

    docs: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc = _row_to_context(row)
                if doc:
                    docs.append(doc)
    except (OSError, json.JSONDecodeError):
        return []
    return docs


def _row_to_context(row: Dict[str, Any]) -> Dict[str, str] | None:
    title = str(row.get("title") or "").strip()
    rag_text = str(row.get("rag_text") or "").strip()
    if not title or not rag_text:
        return None

    artist = str(row.get("artist") or "").strip()
    date = str(row.get("date") or "").strip()
    medium = str(row.get("medium") or "").strip()
    culture = str(row.get("culture") or "").strip()
    department = str(row.get("department") or "").strip()
    license_text = str(row.get("license") or "").strip()
    source_url = str(row.get("source_url") or "").strip()
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    tag_text = ", ".join(str(tag) for tag in tags[:8])

    metadata = " ".join(
        part for part in [artist, date, medium, culture, department, tag_text, license_text] if part
    )
    enriched_text = f"{metadata}\n{rag_text}" if metadata else rag_text

    return {
        "title": title,
        "text": enriched_text,
        "source": str(row.get("source") or "museum-open-access"),
        "kind": "museum-record",
        "grounding": f"open-access museum metadata; source: {source_url}" if source_url else "open-access museum metadata",
    }
