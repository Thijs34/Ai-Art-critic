# Open-Access Museum Ingestion

These scripts build the local RAG corpus from public museum APIs instead of generated text.

## Sources

- The Metropolitan Museum of Art Collection API: https://metmuseum.github.io/
- The Met Open Access policy: https://www.metmuseum.org/about-the-met/policies-and-documents/open-access
- Cleveland Museum of Art Open Access API: https://openaccess-api.clevelandart.org/

The scripts only keep records that are explicitly marked open:

- Met records must have `isPublicDomain == true`.
- Cleveland records must have `share_license_status == "CC0"`.

No image files are downloaded. The corpus stores metadata and public/open image URLs only.

## Run

From the repo root:

```bash
python scripts/ingest_met.py --limit 750 --resume
python scripts/ingest_cleveland.py --limit 750 --resume
python scripts/build_corpus.py
```

The final file used by chat retrieval is:

```text
data/art_sources.jsonl
```

Per-source intermediate files are:

```text
data/met_sources.jsonl
data/cleveland_sources.jsonl
```

## Useful Options

```bash
python scripts/ingest_met.py --limit 250 --delay 0.2 --resume
python scripts/ingest_cleveland.py --limit 250 --delay 0.2 --resume
python scripts/build_corpus.py --input data/met_sources.jsonl --input data/cleveland_sources.jsonl
```

`--resume` skips IDs already present in the output file. `build_corpus.py` deduplicates and validates final records.

## JSONL Shape

Each final record has:

```json
{
  "id": "met_436535",
  "source": "Metropolitan Museum of Art",
  "source_url": "https://www.metmuseum.org/art/collection/search/436535",
  "license": "CC0 / public domain / open access",
  "title": "...",
  "artist": "...",
  "date": "...",
  "medium": "...",
  "culture": "...",
  "department": "...",
  "description": "...",
  "tags": ["..."],
  "image_url": "https://...",
  "rag_text": "Title: ...\nArtist: ...\nDate: ..."
}
```

Keep this corpus metadata-focused. Do not add scraped text or image downloads unless the source explicitly permits that use.
