# AI Art Critic

Initial project scaffold for an AI-powered web application that will analyze artworks and provide critique.

## Project Structure

- `frontend/` — Vite + React app initialization
- `backend/` — Python API server scaffold (FastAPI planned)
- `models/` — AI/model development workspace
- `datasets/` — Data storage for training/evaluation
- `docs/` — Project documentation

## Open-World Artist Pipeline

The open-world artist pipeline adds CLIP embeddings, retrieval, and optional LLM refinement on top of your Test 13 model.

### Setup

1. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

2. (Optional) Set your OpenAI key if you want LLM refinement:

```bash
set OPENAI_API_KEY=your_key_here
```

Or put it in the repo root `.env` file (auto-loaded by the backend):

```text
OPENAI_API_KEY=your_key_here
```

### Usage (Python)

```python
from pathlib import Path
from backend.open_world_artist import OpenWorldArtistPipeline

pipeline = OpenWorldArtistPipeline(
	model_path=Path("models/wikiart_test13_style_artist_warmstart_best.pt"),
	index_dir=Path("retrieval_index"),
)

# Add a few reference images (incremental updates are supported)
pipeline.add_images(
	["datasets/Wikiart/Impressionism/claude-monet_123.jpg"],
	artist="Claude Monet",
	style="Impressionism",
)

# Predict with retrieval + optional LLM refinement
result = pipeline.predict(
	"path/to/query_image.jpg",
	top_k=5,
	retrieval_k=8,
	use_llm=False,
)

print(result["final_artist"])
print(result["candidates"])
```

Notes:
- Retrieval auto-initializes from a sampled WikiArt subset and can be extended via `add_images()`.
- FAISS is used when available; on Windows the pipeline falls back to a numpy cosine search.

### API Endpoint

If you run the backend server, you can call the open-world pipeline directly:

`POST /api/predict-artist-open-world` with multipart form field `image`.

The first request after server start will initialize a lightweight index from a sampled subset of WikiArt
(default 2000 images) and cache it under `retrieval_index/`.
