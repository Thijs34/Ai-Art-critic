"""Lumora API serving the Test 13 multitask style+artist model."""

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except Exception:
    load_dotenv = None

from open_world_artist import OpenWorldArtistPipeline
from conversation import ArtConversationService

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_FILENAME = "wikiart_test13_style_artist_warmstart_best.pt"
MODEL_PATH = ROOT / "models" / MODEL_FILENAME


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Open-world pipeline (retrieval + LLM) ─────────────────────────────────────
OPEN_WORLD_INDEX_MAX = int(os.environ.get("OPEN_WORLD_INDEX_MAX", 2000))
OPEN_WORLD_INDEX_SEED = int(os.environ.get("OPEN_WORLD_INDEX_SEED", 42))
OPEN_WORLD_INDEX_BATCH = int(os.environ.get("OPEN_WORLD_INDEX_BATCH", 32))

open_world_pipeline = OpenWorldArtistPipeline(
    model_path=MODEL_PATH,
    index_dir=ROOT / "retrieval_index",
    auto_init_index=True,
    max_index_images=OPEN_WORLD_INDEX_MAX,
    index_seed=OPEN_WORLD_INDEX_SEED,
    index_batch_size=OPEN_WORLD_INDEX_BATCH,
)
conversation_service = ArtConversationService()


class ArtworkAnalysisStore:
    """In-memory owner for artwork analysis used by chat grounding."""

    def __init__(self, ttl_seconds: int = 60 * 60 * 4, max_items: int = 120):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: Dict[str, Dict[str, Any]] = {}

    def add(self, analysis: Dict[str, Any]) -> str:
        self._prune()
        artwork_id = str(uuid.uuid4())
        self._items[artwork_id] = {
            "analysis": analysis,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._trim()
        return artwork_id

    def get(self, artwork_id: Optional[str]) -> Optional[Dict[str, Any]]:
        self._prune()
        if not artwork_id:
            return None
        item = self._items.get(artwork_id)
        if not item:
            return None
        item["updated_at"] = time.time()
        return item["analysis"]

    def _prune(self) -> None:
        now = time.time()
        expired = [
            artwork_id for artwork_id, item in self._items.items()
            if now - item.get("updated_at", 0) > self.ttl_seconds
        ]
        for artwork_id in expired:
            self._items.pop(artwork_id, None)

    def _trim(self) -> None:
        if len(self._items) <= self.max_items:
            return
        ranked = sorted(self._items.items(), key=lambda item: item[1].get("updated_at", 0), reverse=True)
        self._items = dict(ranked[: self.max_items])


artwork_store = ArtworkAnalysisStore()


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_FILENAME,
        "tasks": ["style", "artist", "open-world-artist", "art-conversation"],
    })


@app.post("/api/predict-artist-open-world")
def predict_artist_open_world():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the file under the key 'image'."}), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Could not open image: {exc}"}), 400

    llm_with_image = os.environ.get("OPEN_WORLD_LLM_WITH_IMAGE", "true").lower() in {"1", "true", "yes"}

    result = open_world_pipeline.predict(
        image=image,
        top_k=5,
        retrieval_k=8,
        use_llm=True,
        llm_with_image=llm_with_image,
        confidence_threshold=float(os.environ.get("OPEN_WORLD_CONF_THRESHOLD", 0.4)),
        margin_threshold=float(os.environ.get("OPEN_WORLD_MARGIN_THRESHOLD", 0.15)),
        style_confidence_threshold=float(os.environ.get("OPEN_WORLD_STYLE_CONF_THRESHOLD", 0.4)),
        enrich_analysis=os.environ.get("OPEN_WORLD_LLM_ENRICH_ANALYSIS", "true").lower() in {"1", "true", "yes"},
    )

    print(
        "[open-world]",
        "used_llm=", result.get("used_open_world_llm"),
        "confidence=", result.get("confidence"),
        "llm_error=", result.get("llm_error"),
        "final_artist=", result.get("final_artist"),
        "final_style=", result.get("final_style"),
        "used_openai_style=", result.get("used_openai_style"),
        "llm_artist=", (result.get("llm") or {}).get("artist"),
        "llm_style=", (result.get("llm") or {}).get("style"),
        "llm_unknown=", (result.get("llm") or {}).get("is_unknown"),
        "llm_time=", (result.get("llm") or {}).get("time_period"),
    )

    style_topk_raw = result.get("style_topk", [])
    style_topk = [
        {**item, "confidence": round(float(item.get("confidence", 0.0)) * 100, 1)}
        for item in style_topk_raw
    ]
    style = style_topk[0] if style_topk else {"label": "Unknown", "confidence": 0}
    if result.get("used_openai_style"):
        llm_style = (result.get("llm") or {}).get("style")
        llm_style_conf = (result.get("llm") or {}).get("style_confidence")
        if llm_style:
            style = {
                "label": llm_style,
                "confidence": round(float(llm_style_conf if isinstance(llm_style_conf, (int, float)) else 0.5) * 100, 1),
                "source": "openai",
            }
    else:
        style = {**style, "source": "local"}

    artist_conf = result.get("confidence", {}).get("top1", 0.0)
    if result.get("used_open_world_llm") and not result.get("confidence", {}).get("high_confidence", False):
        llm_conf = (result.get("llm") or {}).get("confidence")
        if isinstance(llm_conf, (int, float)):
            artist_conf = float(llm_conf)
    artist = {
        "label": result["final_artist"],
        "confidence": round(float(artist_conf) * 100, 1),
    }

    analysis_payload = {
        "style": style,
        "artist": artist,
        "top5": style_topk,
        "style_topk": style_topk_raw,
        "artist_topk": result.get("artist_topk", []),
        "final_artist": result["final_artist"],
        "final_style": result.get("final_style"),
        "candidates": result["candidates"],
        "retrieval_hits": result["retrieval_hits"],
        "llm": result["llm"],
        "llm_error": result["llm_error"],
        "confidence": result["confidence"],
        "used_open_world_llm": result["used_open_world_llm"],
        "used_openai_style": result.get("used_openai_style", False),
        "time_period": (result.get("llm") or {}).get("time_period"),
        "emotional_tone": (result.get("llm") or {}).get("emotional_tone"),
        "title": (result.get("llm") or {}).get("title"),
        "context": (result.get("llm") or {}).get("context"),
        "visual_observations": (result.get("llm") or {}).get("visual_observations") or {},
    }
    artwork_id = artwork_store.add(analysis_payload)

    return jsonify({
        **analysis_payload,
        "artwork_id": artwork_id,
    })


@app.post("/api/chat")
def chat_about_artwork():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    artwork_id = payload.get("artwork_id")
    analysis = artwork_store.get(artwork_id if isinstance(artwork_id, str) else None)
    if isinstance(artwork_id, str) and analysis is None:
        return jsonify({"error": "Unknown artwork_id. Analyze the image again before chatting."}), 404
    if analysis is None:
        analysis = payload.get("analysis") or {}
    session_id = payload.get("session_id")

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if not isinstance(analysis, dict):
        return jsonify({"error": "Analysis must be an object."}), 400

    try:
        result = conversation_service.reply(
            message=message,
            analysis=analysis,
            session_id=session_id if isinstance(session_id, str) else None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Chat failed: {exc}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
