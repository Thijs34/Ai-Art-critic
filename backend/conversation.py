"""Conversational art critic layer with lightweight retrieval and memory."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).parent.parent
ARTEMIS_CSV = ROOT / "datasets" / "ArtEmis" / "Contrastive_with_paths.csv"
CHAT_EMBEDDING_CACHE = ROOT / "retrieval_index" / "art_chat_embedding_cache.json"


STYLE_REFERENCES: Dict[str, str] = {
    "Abstract Expressionism": "Abstract Expressionism often values gesture, scale, spontaneity, and the trace of the painter's body. Conversation can focus on movement, pressure, rhythm, and emotional charge.",
    "Action Painting": "Action Painting foregrounds the act of painting itself: drips, sweeps, splashes, and marks become evidence of motion and decision.",
    "Analytical Cubism": "Analytical Cubism breaks forms into angled planes and muted facets, asking viewers to assemble space from partial viewpoints.",
    "Art Nouveau": "Art Nouveau favors flowing lines, botanical curves, decorative surfaces, and a close relationship between figure, ornament, and design.",
    "Baroque": "Baroque art often uses theatrical light, dramatic gesture, diagonal movement, and emotional immediacy to pull the viewer into the scene.",
    "Color Field Painting": "Color Field Painting relies on broad areas of color, soft edges, and immersive chromatic relationships rather than narrative detail.",
    "Contemporary Realism": "Contemporary Realism uses recognizable representation while often sharpening everyday surfaces, quiet tension, or modern observation.",
    "Cubism": "Cubism fractures viewpoint and object boundaries, turning composition into a structure of planes, edges, and simultaneous perspectives.",
    "Early Renaissance": "Early Renaissance painting often balances religious or civic symbolism with emerging perspective, careful anatomy, and clear spatial order.",
    "Expressionism": "Expressionism distorts color, line, and form to make inner feeling visible; emotional truth can matter more than optical accuracy.",
    "Fauvism": "Fauvism uses heightened, non-naturalistic color and simplified form to create vivid emotional temperature.",
    "High Renaissance": "High Renaissance art is associated with balanced composition, idealized anatomy, controlled perspective, and calm monumentality.",
    "Impressionism": "Impressionism is attentive to changing light, broken brushwork, fleeting perception, and ordinary scenes seen as moments rather than fixed monuments.",
    "Mannerism / Late Renaissance": "Mannerism often stretches proportion, complicates poses, and creates elegant artificiality after the High Renaissance ideal.",
    "Minimalism": "Minimalism reduces form, gesture, and narrative, making scale, repetition, material, and the viewer's attention central.",
    "Naive Art / Primitivism": "Naive Art and Primitivism often use simplified drawing, direct symbolism, flattened space, and deliberately unacademic handling.",
    "New Realism": "New Realism re-engages the visible world with directness, often stressing material presence, social context, or ordinary subject matter.",
    "Northern Renaissance": "Northern Renaissance work is known for precise surfaces, symbolic detail, luminous oil technique, and intimate observation.",
    "Pointillism": "Pointillism builds light and color through small adjacent marks, making optical mixture and surface vibration important.",
    "Pop Art": "Pop Art borrows from mass media, advertising, comics, and consumer imagery, often mixing irony with bold graphic clarity.",
    "Post-Impressionism": "Post-Impressionism extends Impressionism toward stronger structure, symbolic color, expressive contour, or personal vision.",
    "Realism": "Realism pays attention to observed life, social presence, ordinary bodies, and material detail without idealizing the subject too heavily.",
    "Rococo": "Rococo often favors lightness, intimacy, playful asymmetry, pastel color, and decorative elegance.",
    "Romanticism": "Romanticism emphasizes emotion, atmosphere, sublime nature, individuality, and dramatic contrasts between human feeling and the world.",
    "Symbolism": "Symbolism treats visible things as carriers of dream, myth, spirituality, psychology, or private meaning.",
    "Synthetic Cubism": "Synthetic Cubism tends toward flatter shapes, collage-like construction, pattern, and signs of everyday material culture.",
    "Ukiyo-e": "Ukiyo-e uses crisp contour, flattened color, asymmetry, pattern, and scenes of transient beauty from urban life, theatre, landscape, and pleasure districts.",
}


TECHNIQUE_REFERENCES: Dict[str, str] = {
    "brushwork": "When discussing brushwork, connect visible mark size, edge softness, direction, and layering to the feeling of speed, care, pressure, or atmosphere.",
    "composition": "Composition can be read through focal points, diagonals, symmetry, cropping, negative space, rhythm, and how the eye travels through the image.",
    "symbolism": "Symbolic readings should stay tentative unless there is clear evidence. Mention objects, colors, gestures, setting, and repeated motifs as possible clues.",
    "emotion": "Emotional interpretation works best when tied to visual evidence: palette, light, posture, distance, texture, weather, and the scale of forms.",
    "museum label": "Museum-style interpretation should be concise, concrete, and invitational: identify what is visible, then suggest why it matters.",
    "technique": "Technique discussion can cover medium-like effects, layering, contour, tonal modelling, texture, palette, perspective, and surface finish.",
}


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "so", "that", "the", "this",
    "to", "what", "with", "you", "your", "about", "can", "could", "does", "tell",
}


@dataclass
class RetrievedContext:
    title: str
    text: str
    source: str
    score: float
    kind: str = "reference"
    grounding: str = "supporting context"


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower()) if tok not in STOPWORDS]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_percent(value: Any) -> Optional[float]:
    score = _safe_float(value)
    if score is None:
        return None
    if 0 <= score <= 1:
        score *= 100
    return round(max(0.0, min(100.0, score)), 1)


def _norm_label(value: Any, default: str = "Unknown") -> str:
    if isinstance(value, dict):
        value = value.get("label")
    if not value:
        return default
    return str(value).strip() or default


def build_artwork_profile(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Shape model output into a compact profile the chat layer can trust."""
    if not isinstance(analysis, dict):
        analysis = {}

    style = _norm_label(analysis.get("style"))
    artist = _norm_label(analysis.get("artist") or analysis.get("final_artist"), "Unknown Artist")
    top_styles = analysis.get("top5") or analysis.get("style_topk") or []
    confidence = analysis.get("confidence") if isinstance(analysis.get("confidence"), dict) else {}
    llm = analysis.get("llm") if isinstance(analysis.get("llm"), dict) else {}
    style_confidence = _confidence_percent((analysis.get("style") or {}).get("confidence") if isinstance(analysis.get("style"), dict) else None)
    artist_confidence = _confidence_percent((analysis.get("artist") or {}).get("confidence") if isinstance(analysis.get("artist"), dict) else None)
    artist_is_unknown = artist.lower() in {"unknown", "unknown artist", "none"}
    style_is_unknown = style.lower() in {"unknown", "none"}
    low_confidence = bool(confidence.get("high_confidence") is False)
    if style_is_unknown or artist_is_unknown:
        low_confidence = True
    if style_confidence is not None and style_confidence < 45:
        low_confidence = True
    if artist_confidence is not None and artist_confidence < 45:
        low_confidence = True

    return {
        "style": style,
        "artist": artist,
        "style_confidence": style_confidence,
        "artist_confidence": artist_confidence,
        "top_styles": [
            {
                "label": item.get("label"),
                "confidence": _confidence_percent(item.get("confidence")),
            }
            for item in top_styles[:5]
            if isinstance(item, dict)
        ],
        "time_period": analysis.get("timePeriod") or analysis.get("time_period") or llm.get("time_period"),
        "title": analysis.get("artworkTitle") or analysis.get("title") or llm.get("title"),
        "emotional_tone": analysis.get("emotionalTone") or analysis.get("emotional_tone") or llm.get("emotional_tone"),
        "context": analysis.get("context") or llm.get("context"),
        "retrieval_hits": [
            {
                "artist": hit.get("artist"),
                "style": hit.get("style"),
                "similarity": hit.get("similarity"),
            }
            for hit in (analysis.get("retrieval_hits") or [])[:5]
            if isinstance(hit, dict)
        ],
        "low_confidence": low_confidence,
        "artist_is_unknown": artist_is_unknown,
        "style_is_unknown": style_is_unknown,
        "grounding_guidance": _grounding_guidance(style_confidence, artist_confidence, low_confidence, artist_is_unknown),
    }


def _grounding_guidance(
    style_confidence: Optional[float],
    artist_confidence: Optional[float],
    low_confidence: bool,
    artist_is_unknown: bool,
) -> List[str]:
    guidance = [
        "Tie interpretation to visible evidence: color, composition, brushwork, texture, light, focal points, and mood.",
        "Treat symbolism, intent, and exact historical context as tentative unless profile evidence supports them.",
    ]
    if low_confidence:
        guidance.append("Model confidence is limited; phrase style and attribution as possibilities, not facts.")
    if artist_is_unknown or (artist_confidence is not None and artist_confidence < 55):
        guidance.append("Avoid strong artist attribution; discuss visual resemblance or stylistic affinity instead.")
    if style_confidence is not None and style_confidence < 55:
        guidance.append("Use the predicted style as a lens, while acknowledging nearby alternatives from top styles.")
    return guidance


class PersistentEmbeddingCache:
    """Small JSON-backed cache so semantic retrieval does not re-embed the same notes."""

    def __init__(self, path: Path = CHAT_EMBEDDING_CACHE, max_entries: Optional[int] = None):
        self.path = path
        self.max_entries = max_entries or int(os.environ.get("ART_RAG_EMBED_CACHE_MAX", 3000))
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
                self.entries = payload["entries"]
        except (OSError, json.JSONDecodeError):
            self.entries = {}

    def key(self, model: str, text: str) -> str:
        normalized = " ".join(text[:4000].split())
        digest = hashlib.sha256(f"{model}\n{normalized}".encode("utf-8")).hexdigest()
        return digest

    def get(self, model: str, text: str) -> Optional[List[float]]:
        self.load()
        entry = self.entries.get(self.key(model, text))
        if not isinstance(entry, dict):
            return None
        vector = entry.get("vector")
        if not isinstance(vector, list):
            return None
        entry["last_used"] = time.time()
        return vector

    def set(self, model: str, text: str, vector: Sequence[float]) -> None:
        self.load()
        self.entries[self.key(model, text)] = {
            "model": model,
            "text_preview": text[:160],
            "vector": list(vector),
            "created_at": time.time(),
            "last_used": time.time(),
        }
        self._trim()
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".tmp")
            payload = {
                "version": 1,
                "description": "Lazy cache for optional art chat semantic retrieval embeddings.",
                "entries": self.entries,
            }
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            tmp_path.replace(self.path)
        except OSError:
            return

    def _trim(self) -> None:
        if len(self.entries) <= self.max_entries:
            return
        ranked = sorted(
            self.entries.items(),
            key=lambda item: _safe_float(item[1].get("last_used"), 0.0) or 0.0,
            reverse=True,
        )
        self.entries = dict(ranked[: self.max_entries])


class OptionalSemanticReranker:
    """Tiny optional semantic path using OpenAI embeddings when explicitly enabled."""

    def __init__(self):
        self.enabled = os.environ.get("ART_RAG_USE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}
        self.model = os.environ.get("ART_RAG_EMBED_MODEL", "text-embedding-3-small")
        self._client = None
        self.cache = PersistentEmbeddingCache()
        self.last_cache_hits = 0
        self.last_cache_misses = 0

    def rerank(self, query: str, docs: Sequence[RetrievedContext]) -> List[RetrievedContext]:
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        if not self.enabled or not docs or not os.environ.get("OPENAI_API_KEY"):
            return list(docs)
        try:
            query_vec = self._embed(query)
            ranked = []
            for doc in docs:
                doc_vec = self._embed(f"{doc.title}\n{doc.text}")
                sem_score = _cosine(query_vec, doc_vec)
                ranked.append(RetrievedContext(
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    score=(doc.score * 0.65) + (sem_score * 0.35),
                    kind=doc.kind,
                    grounding=doc.grounding,
                ))
            return sorted(ranked, key=lambda item: item.score, reverse=True)
        except Exception:
            return list(docs)

    def _embed(self, text: str) -> List[float]:
        text = text[:4000]
        cached = self.cache.get(self.model, text)
        if cached is not None:
            self.last_cache_hits += 1
            return cached
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = self._client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        self.last_cache_misses += 1
        self.cache.set(self.model, text, vector)
        return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class LightweightArtRetriever:
    """Small retriever for style, technique, and ArtEmis emotional examples."""

    def __init__(self, artemis_path: Path = ARTEMIS_CSV, max_artemis_rows: Optional[int] = None):
        self.artemis_path = artemis_path
        self.max_artemis_rows = max_artemis_rows or int(os.environ.get("ARTEMIS_RAG_MAX_ROWS", 6000))
        self._artemis_loaded = False
        self._artemis_docs: List[RetrievedContext] = []
        self.semantic = OptionalSemanticReranker()

    def retrieve(self, profile: Dict[str, Any], query: str, limit: int = 6) -> List[RetrievedContext]:
        buckets: Dict[str, List[RetrievedContext]] = {
            "style": self._retrieve_style(profile),
            "technique": self._retrieve_technique(query),
            "artemis": self._retrieve_artemis(profile, query, limit=limit * 3),
        }
        candidates = [doc for docs in buckets.values() for doc in docs]
        candidates = self.semantic.rerank(self._semantic_query(profile, query), candidates)
        return self._balanced(candidates, limit=limit)

    def _retrieve_style(self, profile: Dict[str, Any]) -> List[RetrievedContext]:
        style = profile.get("style") or ""
        if style in STYLE_REFERENCES:
            confidence = profile.get("style_confidence")
            score = 1.15 if confidence is None or confidence >= 55 else 0.85
            return [RetrievedContext(
                f"Movement: {style}",
                STYLE_REFERENCES[style],
                "curated-style",
                score,
                kind="style",
                grounding="style lens, not definitive attribution",
            )]
        return []

    def _retrieve_technique(self, query: str) -> List[RetrievedContext]:
        docs: List[RetrievedContext] = []
        query_tokens = set(_tokenize(query))
        for key, text in TECHNIQUE_REFERENCES.items():
            key_tokens = set(_tokenize(key + " " + text))
            overlap = len(query_tokens & key_tokens)
            baseline = key in {"brushwork", "composition", "emotion"}
            if key in query.lower() or overlap or baseline:
                docs.append(RetrievedContext(
                    f"Interpretive lens: {key}",
                    text,
                    "curated-technique",
                    0.72 + min(0.25, overlap * 0.06),
                    kind="technique",
                    grounding="visual-analysis lens",
                ))
        return sorted(docs, key=lambda doc: doc.score, reverse=True)

    def _retrieve_artemis(self, profile: Dict[str, Any], query: str, limit: int) -> List[RetrievedContext]:
        self._load_artemis()
        if not self._artemis_docs:
            return []

        query_tokens = set(_tokenize(" ".join([
            query,
            str(profile.get("style") or ""),
            str(profile.get("context") or ""),
        ])))
        style = str(profile.get("style") or "").lower()
        scored: List[RetrievedContext] = []
        for doc in self._artemis_docs:
            tokens = set(_tokenize(doc.text + " " + doc.title))
            score = len(query_tokens & tokens) / max(4, len(query_tokens))
            if style and style in doc.title.lower():
                score += 0.22
            if score > 0:
                scored.append(RetrievedContext(
                    doc.title,
                    doc.text,
                    doc.source,
                    min(0.78, score),
                    kind="emotional-parallel",
                    grounding="ArtEmis emotional parallel, not factual evidence",
                ))
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:limit]

    def _load_artemis(self) -> None:
        if self._artemis_loaded:
            return
        self._artemis_loaded = True
        if not self.artemis_path.exists():
            return

        per_emotion: Dict[str, int] = defaultdict(int)
        docs: List[RetrievedContext] = []
        try:
            with self.artemis_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    utterance = (row.get("utterance") or "").strip()
                    emotion = (row.get("emotion") or "").strip()
                    style = (row.get("art_style") or "").strip()
                    if not utterance or not emotion or len(utterance) < 24:
                        continue
                    if per_emotion[emotion] >= 300:
                        continue
                    per_emotion[emotion] += 1
                    docs.append(RetrievedContext(
                        title=f"ArtEmis emotional example: {emotion} / {style}",
                        text=utterance,
                        source="ArtEmis",
                        score=0.0,
                        kind="emotional-parallel",
                        grounding="emotional phrasing reference",
                    ))
                    if len(docs) >= self.max_artemis_rows:
                        break
        except (OSError, csv.Error):
            docs = []
        self._artemis_docs = docs

    def _balanced(self, docs: Sequence[RetrievedContext], limit: int) -> List[RetrievedContext]:
        by_kind: Dict[str, List[RetrievedContext]] = defaultdict(list)
        for doc in sorted(docs, key=lambda item: item.score, reverse=True):
            by_kind[doc.kind].append(doc)

        selected: List[RetrievedContext] = []
        caps = {"style": 1, "technique": 2, "emotional-parallel": 2}
        for kind in ("style", "technique", "emotional-parallel"):
            for doc in by_kind.get(kind, [])[:caps.get(kind, 1)]:
                if len(selected) < limit:
                    selected.append(doc)

        seen = {(doc.title, doc.source) for doc in selected}
        for doc in sorted(docs, key=lambda item: item.score, reverse=True):
            key = (doc.title, doc.source)
            if key not in seen and len(selected) < limit:
                selected.append(doc)
                seen.add(key)
        return selected

    def _semantic_query(self, profile: Dict[str, Any], query: str) -> str:
        return " ".join([
            query,
            str(profile.get("style") or ""),
            str(profile.get("context") or ""),
            "visual evidence color composition brushwork texture mood symbolism",
        ])


class ConversationMemory:
    def __init__(self, max_turns: int = 8, ttl_seconds: int = 60 * 60 * 4):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create(self, session_id: Optional[str]) -> str:
        self._prune()
        if session_id and session_id in self._sessions:
            self._sessions[session_id]["updated_at"] = time.time()
            return session_id
        new_id = str(uuid.uuid4())
        self._sessions[new_id] = {"updated_at": time.time(), "turns": deque(maxlen=self.max_turns)}
        return new_id

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.setdefault(
            session_id,
            {"updated_at": time.time(), "turns": deque(maxlen=self.max_turns)},
        )
        session["updated_at"] = time.time()
        turns: Deque[Dict[str, str]] = session["turns"]
        turns.append({"role": role, "content": content[:1600]})

    def history(self, session_id: str) -> List[Dict[str, str]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return list(session["turns"])

    def relevant_history(self, session_id: str, query: str, max_messages: int = 6) -> List[Dict[str, str]]:
        history = self.history(session_id)
        if len(history) <= max_messages:
            return history
        query_tokens = set(_tokenize(query))
        recent = history[-4:]
        older_scored: List[Tuple[int, Dict[str, str]]] = []
        for turn in history[:-4]:
            turn_tokens = set(_tokenize(turn.get("content", "")))
            older_scored.append((len(query_tokens & turn_tokens), turn))
        older = [turn for score, turn in sorted(older_scored, key=lambda item: item[0], reverse=True) if score > 0]
        combined = older[: max(0, max_messages - len(recent))] + recent
        return combined[-max_messages:]

    def _prune(self) -> None:
        now = time.time()
        expired = [
            session_id for session_id, session in self._sessions.items()
            if now - session.get("updated_at", 0) > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)


class ArtConversationService:
    def __init__(self):
        self.retriever = LightweightArtRetriever()
        self.memory = ConversationMemory()

    def reply(self, message: str, analysis: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("Message is required.")

        profile = build_artwork_profile(analysis or {})
        active_session = self.memory.get_or_create(session_id)
        retrieved = self.retriever.retrieve(profile, message, limit=6)
        history = self.memory.relevant_history(active_session, message, max_messages=6)
        answer = self._generate_answer(message, profile, retrieved, history)

        self.memory.add_turn(active_session, "user", message)
        self.memory.add_turn(active_session, "assistant", answer)

        return {
            "session_id": active_session,
            "answer": answer,
            "profile": profile,
            "references": [
                {
                    "title": doc.title,
                    "source": doc.source,
                    "type": doc.kind,
                    "grounding": doc.grounding,
                    "score": round(doc.score, 3),
                }
                for doc in retrieved[:4]
            ],
            "metadata": {
                "low_confidence": profile.get("low_confidence", False),
                "retrieval_sources": sorted({doc.source for doc in retrieved}),
                "retrieval_types": sorted({doc.kind for doc in retrieved}),
                "used_semantic_retrieval": self.retriever.semantic.enabled,
                "semantic_cache_hits": self.retriever.semantic.last_cache_hits,
                "semantic_cache_misses": self.retriever.semantic.last_cache_misses,
                "history_messages_sent": len(history),
            },
        }

    def _generate_answer(
        self,
        message: str,
        profile: Dict[str, Any],
        retrieved: Sequence[RetrievedContext],
        history: Sequence[Dict[str, str]],
    ) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "OPENAI_API_KEY_PLACEHOLDER":
            return self._local_answer(message, profile, retrieved)

        try:
            from openai import OpenAI
        except Exception:
            return self._local_answer(message, profile, retrieved)

        client = OpenAI(api_key=api_key)
        model = os.environ.get("ART_CHAT_MODEL", "gpt-4o-mini")
        system_prompt = (
            "You are an art critic and museum educator. Stay grounded in the artwork profile "
            "and retrieved notes. Be warm, specific, and conversational. Explain visible evidence: "
            "color, composition, brushwork, texture, light, focal points, mood, and possible symbols. "
            "Do not overclaim artist, intent, symbolism, or history. ArtEmis notes are emotional "
            "parallels only, not facts about the uploaded artwork. Do not invent specific objects, "
            "colors, or events unless they appear in the profile or notes; frame unknowns as ways to look. "
            "Avoid generic reassurance and classroom filler. Keep replies under 180 words. Format for chat "
            "readability with short paragraphs, **bold** emphasis, or bullet points when useful."
        )
        context_block = json.dumps({
            "artwork_profile": profile,
            "retrieved_notes": [
                {
                    "title": doc.title,
                    "source": doc.source,
                    "type": doc.kind,
                    "grounding": doc.grounding,
                    "text": doc.text,
                }
                for doc in retrieved
            ],
        }, ensure_ascii=True)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}"},
        ]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": message})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=260,
            )
            content = response.choices[0].message.content
            return (content or "").strip() or self._local_answer(message, profile, retrieved)
        except Exception:
            return self._local_answer(message, profile, retrieved)

    def _local_answer(
        self,
        message: str,
        profile: Dict[str, Any],
        retrieved: Sequence[RetrievedContext],
    ) -> str:
        style = profile.get("style") or "the predicted style"
        artist = profile.get("artist") or "an unknown artist"
        uncertain = profile.get("low_confidence") or profile.get("artist_is_unknown")
        attribution = (
            f"The model points toward {artist}, but I would treat that as tentative."
            if uncertain
            else f"The model points toward {artist}."
        )
        style_line = (
            f"I would use {style} as a viewing lens"
            if uncertain
            else f"I would read this through {style}"
        )
        answer = (
            f"{style_line}. {attribution}\n\n"
            "**Start with visible evidence:**\n"
            "- The color temperature and contrast.\n"
            "- Where the eye is pulled first.\n"
            "- Whether the edges feel hard, soft, quick, careful, or graphic."
        )
        technique = next((doc for doc in retrieved if doc.kind == "technique"), None)
        style_ref = next((doc for doc in retrieved if doc.kind == "style"), None)
        emotion = next((doc for doc in retrieved if doc.kind == "emotional-parallel"), None)
        if technique:
            answer += f"\n\n**Technique lens:** {technique.text}"
        elif style_ref:
            answer += f"\n\n**Style lens:** {style_ref.text}"
        if emotion:
            answer += f"\n\n**Emotional parallel:** ArtEmis offers language like: {emotion.text}"
        if profile.get("context"):
            answer += f"\n\n**Context note:** {profile['context']}"
        return answer[:950]
