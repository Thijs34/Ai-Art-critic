"""Open-world artist recognition pipeline with CLIP embeddings and retrieval."""

from __future__ import annotations

import base64
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision import transforms

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except Exception:
    load_dotenv = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import open_clip
except Exception as exc:  # pragma: no cover
    raise RuntimeError("open_clip_torch is required for CLIP embeddings") from exc


ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "wikiart_test13_style_artist_warmstart_best.pt"
DEFAULT_INDEX_DIR = ROOT / "retrieval_index"
DEFAULT_MAX_INDEX_IMAGES = 2000
DEFAULT_INDEX_SEED = 42
DEFAULT_INDEX_BATCH = 32
STYLE_TRAIN_CSV = ROOT / "datasets" / "Wikiart" / "style_train.csv"
STYLE_VAL_CSV = ROOT / "datasets" / "Wikiart" / "style_val.csv"
ARTIST_TRAIN_CSV = ROOT / "datasets" / "Wikiart" / "artist_train.csv"
ARTIST_VAL_CSV = ROOT / "datasets" / "Wikiart" / "artist_val.csv"

STYLE_NAME_FALLBACK = {
    0: "Abstract Expressionism",
    1: "Action Painting",
    2: "Analytical Cubism",
    3: "Art Nouveau",
    4: "Baroque",
    5: "Color Field Painting",
    6: "Contemporary Realism",
    7: "Cubism",
    8: "Early Renaissance",
    9: "Expressionism",
    10: "Fauvism",
    11: "High Renaissance",
    12: "Impressionism",
    13: "Mannerism / Late Renaissance",
    14: "Minimalism",
    15: "Naive Art / Primitivism",
    16: "New Realism",
    17: "Northern Renaissance",
    18: "Pointillism",
    19: "Pop Art",
    20: "Post-Impressionism",
    21: "Realism",
    22: "Rococo",
    23: "Romanticism",
    24: "Symbolism",
    25: "Synthetic Cubism",
    26: "Ukiyo-e",
}


class StyleArtistModel(nn.Module):
    def __init__(self, model_name: str, image_size: int, n_style: int, n_artist: int):
        super().__init__()
        self.style_model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=n_style,
            img_size=image_size,
        )
        feat_dim = self.style_model.num_features
        self.artist_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.2),
            nn.Linear(feat_dim, n_artist),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.style_model.forward_features(x)
        feats = self.style_model.forward_head(feats, pre_logits=True)
        return feats

    def forward(self, x: torch.Tensor):
        feats = self.extract_features(x)
        style_input = self.style_model.head_drop(feats) if hasattr(self.style_model, "head_drop") else feats
        style_logits = self.style_model.head(style_input)
        artist_logits = self.artist_head(feats)
        return style_logits, artist_logits


def _pretty_style_name(raw_folder: str) -> str:
    name = raw_folder.replace("_", " ")
    name = name.replace("Art Nouveau Modern", "Art Nouveau")
    name = name.replace("Post Impressionism", "Post-Impressionism")
    return name


def _pretty_artist_name(raw_slug: str) -> str:
    parts = re_split(raw_slug.strip())
    if not parts:
        return "Unknown Artist"

    lower_words = {"de", "del", "der", "di", "la", "le", "van", "von", "da"}
    formatted = []
    for i, token in enumerate(parts):
        if not token:
            continue
        token_l = token.lower()
        if i > 0 and token_l in lower_words:
            formatted.append(token_l)
        else:
            formatted.append(token_l.capitalize())
    return " ".join(formatted) if formatted else "Unknown Artist"


def re_split(value: str) -> List[str]:
    parts = []
    token = ""
    for ch in value:
        if ch in "-_":
            if token:
                parts.append(token)
                token = ""
        else:
            token += ch
    if token:
        parts.append(token)
    return parts


def _parse_csv_label_rows(csv_path: Path):
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            rel_path = row[0].strip()
            try:
                label = int(row[1])
            except ValueError:
                continue
            if not rel_path:
                continue
            yield rel_path, label


def load_style_class_names(n_style: int) -> Dict[int, str]:
    style_map: Dict[int, str] = {}
    for csv_path in (STYLE_TRAIN_CSV, STYLE_VAL_CSV):
        for rel_path, label in _parse_csv_label_rows(csv_path):
            style_folder = PurePosixPath(rel_path).parts[0]
            style_map.setdefault(label, _pretty_style_name(style_folder))

    for label in range(n_style):
        if label not in style_map and label in STYLE_NAME_FALLBACK:
            style_map[label] = STYLE_NAME_FALLBACK[label]
        style_map.setdefault(label, f"Style {label}")
    return style_map


def load_artist_class_names(n_artist: int) -> Dict[int, str]:
    artist_map: Dict[int, str] = {}
    for csv_path in (ARTIST_TRAIN_CSV, ARTIST_VAL_CSV):
        for rel_path, label in _parse_csv_label_rows(csv_path):
            file_stem = PurePosixPath(rel_path).stem
            artist_slug = file_stem.split("_", 1)[0] if "_" in file_stem else file_stem
            artist_map.setdefault(label, _pretty_artist_name(artist_slug))

    for label in range(n_artist):
        artist_map.setdefault(label, f"Artist {label}")
    return artist_map


@dataclass
class ModelBundle:
    model: StyleArtistModel
    image_size: int
    style_names: Dict[int, str]
    artist_names: Dict[int, str]
    device: torch.device
    transform: transforms.Compose


def load_model_bundle(model_path: Path, device: Optional[str] = None) -> ModelBundle:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_name = ckpt.get("model_name", cfg.get("model_name", "vit_large_patch14_dinov2"))
    n_style = int(ckpt.get("n_style", ckpt.get("num_classes", len(STYLE_NAME_FALLBACK))))
    n_artist = int(ckpt.get("n_artist", 0))
    image_size = int(cfg.get("image_size", 448))

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = StyleArtistModel(
        model_name=model_name,
        image_size=image_size,
        n_style=n_style,
        n_artist=n_artist,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(torch_device)
    model.eval()

    style_names = load_style_class_names(n_style)
    artist_names = load_artist_class_names(n_artist)

    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ModelBundle(
        model=model,
        image_size=image_size,
        style_names=style_names,
        artist_names=artist_names,
        device=torch_device,
        transform=transform,
    )


def build_topk(probs: torch.Tensor, class_names: Dict[int, str], top_k: int) -> List[Dict[str, Any]]:
    top = torch.topk(probs, k=min(top_k, len(class_names)))
    return [
        {
            "label": class_names.get(idx.item(), f"Class {idx.item()}"),
            "confidence": float(score.item()),
            "index": int(idx.item()),
        }
        for idx, score in zip(top.indices, top.values)
    ]


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    def encode_images(self, images: Sequence[Union[str, Path, Image.Image]]) -> np.ndarray:
        tensors: List[torch.Tensor] = []
        for item in images:
            if isinstance(item, (str, Path)):
                image = Image.open(item).convert("RGB")
            else:
                image = item
            tensors.append(self.preprocess(image))
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_image(batch)
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds.cpu().numpy().astype("float32")


class RetrievalStore:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.metadata_path = self.index_dir / "metadata.jsonl"
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.index = None
        self._load()

    def _load(self) -> None:
        if self.embeddings_path.exists() and self.metadata_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            self.metadata = []
            with self.metadata_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    self.metadata.append(json.loads(line))
            self._build_index()

    def has_data(self) -> bool:
        return self.embeddings is not None and len(self.metadata) > 0

    def _build_index(self) -> None:
        if self.embeddings is None:
            self.index = None
            return
        if faiss is not None:
            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings)
            self.index = index
        else:
            self.index = "numpy"

    def add(self, embeddings: np.ndarray, metadata: Sequence[Dict[str, Any]]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")
        if self.embeddings is None:
            self.embeddings = embeddings.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.metadata.extend(metadata)
        self._build_index()

    def save(self) -> None:
        if self.embeddings is None:
            return
        np.save(self.embeddings_path, self.embeddings)
        with self.metadata_path.open("w", encoding="utf-8") as fh:
            for row in self.metadata:
                fh.write(json.dumps(row) + "\n")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.metadata:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]
        if faiss is not None and self.index is not None and self.index != "numpy":
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                item = dict(self.metadata[int(idx)])
                item["similarity"] = float(score)
                results.append(item)
            return results

        scores = (self.embeddings @ query_embedding[0]).astype("float32")
        top_indices = np.argsort(-scores)[: min(top_k, len(self.metadata))]
        results = []
        for idx in top_indices:
            item = dict(self.metadata[int(idx)])
            item["similarity"] = float(scores[idx])
            results.append(item)
        return results


def _aggregate_retrieval_scores(retrieval_hits: List[Dict[str, Any]]) -> Dict[str, float]:
    if not retrieval_hits:
        return {}
    scores = np.array([hit["similarity"] for hit in retrieval_hits], dtype="float32")
    min_s, max_s = float(scores.min()), float(scores.max())
    if max_s == min_s:
        scaled = [1.0 for _ in retrieval_hits]
    else:
        scaled = [(s - min_s) / (max_s - min_s) for s in scores]

    agg: Dict[str, float] = {}
    for hit, score in zip(retrieval_hits, scaled):
        artist = hit.get("artist", "Unknown Artist")
        agg[artist] = agg.get(artist, 0.0) + float(score)
    return agg


def combine_candidates(
    model_topk: List[Dict[str, Any]],
    retrieval_hits: List[Dict[str, Any]],
    model_weight: float = 0.6,
    retrieval_weight: float = 0.4,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    model_scores = {item["label"]: float(item["confidence"]) for item in model_topk}
    retrieval_scores = _aggregate_retrieval_scores(retrieval_hits)

    candidates = set(model_scores) | set(retrieval_scores)
    ranked = []
    for artist in candidates:
        score = model_weight * model_scores.get(artist, 0.0) + retrieval_weight * retrieval_scores.get(artist, 0.0)
        ranked.append({
            "artist": artist,
            "score": float(score),
            "model_score": float(model_scores.get(artist, 0.0)),
            "retrieval_score": float(retrieval_scores.get(artist, 0.0)),
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


def _encode_image_to_data_url(image: Image.Image) -> str:
    buffer = io_bytes(image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def io_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def refine_with_llm(
    candidates: List[Dict[str, Any]],
    retrieval_hits: List[Dict[str, Any]],
    style_hint: Optional[str] = None,
    model_confidence: Optional[Dict[str, float]] = None,
    image: Optional[Image.Image] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    include_candidates: bool = True,
    include_retrieval: bool = True,
) -> Dict[str, Any]:
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "OPENAI_API_KEY_PLACEHOLDER")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for LLM refinement") from exc

    client = OpenAI(api_key=api_key)

    candidate_lines: List[str] = []
    retrieval_lines: List[str] = []
    if include_candidates:
        candidate_lines = [
            f"- {c['artist']} | combined={c['score']:.3f} | model={c['model_score']:.3f} | retrieval={c['retrieval_score']:.3f}"
            for c in candidates
        ]
        if not candidate_lines:
            candidate_lines = ["- (none)"]
    if include_retrieval:
        retrieval_lines = [
            f"- {hit.get('artist', 'Unknown Artist')} | sim={hit.get('similarity', 0.0):.3f} | path={hit.get('image_path', '')}"
            for hit in retrieval_hits
        ]
        if not retrieval_lines:
            retrieval_lines = ["- (none)"]

    system_prompt = (
        "You are an open-world art attribution assistant. "
        "Candidates and retrieval hits are weak hints, not constraints. "
        "You may ignore them entirely if they do not fit the artwork. "
        "You may suggest an artist not in the dataset, or return 'unknown' if unsure. "
        "Also provide a best-guess style, time period, mood/emotional tone, short display title, "
        "and brief visual context from the artwork. Keep reasoning short and grounded in visible evidence. "
        "The emotional tone should be a concise display phrase, such as 'quiet and contemplative' "
        "or 'tense, somber, and dramatic', inferred from color, light, composition, figures, and texture. "
        "Visual observations should be concrete things visible in the image, not historical claims. "
        "Do not overclaim exact artist, intent, symbolism, or historical context."
    )

    user_lines: List[str] = []
    if include_candidates:
        user_lines.extend(["Candidates:", *candidate_lines, ""])
    if include_retrieval:
        user_lines.extend(["Retrieval hits:", *retrieval_lines])
    user_prompt = "\n".join(user_lines) if user_lines else "No candidate hints provided."
    user_prompt += (
        "\n\nReturn a JSON object with keys: artist, reason, is_unknown, confidence, "
        "style, style_confidence, time_period, emotional_tone, title, context, visual_observations. "
        "Use style_confidence and confidence as numbers from 0 to 1. "
        "Set emotional_tone to 2-6 words and avoid returning an empty string. "
        "The title should be a concise suggested title, not a factual artwork title unless known. "
        "visual_observations must be an object with short strings for palette, composition, lighting, "
        "subject_matter, brushwork_texture, and focal_points."
    )
    if model_confidence:
        user_prompt += (
            "\n\nModel confidence (low means unreliable): "
            f"artist_top1={model_confidence.get('top1', 0.0):.3f}, "
            f"artist_top2={model_confidence.get('top2', 0.0):.3f}, "
            f"artist_margin={model_confidence.get('margin', 0.0):.3f}, "
            f"style_top1={model_confidence.get('style_top1', 0.0):.3f}"
        )
    if style_hint:
        user_prompt += f"\n\nStyle hint: {style_hint}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if image is not None:
        data_url = _encode_image_to_data_url(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Optional image for visual context."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        response_format={
            "type": "json_object",
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty response")

    result = json.loads(content)
    if not isinstance(result, dict):
        raise RuntimeError("LLM returned non-object response")

    artist = result.get("artist") or "unknown"
    is_unknown = bool(result.get("is_unknown")) or artist.strip().lower() == "unknown"
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5 if not is_unknown else 0.0
    confidence = float(min(1.0, max(0.0, confidence)))
    style_confidence = result.get("style_confidence")
    if not isinstance(style_confidence, (int, float)):
        style_confidence = 0.5
    style_confidence = float(min(1.0, max(0.0, style_confidence)))

    return {
        "artist": str(artist),
        "reason": str(result.get("reason") or ""),
        "is_unknown": is_unknown,
        "confidence": confidence,
        "style": str(result.get("style") or ""),
        "style_confidence": style_confidence,
        "time_period": str(result.get("time_period") or ""),
        "emotional_tone": str(result.get("emotional_tone") or ""),
        "title": str(result.get("title") or ""),
        "context": str(result.get("context") or ""),
        "visual_observations": _clean_visual_observations(result.get("visual_observations")),
    }


def _clean_visual_observations(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "palette",
        "composition",
        "lighting",
        "subject_matter",
        "brushwork_texture",
        "focal_points",
    }
    cleaned: Dict[str, str] = {}
    for key in allowed:
        raw = value.get(key)
        if isinstance(raw, list):
            raw = ", ".join(str(item) for item in raw[:4])
        text = " ".join(str(raw or "").split())
        if text:
            cleaned[key] = text[:220]
    return cleaned


def _style_from_rel_path(rel_path: str) -> str:
    folder = PurePosixPath(rel_path).parts[0]
    return _pretty_style_name(folder)


def _artist_from_rel_path(rel_path: str) -> str:
    file_stem = PurePosixPath(rel_path).stem
    artist_slug = file_stem.split("_", 1)[0] if "_" in file_stem else file_stem
    return _pretty_artist_name(artist_slug)


def _sample_artist_rows(
    csv_paths: Sequence[Path],
    max_items: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    sample: List[str] = []
    seen = 0
    for csv_path in csv_paths:
        for rel_path, _ in _parse_csv_label_rows(csv_path):
            seen += 1
            if len(sample) < max_items:
                sample.append(rel_path)
            else:
                idx = rng.randint(0, seen - 1)
                if idx < max_items:
                    sample[idx] = rel_path
    return sample


class OpenWorldArtistPipeline:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: Optional[str] = None,
        auto_init_index: bool = True,
        max_index_images: int = DEFAULT_MAX_INDEX_IMAGES,
        index_seed: int = DEFAULT_INDEX_SEED,
        index_batch_size: int = DEFAULT_INDEX_BATCH,
    ):
        self.bundle = load_model_bundle(model_path, device=device)
        self.embedder = ClipEmbedder(model_name=clip_model, pretrained=clip_pretrained, device=device)
        self.store = RetrievalStore(index_dir=index_dir)
        if auto_init_index:
            self.ensure_index(max_images=max_index_images, seed=index_seed, batch_size=index_batch_size)

    def add_images(
        self,
        image_paths: Sequence[Union[str, Path]],
        artist: str,
        style: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        embeddings = self.embedder.encode_images(image_paths)
        metadata: List[Dict[str, Any]] = []
        for image_path in image_paths:
            entry = {
                "artist": artist,
                "style": style or "",
                "image_path": str(image_path),
            }
            if extra_metadata:
                entry.update(extra_metadata)
            metadata.append(entry)
        self.store.add(embeddings, metadata)
        self.store.save()

    def retrieve(self, image: Union[str, Path, Image.Image], top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.encode_images([image])[0]
        return self.store.search(embedding, top_k=top_k)

    def ensure_index(self, max_images: int, seed: int, batch_size: int) -> None:
        if self.store.has_data():
            return

        csv_paths = [ARTIST_TRAIN_CSV, ARTIST_VAL_CSV]
        sampled_paths = _sample_artist_rows(csv_paths, max_items=max_images, seed=seed)
        if not sampled_paths:
            return

        image_paths: List[Path] = []
        metadata: List[Dict[str, Any]] = []
        for rel_path in sampled_paths:
            full_path = ROOT / "datasets" / "Wikiart" / rel_path
            if not full_path.exists():
                continue
            artist = _artist_from_rel_path(rel_path)
            style = _style_from_rel_path(rel_path)
            image_paths.append(full_path)
            metadata.append({
                "artist": artist,
                "style": style,
                "image_path": str(full_path),
            })

        if not image_paths:
            return

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_meta = metadata[start:start + batch_size]
            embeddings = self.embedder.encode_images(batch_paths)
            self.store.add(embeddings, batch_meta)
        self.store.save()

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
        retrieval_k: int = 8,
        use_llm: bool = True,
        llm_with_image: bool = False,
        confidence_threshold: float = 0.4,
        margin_threshold: float = 0.15,
        style_confidence_threshold: float = 0.4,
        enrich_analysis: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image

        tensor = self.bundle.transform(pil_image).unsqueeze(0).to(self.bundle.device)

        with torch.no_grad():
            style_logits, artist_logits = self.bundle.model(tensor)
            style_logits_flip, artist_logits_flip = self.bundle.model(torch.flip(tensor, dims=[3]))
            style_probs = torch.softmax((style_logits + style_logits_flip) / 2.0, dim=1)[0]
            artist_probs = torch.softmax((artist_logits + artist_logits_flip) / 2.0, dim=1)[0]

        style_topk = build_topk(style_probs, self.bundle.style_names, top_k=top_k)
        artist_topk = build_topk(artist_probs, self.bundle.artist_names, top_k=top_k)

        style_top1_prob = float(style_topk[0]["confidence"]) if style_topk else 0.0
        style_low_confidence = style_top1_prob < style_confidence_threshold
        top1_prob = float(artist_topk[0]["confidence"]) if artist_topk else 0.0
        top2_prob = float(artist_topk[1]["confidence"]) if len(artist_topk) > 1 else 0.0
        prob_margin = top1_prob - top2_prob
        high_confidence = top1_prob >= confidence_threshold and prob_margin >= margin_threshold

        retrieval_hits = self.retrieve(pil_image, top_k=retrieval_k)
        candidates = combine_candidates(artist_topk, retrieval_hits, top_n=top_k * 2)

        final_artist = artist_topk[0]["label"] if artist_topk else "Unknown"
        final_style = style_topk[0]["label"] if style_topk else "Unknown"
        open_world = False
        used_openai_style = False
        llm_result = None
        llm_error = None
        needs_artist_fallback = not high_confidence
        if use_llm and (needs_artist_fallback or style_low_confidence or enrich_analysis):
            style_hint = style_topk[0]["label"] if style_topk else None
            try:
                use_hints = os.environ.get("OPEN_WORLD_LLM_USE_HINTS", "false").lower() in {"1", "true", "yes"}
                llm_result = refine_with_llm(
                    candidates=candidates,
                    retrieval_hits=retrieval_hits,
                    style_hint=style_hint,
                    model_confidence={
                        "top1": top1_prob,
                        "top2": top2_prob,
                        "margin": prob_margin,
                        "style_top1": style_top1_prob,
                    },
                    image=pil_image if llm_with_image else None,
                    include_candidates=use_hints,
                    include_retrieval=use_hints,
                )
                if needs_artist_fallback:
                    if llm_result.get("is_unknown"):
                        final_artist = "unknown"
                    else:
                        final_artist = llm_result.get("artist", final_artist)
                if style_low_confidence and llm_result.get("style"):
                    final_style = llm_result.get("style", final_style)
                    used_openai_style = True
                open_world = True
            except Exception as exc:  # pragma: no cover
                llm_error = str(exc)
                if needs_artist_fallback:
                    final_artist = "unknown"
                open_world = True

        return {
            "style_topk": style_topk,
            "artist_topk": artist_topk,
            "retrieval_hits": retrieval_hits,
            "candidates": candidates,
            "final_artist": final_artist,
            "final_style": final_style,
            "llm": llm_result,
            "llm_error": llm_error,
            "confidence": {
                "top1": top1_prob,
                "top2": top2_prob,
                "margin": prob_margin,
                "high_confidence": high_confidence,
                "style_top1": style_top1_prob,
                "style_low_confidence": style_low_confidence,
            },
            "used_open_world_llm": open_world,
            "used_openai_style": used_openai_style,
        }
