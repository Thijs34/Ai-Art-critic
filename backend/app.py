"""Lumora API serving the Test 13 multitask style+artist model."""

import csv
import os
import re
from pathlib import Path, PurePosixPath

import timm
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_FILENAME = "wikiart_test13_style_artist_warmstart_best.pt"
MODEL_PATH = ROOT / "models" / MODEL_FILENAME
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
    parts = re.split(r"[-_]", raw_slug.strip())
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


def load_style_class_names(n_style: int):
    style_map = {}
    for csv_path in (STYLE_TRAIN_CSV, STYLE_VAL_CSV):
        for rel_path, label in _parse_csv_label_rows(csv_path):
            style_folder = PurePosixPath(rel_path).parts[0]
            style_map.setdefault(label, _pretty_style_name(style_folder))

    for label in range(n_style):
        if label not in style_map and label in STYLE_NAME_FALLBACK:
            style_map[label] = STYLE_NAME_FALLBACK[label]
        style_map.setdefault(label, f"Style {label}")
    return style_map


def load_artist_class_names(n_artist: int):
    artist_map = {}
    for csv_path in (ARTIST_TRAIN_CSV, ARTIST_VAL_CSV):
        for rel_path, label in _parse_csv_label_rows(csv_path):
            file_stem = PurePosixPath(rel_path).stem
            artist_slug = file_stem.split("_", 1)[0] if "_" in file_stem else file_stem
            artist_map.setdefault(label, _pretty_artist_name(artist_slug))

    for label in range(n_artist):
        artist_map.setdefault(label, f"Artist {label}")
    return artist_map


def build_topk(probs: torch.Tensor, class_names: dict[int, str], top_k: int):
    top = torch.topk(probs, k=min(top_k, len(class_names)))
    return [
        {
            "label": class_names.get(idx.item(), f"Class {idx.item()}"),
            "confidence": round(score.item() * 100, 1),
        }
        for idx, score in zip(top.indices, top.values)
    ]


# ── Download model from Hugging Face Hub if not present ───────────────────────
HF_REPO = os.environ.get("HF_MODEL_REPO")
if HF_REPO and not MODEL_PATH.exists():
    print(f"Model not found locally - downloading from {HF_REPO}...")
    from huggingface_hub import hf_hub_download

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_PATH.parent),
    )
    print(f"Model downloaded to {downloaded}")


# ── Model loading ──────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
MODEL_NAME = ckpt.get("model_name", cfg.get("model_name", "vit_large_patch14_dinov2"))
N_STYLE = int(ckpt.get("n_style", ckpt.get("num_classes", len(STYLE_NAME_FALLBACK))))
N_ARTIST = int(ckpt.get("n_artist", 0))
IMAGE_SIZE = int(cfg.get("image_size", 448))

model = StyleArtistModel(
    model_name=MODEL_NAME,
    image_size=IMAGE_SIZE,
    n_style=N_STYLE,
    n_artist=N_ARTIST,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

STYLE_CLASS_NAMES = load_style_class_names(N_STYLE)
ARTIST_CLASS_NAMES = load_artist_class_names(N_ARTIST)

print(
    "Model ready - "
    f"style classes: {N_STYLE}, artist classes: {N_ARTIST}, "
    f"best val style top-1: {ckpt.get('best_val_style_top1', '?')}"
)

# ── Eval transform (matches test13 eval transform) ────────────────────────────
_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image: Image.Image, top_k: int = 5):
    """Return top-k style and artist predictions with confidence scores."""
    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        style_logits, artist_logits = model(tensor)
        style_logits_flip, artist_logits_flip = model(torch.flip(tensor, dims=[3]))

        style_probs = torch.softmax((style_logits + style_logits_flip) / 2.0, dim=1)[0]
        artist_probs = torch.softmax((artist_logits + artist_logits_flip) / 2.0, dim=1)[0]

    style_topk = build_topk(style_probs, STYLE_CLASS_NAMES, top_k=top_k)
    artist_topk = build_topk(artist_probs, ARTIST_CLASS_NAMES, top_k=top_k)
    return style_topk, artist_topk


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "tasks": ["style", "artist"],
        "image_size": IMAGE_SIZE,
    })


@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the file under the key 'image'."}), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Could not open image: {exc}"}), 400

    style_top5, artist_top5 = predict(image, top_k=5)
    top_style = style_top5[0]
    top_artist = artist_top5[0]

    return jsonify({
        "style": {
            "label": top_style["label"],
            "confidence": top_style["confidence"],
        },
        "artist": {
            "label": top_artist["label"],
            "confidence": top_artist["confidence"],
        },
        "top5": style_top5,
        "artist_top5": artist_top5,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
