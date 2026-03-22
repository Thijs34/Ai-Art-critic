"""
Lumora – Art Style Classification API
Serves the wikiart_test7 ViT model for art style prediction.
"""

from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "wikiart_test7_best.pt"

# ── Class names (label index → human-readable style) ──────────────────────────
CLASS_NAMES = {
    0:  "Abstract Expressionism",
    1:  "Action Painting",
    2:  "Analytical Cubism",
    3:  "Art Nouveau",
    4:  "Baroque",
    5:  "Color Field Painting",
    6:  "Contemporary Realism",
    7:  "Cubism",
    8:  "Early Renaissance",
    9:  "Expressionism",
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

# ── Model loading ──────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

model = timm.create_model(
    ckpt["model_name"],
    pretrained=False,
    num_classes=ckpt["num_classes"],
)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model ready — {ckpt['num_classes']} classes, best val top-1: {ckpt.get('best_val_top1', '?'):.4f}")

# ── Eval transform (matches training code exactly) ─────────────────────────────
IMAGE_SIZE = 384
_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),   # 438 px
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image: Image.Image, top_k: int = 5):
    """Return top-k style predictions with confidence scores."""
    tensor = _transform(image).unsqueeze(0)          # (1, 3, 384, 384)

    with torch.no_grad():
        logits = model(tensor)
        # TTA: average with horizontal flip (same as training eval)
        logits_flip = model(torch.flip(tensor, dims=[3]))
        logits = (logits + logits_flip) / 2.0

        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    results = [
        {
            "label":      CLASS_NAMES[idx.item()],
            "confidence": round(score.item() * 100, 1),
        }
        for idx, score in zip(top.indices, top.values)
    ]
    return results


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": ckpt["model_name"]})


@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the file under the key 'image'."}), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Could not open image: {exc}"}), 400

    predictions = predict(image, top_k=5)
    top = predictions[0]

    return jsonify({
        "style": {
            "label":      top["label"],
            "confidence": top["confidence"],
        },
        "top5": predictions,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
