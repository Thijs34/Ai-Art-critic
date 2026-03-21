
## wikiart_style_classification_test1.ipynb

- [M0] # WikiArt Style Classifier (Improved Version) This is a **new notebook** created to test improvements over the previous version. Why a new notebook? - The earlier model reached around 50-60% validation accuracy - We want better learning and better generalizati...
- [M1] ## 1. Project Overview Goal: predict the artistic style of a painting (for example Impressionism, Cubism, Realism) from an image using PyTorch and torchvision. We will: - automatically find WikiArt CSV files and image folders - build a custom `Dataset` - train...
- [M2] ## 2. Import Libraries These imports cover data loading, image transforms, modeling, and training.
- [C3]
```python
from pathlib import Path
import random
import warnings
import copy
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
# Ignore PIL warning for very large source images (we resize before training).
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```
- [M4] ## 3. Dataset Explanation and Preview WikiArt data is organized in two ways: - image folders by style (for example `Impressionism/...jpg`) - CSV files with `relative_path,label` (no header) The code below automatically finds the project root and the train/vali...
- [C5]
```python
def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "datasets").exists() and (candidate / "README.md").exists():
            return candidate
    return start.resolve()
def find_split_csv(data_dir: Path, split: str) -> Path:
    split = split.lower()
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    preferred = [p for p in csvs if split in p.stem.lower() and "style" in p.stem.lower()]
    if preferred:
        return preferred[0]
    fallback = [p for p in csvs if split in p.stem.lower()]
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Could not find a '{split}' CSV in {data_dir}")
```
- [C6]
```python
# Build a label -> style name map using the folder name in relative path
def extract_style_name(relative_path: str) -> str:
    return Path(relative_path).parts[0]
train_df["style_name"] = train_df["relative_path"].map(extract_style_name)
val_df["style_name"] = val_df["relative_path"].map(extract_style_name)
label_to_style = (
    train_df[["label", "style_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["style_name"]
    .to_dict()
)
num_classes = len(label_to_style)
print(f"Number of style classes: {num_classes}")
print("First 10 class mappings:")
for lbl, style in list(label_to_style.items())[:10]:
    print(f"{lbl:2d} -> {style}")
```
- [M7] ## 4. Image Transformations (Augmentation + Normalization) For training, we add simple augmentation to improve generalization: - random horizontal flip - small random rotation - light color jitter For validation/test, we use only deterministic transforms. All ...
- [C8]
```python
image_size = 224
batch_size = 32
num_workers = 0  # safer default for Windows
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
```
- [C10]
```python
class WikiArtStyleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, transform=None, max_retries: int = 10):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.max_retries = max_retries
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        last_error = None
        for _ in range(self.max_retries):
            row = self.df.iloc[idx]
            img_path = self.image_root / row["relative_path"]
            label = int(row["label"])
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
```
- [M11] ## 6. Create DataLoaders (Train / Validation / Optional Test) The dataset has train + validation CSVs only. If enabled, we split **15% of validation** into a small test set for final evaluation.
- [C12]
```python
def filter_existing_rows(df: pd.DataFrame, image_root: Path, split_name: str) -> pd.DataFrame:
    full_paths = df["relative_path"].map(lambda p: image_root / p)
    exists_mask = full_paths.map(lambda p: p.exists())
    removed = int((~exists_mask).sum())
    kept = int(exists_mask.sum())
    print(f"{split_name}: kept {kept:,}, removed {removed:,} missing files")
    cleaned = df.loc[exists_mask].reset_index(drop=True)
    if cleaned.empty:
        raise RuntimeError(f"No valid rows left for split: {split_name}")
    return cleaned
train_df_clean = filter_existing_rows(train_df, wikiart_dir, "train")
val_df_clean = filter_existing_rows(val_df, wikiart_dir, "validation")
CREATE_TEST_SPLIT = True
TEST_FRACTION_FROM_VAL = 0.15  # 10-20% is a good range
if CREATE_TEST_SPLIT and len(val_df_clean) > 20:
    test_df = val_df_clean.sample(frac=TEST_FRACTION_FROM_VAL, random_state=SEED)
    val_eval_df = val_df_clean.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
```
- [M13] ## 7. Load Pretrained CNN and Fine-Tune Pick a model (`resnet18` or `resnet50`), load pretrained weights, and replace the final layer with `num_classes` outputs.
- [C14]
```python
MODEL_NAME = "resnet18"  # change to "resnet50" if you want a larger model
LEARNING_RATE = 3e-4
EPOCHS = 4  # 3-5 is a good quick testing range
def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("MODEL_NAME must be 'resnet18' or 'resnet50'")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
model = build_model(MODEL_NAME, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
```
- [M15] ## 8. Training Loop (Print Train + Validation Metrics) Each epoch prints: - train loss, train accuracy - validation loss, validation accuracy We also keep the best model checkpoint in memory (by validation accuracy).
- [C16]
```python
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
```
- [M17] ## 9. Validation and Optional Test Evaluation Now we evaluate final model quality: - best validation accuracy - optional test accuracy (if we created a test split from validation)
- [C18]
```python
if best_state is not None:
    model.load_state_dict(best_state)
def evaluate_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples
val_acc_final = evaluate_accuracy(model, val_loader)
print(f"Best validation accuracy: {val_acc_final:.3f}")
if test_loader is not None:
```
- [M19] ## 10. Predict a Single Image This helper function runs inference on one image and returns the predicted style label and name.
- [C20]
```python
def predict_image_style(model, image_path: Path, transform, label_to_style_map):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred_label = int(outputs.argmax(dim=1).item())
    pred_style = label_to_style_map.get(pred_label, f"Unknown label {pred_label}")
    return pred_label, pred_style
if test_df is not None and len(test_df) > 0:
    sample_row = test_df.iloc[0]
else:
    sample_row = val_eval_df.iloc[0]
sample_relative_path = sample_row["relative_path"]
true_label = int(sample_row["label"])
sample_image_path = wikiart_dir / sample_relative_path
pred_label, pred_style = predict_image_style(model, sample_image_path, eval_transform, label_to_style)
true_style = label_to_style.get(true_label, f"Unknown label {true_label}")
```
- [M21] ## 11. Next Steps (Web App Integration) You can integrate this model into your web app like this: 1. Save the trained model weights 2. Create a backend API endpoint for image upload 3. Preprocess uploaded image with the same `eval_transform` 4. Run model infer...

## wikiart_style_classification_test2.ipynb

- [M0] # WikiArt Style Classification with PyTorch In this beginner-friendly tutorial, we will build an AI model that predicts the **artistic style** of a painting (for example Impressionism, Cubism, Realism) using the WikiArt dataset. We will use a **pretrained CNN*...
- [M1] ## 1. Project Overview Our goal is simple: - Read the WikiArt training and validation CSV files - Load images and labels with a custom PyTorch `Dataset` - Train a pretrained CNN to predict style labels - Evaluate accuracy on validation data - Make a prediction...
- [M2] ## 2. Import Libraries First, we import the tools we need for data loading, image preprocessing, model training, and evaluation.
- [C3]
```python
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
# Avoid noisy warnings for very large source images; we resize them immediately
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
# Make results more reproducible for this tutorial
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```
- [C7]
```python
def find_split_csv(data_dir: Path, split: str) -> Path:
    """Find a CSV for a split (train/val) with preference for style-related filenames."""
    split = split.lower()
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    preferred = [p for p in csvs if split in p.stem.lower() and "style" in p.stem.lower()]
    if preferred:
        return preferred[0]
    fallback = [p for p in csvs if split in p.stem.lower()]
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Could not find a '{split}' CSV in {data_dir}")
train_csv = find_split_csv(wikiart_dir, "train")
val_csv = find_split_csv(wikiart_dir, "val")
print(f"Train CSV: {train_csv.name}")
print(f"Val CSV:   {val_csv.name}")
train_df = pd.read_csv(train_csv, header=None, names=["relative_path", "label"])
```
- [M8] ## 4. Prepare Label Mapping The label is numeric, but each image path includes the style folder name. We will build a mapping from label ID to style name so our predictions are easier to understand.
- [C9]
```python
def extract_style_name(relative_path: str) -> str:
    return Path(relative_path).parts[0]
train_df["style_name"] = train_df["relative_path"].map(extract_style_name)
val_df["style_name"] = val_df["relative_path"].map(extract_style_name)
label_to_style = (
    train_df[["label", "style_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["style_name"]
    .to_dict()
)
num_classes = len(label_to_style)
print(f"Number of classes: {num_classes}")
print("First 10 label -> style mappings:")
for label, style in list(label_to_style.items())[:10]:
    print(f"{label:2d} -> {style}")
```
- [M10] ## 5. Prepare Image Transformations Neural networks need consistent input size and scale. We will: - resize each image - convert image to tensor - normalize with ImageNet mean/std (good practice for pretrained models)
- [C11]
```python
image_size = 224
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
- [C13]
```python
class WikiArtStyleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, transform=None, max_retries: int = 10):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.max_retries = max_retries
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        last_error = None
        # Retry with a different sample if a file is missing/corrupt.
        for _ in range(self.max_retries):
            row = self.df.iloc[idx]
            img_path = self.image_root / row["relative_path"]
            label = int(row["label"])
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
```
- [M14] ## 7. Create DataLoaders We will build DataLoaders for training and validation. Because this dataset has no test split, we can optionally split a small part of validation into a simulated test set.
- [C15]
```python
batch_size = 32
num_workers = 0  # Keep 0 for beginner-friendly compatibility on Windows
def filter_existing_rows(df: pd.DataFrame, image_root: Path, split_name: str) -> pd.DataFrame:
    full_paths = df["relative_path"].map(lambda p: image_root / p)
    exists_mask = full_paths.map(lambda p: p.exists())
    removed = int((~exists_mask).sum())
    kept = int(exists_mask.sum())
    print(f"{split_name}: kept {kept:,} rows, removed {removed:,} missing files")
    cleaned = df.loc[exists_mask].reset_index(drop=True)
    if cleaned.empty:
        raise RuntimeError(f"No valid images left in {split_name} after filtering missing files.")
    return cleaned
train_df_clean = filter_existing_rows(train_df, wikiart_dir, "train")
val_df_clean = filter_existing_rows(val_df, wikiart_dir, "validation")
train_dataset = WikiArtStyleDataset(train_df_clean, wikiart_dir, transform=train_transform)
val_dataset_full = WikiArtStyleDataset(val_df_clean, wikiart_dir, transform=val_transform)
CREATE_TEST_SPLIT = True
TEST_FRACTION_FROM_VAL = 0.2
```
- [M16] ## 8. Build the Model We load a pretrained **ResNet-18** model and replace its final classification layer so it predicts the number of WikiArt styles.
- [C17]
```python
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(model.fc)
```
- [M18] ## 9. Train the Model This training loop runs for a few epochs, updates model weights on training data, and checks validation performance after each epoch.
- [C19]
```python
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
```
- [M20] ## 10. Evaluate the Model Now we calculate final accuracy on validation data. If we created a simulated test split from validation, we also report that.
- [C21]
```python
def evaluate_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples
val_accuracy = evaluate_accuracy(model, val_loader)
print(f"Validation accuracy: {val_accuracy:.3f}")
if test_loader is not None:
    test_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Simulated test accuracy: {test_accuracy:.3f}")
```
- [M22] ## 11. Make a Prediction for One Image Finally, we load a single image, run inference, and convert the predicted numeric label back to a style name.
- [C23]
```python
def predict_image_style(model, image_path: Path, transform, label_to_style_map):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred_label = int(outputs.argmax(dim=1).item())
    pred_style = label_to_style_map.get(pred_label, f"Unknown label {pred_label}")
    return pred_label, pred_style
sample_relative_path = val_df.iloc[0]["relative_path"]
sample_image_path = wikiart_dir / sample_relative_path
pred_label, pred_style = predict_image_style(model, sample_image_path, val_transform, label_to_style)
print(f"Image: {sample_relative_path}")
print(f"Predicted label: {pred_label}")
print(f"Predicted style: {pred_style}")
```
- [M24] ## 12. Next Steps Great work. You now have a complete beginner pipeline for art style classification. Possible improvements: - Train for more epochs and tune learning rate - Add stronger data augmentation - Save and load model checkpoints - Track metrics like ...

## wikiart_style_classification_test3.ipynb

- [M0] # WikiArt Style Classification (ResNet50 - Third Notebook) This is a fresh third notebook focused on improving performance with stronger training settings. Why this new notebook? - Earlier notebooks reached around 50-60% validation accuracy - We want better le...
- [M1] ## 1. Project Overview In this notebook we will: - Automatically locate WikiArt CSV files and image folders - Build a custom PyTorch Dataset - Filter bad rows (missing/unreadable images) - Train a pretrained ResNet50 model - Track train and validation loss/acc...
- [C3]
```python
from pathlib import Path
import random
import warnings
import copy
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- [M4] ## 3. Dataset Explanation and Preview WikiArt style data is stored as: - image files inside style folders - CSV rows in the format `relative_path,label` (no header) The code below auto-detects the project root and finds train/validation CSVs.
- [C5]
```python
def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "datasets").exists() and (candidate / "README.md").exists():
            return candidate
    return start.resolve()
def find_split_csv(data_dir: Path, split: str) -> Path:
    split = split.lower()
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    preferred = [p for p in csvs if split in p.stem.lower() and "style" in p.stem.lower()]
    if preferred:
        return preferred[0]
    fallback = [p for p in csvs if split in p.stem.lower()]
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Could not find a '{split}' CSV in {data_dir}")
```
- [C6]
```python
def extract_style_name(relative_path: str) -> str:
    return Path(relative_path).parts[0]
train_df["style_name"] = train_df["relative_path"].map(extract_style_name)
val_df["style_name"] = val_df["relative_path"].map(extract_style_name)
label_to_style = (
    train_df[["label", "style_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["style_name"]
    .to_dict()
)
num_classes = len(label_to_style)
print(f"Number of classes: {num_classes}")
print("First 10 mappings:")
for lbl, style in list(label_to_style.items())[:10]:
    print(f"{lbl:2d} -> {style}")
```
- [M7] ## 4. Image Transformations with Augmentation Training augmentations: - random horizontal flip - random rotation (+/-10 degrees) - color jitter (brightness/contrast/saturation) - random resized crop (slight zoom/crop) Validation/test transforms are simpler and...
- [C8]
```python
image_size = 224
batch_size = 32  # choose 32-64 depending on GPU memory
num_workers = 0  # beginner-friendly default on Windows
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
```
- [M11] ## 6. DataLoaders (Train, Validation, Optional Test) Because WikiArt here has train + validation CSV only, we optionally split 15% of validation as test. We also filter out rows with missing or unreadable image files before training.
- [C12]
```python
def is_readable_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return False
def filter_valid_rows(df: pd.DataFrame, image_root: Path, split_name: str) -> pd.DataFrame:
    full_paths = df["relative_path"].map(lambda p: image_root / p)
    valid_mask = full_paths.map(is_readable_image)
    kept = int(valid_mask.sum())
    removed = int((~valid_mask).sum())
    print(f"{split_name}: kept {kept:,}, removed {removed:,} bad rows")
    cleaned = df.loc[valid_mask].reset_index(drop=True)
    if cleaned.empty:
        raise RuntimeError(f"No valid rows left for split '{split_name}'.")
```
- [M13] ## 7. Load Pretrained ResNet50 and Fine-Tune We load ImageNet-pretrained ResNet50 and replace the final fully connected layer with the number of WikiArt classes.
- [C14]
```python
LEARNING_RATE = 3e-4
EPOCHS = 10  # requested 8-12 range
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model.fc)
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {batch_size}")
```
- [M15] ## 8. Training Loop with Progress Printing Each epoch reports: - train loss and train accuracy - validation loss and validation accuracy We save the best model state based on validation accuracy.
- [C16]
```python
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
```
- [M17] ## 9. Validation and Test Evaluation We evaluate final accuracy on validation and (if available) on test split from validation.
- [C18]
```python
if best_state is not None:
    model.load_state_dict(best_state)
def evaluate_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples
final_val_acc = evaluate_accuracy(model, val_loader)
print(f"Final validation accuracy: {final_val_acc:.3f}")
if test_loader is not None:
```
- [M19] ## 10. Multi-Image Prediction Example This example predicts style for **5 images** and prints true vs predicted labels.
- [C20]
```python
def predict_image_style(model, image_path: Path, transform, label_to_style_map):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred_label = int(outputs.argmax(dim=1).item())
    pred_style = label_to_style_map.get(pred_label, f"Unknown label {pred_label}")
    return pred_label, pred_style
# Pick 5 examples from test split if available, otherwise from validation
if test_df is not None and len(test_df) > 0:
    sample_df = test_df.sample(n=min(5, len(test_df)), random_state=SEED).reset_index(drop=True)
else:
    sample_df = val_eval_df.sample(n=min(5, len(val_eval_df)), random_state=SEED).reset_index(drop=True)
print("Prediction examples (5 images):")
for i, row in sample_df.iterrows():
    sample_relative_path = row["relative_path"]
    true_label = int(row["label"])
```
- [M21] ## 11. Next Steps (Web App Integration) To use this in your app later: 1. Save best model weights (`torch.save`) 2. Create backend endpoint for image upload 3. Preprocess uploaded image with the same eval transform 4. Run inference and return predicted style 5...

## wikiart_style_classification_test4.ipynb

- [M0] # WikiArt Style Classification - Test 4 (Improved Pipeline) This notebook is a stronger experimental version of `test3`. Main upgrades in this version: - two-stage training (freeze backbone -> fine-tune all layers) - stronger backbone (`ConvNeXt-Tiny`) - class...
- [C2]
```python
from pathlib import Path
import random
import warnings
import copy
import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```
- [C4]
```python
def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "datasets").exists() and (candidate / "README.md").exists():
            return candidate
    return start.resolve()
def find_split_csv(data_dir: Path, split: str) -> Path:
    split = split.lower()
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    preferred = [p for p in csvs if split in p.stem.lower() and "style" in p.stem.lower()]
    if preferred:
        return preferred[0]
    fallback = [p for p in csvs if split in p.stem.lower()]
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Could not find a '{split}' CSV in {data_dir}")
```
- [C5]
```python
def extract_style_name(relative_path: str) -> str:
    return Path(relative_path).parts[0]
train_df["style_name"] = train_df["relative_path"].map(extract_style_name)
val_df["style_name"] = val_df["relative_path"].map(extract_style_name)
label_to_style = (
    train_df[["label", "style_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["style_name"]
    .to_dict()
)
num_classes = len(label_to_style)
print(f"Number of classes: {num_classes}")
print("First 10 mappings:")
for lbl, style in list(label_to_style.items())[:10]:
    print(f"{lbl:2d} -> {style}")
```
- [M6] ## 3. Data Preprocessing and Augmentation Training augmentations are stronger in this notebook: - RandomResizedCrop (slight zoom/crop) - RandomHorizontalFlip - RandomRotation (+/-10) - RandomAffine and RandomPerspective - stronger ColorJitter - RandomErasing V...
- [C7]
```python
image_size = 256
batch_size = 16
# NOTE: In Windows notebooks, worker processes can add major overhead or appear to stall.
# Keep this conservative for stable throughput.
if device.type == "cuda":
    if os.name == "nt":
        num_workers = 0
    else:
        num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
else:
    num_workers = 0
pin_memory = device.type == "cuda"
persistent_workers = num_workers > 0
use_amp = device.type == "cuda"
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
```
- [M8] ## 4. Dataset and DataLoader Creation We reuse the custom dataset approach and filter out missing/unreadable images before training.
- [C10]
```python
import time
def filter_valid_rows(df: pd.DataFrame, image_root: Path, split_name: str, strict_verify: bool = False) -> pd.DataFrame:
    start = time.time()
    full_paths = df["relative_path"].map(lambda p: image_root / p)
    # Fast path: check path existence only (much faster for large datasets).
    exists_mask = full_paths.map(lambda p: p.exists())
    # Optional strict mode: verify only existing images (slower but more thorough).
    if strict_verify:
        valid_mask = exists_mask.copy()
        existing_indices = np.where(exists_mask.to_numpy())[0]
        total_existing = len(existing_indices)
        print(f"{split_name}: strict verification enabled for {total_existing:,} files...")
        for i, idx in enumerate(existing_indices, start=1):
            p = full_paths.iloc[idx]
            try:
                with Image.open(p) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError):
```
- [M11] ## 5. Model Setup (Stronger Backbone) Instead of ResNet50, we test a stronger architecture: **ConvNeXt-Tiny** with pretrained weights. Its classifier layer is replaced to predict the number of WikiArt styles.
- [C12]
```python
# Accuracy-first profile for final experiments.
# Set FAST_ITERATION_MODE=True only when you want quick sanity checks.
FAST_ITERATION_MODE = False
if FAST_ITERATION_MODE:
    MODEL_NAME = "resnet50"
    TOTAL_EPOCHS = 8
    HEAD_EPOCHS = 2
    MAX_TRAIN_BATCHES = 40
    MAX_VAL_BATCHES = 20
    HEAD_LR = 1e-3
    FT_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    MAX_GRAD_NORM = 1.0
else:
    MODEL_NAME = "convnext_tiny"
    TOTAL_EPOCHS = 36
    HEAD_EPOCHS = 5
```
- [M13] ## 6. Training Improvements This notebook applies several training improvements: - **Two-stage training**: first train classifier head, then fine-tune full model - **Class-weighted loss**: helps with class imbalance - **Learning-rate scheduler** (`ReduceLROnPl...
- [C14]
```python
def get_classifier_parameters(model, model_name: str):
    if model_name == "convnext_tiny":
        return model.classifier.parameters()
    if model_name == "resnet50":
        return model.fc.parameters()
    raise ValueError("Unsupported model name")
def freeze_backbone(model, model_name: str):
    for p in model.parameters():
        p.requires_grad = False
    for p in get_classifier_parameters(model, model_name):
        p.requires_grad = True
def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
def rand_bbox(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
```
- [M15] ## 7. Training Loop We track per-epoch metrics: - train/validation loss - train/validation Top-1 accuracy - train/validation Top-5 accuracy We also save the best checkpoint and restore it at the end.
- [C16]
```python
def topk_accuracies(logits, targets, topk=(1, 5)):
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        kk = min(k, logits.size(1))
        correct_k = correct[:kk].reshape(-1).float().sum(0)
        results.append(correct_k.item() / targets.size(0))
    return results
def run_one_epoch(model, loader, criterion, optimizer=None, log_prefix="", print_every=20, max_batches=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
```
- [C17]
```python
# Optional training curves (only if matplotlib is available)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib is not installed in this environment. Skipping plots.")
if plt is not None and len(history_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_top1"], label="train_top1")
    axes[1].plot(history_df["epoch"], history_df["val_top1"], label="val_top1")
    axes[1].plot(history_df["epoch"], history_df["val_top5"], label="val_top5")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
```
- [M18] ## 8. Final Evaluation We report Top-1 and Top-5 on validation and on test split (if created).
- [C19]
```python
def evaluate_model(model, loader, criterion):
    loss, top1, top5 = run_one_epoch(model, loader, criterion, optimizer=None)
    return {"loss": loss, "top1": top1, "top5": top5}
val_metrics = evaluate_model(model, val_loader, criterion)
print(
    f"Validation -> loss: {val_metrics['loss']:.4f}, "
    f"top1: {val_metrics['top1']:.3f}, top5: {val_metrics['top5']:.3f}"
)
if test_loader is not None:
    test_metrics = evaluate_model(model, test_loader, criterion)
    print(
        f"Test       -> loss: {test_metrics['loss']:.4f}, "
        f"top1: {test_metrics['top1']:.3f}, top5: {test_metrics['top5']:.3f}"
    )
```
- [M20] ## 9. Example Predictions This reuses prediction logic from the reference notebook and prints several examples.
- [C21]
```python
def predict_image_style(model, image_path: Path, transform, label_to_style_map, topk=5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_labels = probs.topk(k=min(topk, probs.shape[1]), dim=1)
    top_labels = top_labels[0].tolist()
    top_probs = top_probs[0].tolist()
    top_styles = [label_to_style_map.get(lbl, f"Unknown {lbl}") for lbl in top_labels]
    return top_labels[0], top_styles[0], list(zip(top_styles, top_probs))
source_df = test_df if test_df is not None and len(test_df) > 0 else val_eval_df
sample_df = source_df.sample(n=min(5, len(source_df)), random_state=SEED).reset_index(drop=True)
for i, row in sample_df.iterrows():
    sample_relative_path = row["relative_path"]
    true_label = int(row["label"])
    true_style = label_to_style.get(true_label, f"Unknown {true_label}")
```
- [M22] ## 10. Next Steps To continue improving: 1. Try larger input size (for example 256) if GPU memory allows 2. Tune augmentation strength and batch size 3. Add label smoothing or mixup/cutmix 4. Export best model and integrate with your web app backend This noteb...
- [C23]
```python
# Quick summary of final training results
print("History rows:", len(history_df))
if len(history_df) > 0:
    print("\nBest validation Top-1 row:")
    best_idx = history_df["val_top1"].idxmax()
    display(history_df.loc[[best_idx]])
    print("\nLast 5 epochs:")
    display(history_df.tail(5))
print("\nFinal evaluation metrics:")
print("Validation:", val_metrics)
if "test_metrics" in globals():
    print("Test:", test_metrics)
print("\nBest checkpoint path:", checkpoint_path)
print("Best epoch:", best_epoch)
print("Best val_top1:", best_val_top1)
```
- [M24] ## 11. Save Results Registry (Tests 1-4) This cell exports a consolidated metrics table for tests 1-4 so experiment results are persisted outside notebook outputs.
- [C25]
```python
import json
import re
from datetime import datetime, timezone
def _collect_output_texts(cell):
    texts = []
    for out in cell.get("outputs", []):
        out_type = out.get("output_type")
        if out_type == "stream":
            t = out.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            texts.append(str(t))
        elif out_type in {"execute_result", "display_data"}:
            data = out.get("data", {})
            for key in ["text/plain", "text/markdown"]:
                if key in data:
                    t = data[key]
                    if isinstance(t, list):
```

## wikiart_style_classification_test5.ipynb

- [M0] # Recovered Results (Test5) This notebook file is structurally damaged, but your **saved test5 results are still on disk**. Run the cell below this note to load and display: - `models/results/wikiart_tests_1_to_5_summary.csv` - `models/results/wikiart_test5_hi...
- [C1]
```python
from pathlib import Path
import pandas as pd
project_root = Path.cwd()
summary_path = project_root / "models" / "results" / "wikiart_tests_1_to_5_summary.csv"
history_path = project_root / "models" / "results" / "wikiart_test5_history.csv"
ckpt_path = project_root / "models" / "wikiart_test5_best.pt"
if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
    row = summary_df.loc[summary_df["experiment"].astype(str).str.lower() == "test5"]
    if not row.empty:
        cols = [c for c in ["val_top1", "test_top1", "best_epoch", "model_name", "saved_at_utc"] if c in row.columns]
        print("Recovered test5 summary:")
        display(row[cols])
    else:
        print("No test5 row found in summary CSV.")
else:
    print(f"Missing summary file: {summary_path}")
if history_path.exists():
```
- [M2] # WikiArt Style Classification - Test 4 (Improved Pipeline) This notebook is a stronger experimental version of `test3`. Main upgrades in this version: - two-stage training (freeze backbone -> fine-tune all layers) - stronger backbone (`ConvNeXt-Tiny`) - class...
- [C5]
```python
def extract_style_name(relative_path: str) -> str:
    return Path(relative_path).parts[0]
train_df["style_name"] = train_df["relative_path"].map(extract_style_name)
val_df["style_name"] = val_df["relative_path"].map(extract_style_name)
label_to_style = (
    train_df[["label", "style_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["style_name"]
    .to_dict()
)
num_classes = len(label_to_style)
print(f"Number of classes: {num_classes}")
print("First 10 mappings:")
for lbl, style in list(label_to_style.items())[:10]:
    print(f"{lbl:2d} -> {style}")
```
- [C6]
```python
image_size = 224
batch_size = 24
# NOTE: In Windows notebooks, worker processes can add major overhead or appear to stall.
# Keep this conservative for stable throughput.
if device.type == "cuda":
    if os.name == "nt":
        num_workers = 0
    else:
        num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
else:
    num_workers = 0
pin_memory = device.type == "cuda"
persistent_workers = num_workers > 0
use_amp = device.type == "cuda"
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
```
- [M8] ## 5. Model Setup (Fast + Strong) We keep a strong pretrained backbone but tune hyperparameters for better **accuracy-per-time**. Key differences from test4: - fewer epochs + earlier unfreeze - cosine LR decay instead of plateau-only reduction - exponential mo...
- [M9] ## 6. Training Improvements This notebook uses: - two-stage fine-tuning (head -> full model) - class-weighted label-smoothed cross-entropy - MixUp/CutMix regularization - cosine LR schedule - EMA model averaging - early stopping on validation Top-1
- [M10] ## 7. Training Loop We track per-epoch metrics: - train/validation loss - train/validation Top-1 accuracy - train/validation Top-5 accuracy We also save the best checkpoint and restore it at the end.
- [C11]
```python
# Optional training curves (only if matplotlib is available)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib is not installed in this environment. Skipping plots.")
if plt is not None and len(history_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_top1"], label="train_top1")
    axes[1].plot(history_df["epoch"], history_df["val_top1"], label="val_top1")
    axes[1].plot(history_df["epoch"], history_df["val_top5"], label="val_top5")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
```
- [C12]
```python
def evaluate_model(model, loader, criterion):
    loss, top1, top5 = run_one_epoch(model, loader, criterion, optimizer=None)
    return {"loss": loss, "top1": top1, "top5": top5}
val_metrics = evaluate_model(model, val_loader, criterion)
print(
    f"Validation -> loss: {val_metrics['loss']:.4f}, "
    f"top1: {val_metrics['top1']:.3f}, top5: {val_metrics['top5']:.3f}"
)
if test_loader is not None:
    test_metrics = evaluate_model(model, test_loader, criterion)
    print(
        f"Test       -> loss: {test_metrics['loss']:.4f}, "
        f"top1: {test_metrics['top1']:.3f}, top5: {test_metrics['top5']:.3f}"
    )
```
- [C13]
```python
def predict_image_style(model, image_path: Path, transform, label_to_style_map, topk=5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_labels = probs.topk(k=min(topk, probs.shape[1]), dim=1)
    top_labels = top_labels[0].tolist()
    top_probs = top_probs[0].tolist()
    top_styles = [label_to_style_map.get(lbl, f"Unknown {lbl}") for lbl in top_labels]
    return top_labels[0], top_styles[0], list(zip(top_styles, top_probs))
source_df = test_df if test_df is not None and len(test_df) > 0 else val_eval_df
sample_df = source_df.sample(n=min(5, len(source_df)), random_state=SEED).reset_index(drop=True)
for i, row in sample_df.iterrows():
    sample_relative_path = row["relative_path"]
    true_label = int(row["label"])
    true_style = label_to_style.get(true_label, f"Unknown {true_label}")
```
- [C14]
```python
# Quick summary of final training results
print("History rows:", len(history_df))
if len(history_df) > 0:
    print("\nBest validation Top-1 row:")
    best_idx = history_df["val_top1"].idxmax()
    display(history_df.loc[[best_idx]])
    print("\nLast 5 epochs:")
    display(history_df.tail(5))
print("\nFinal evaluation metrics:")
print("Validation:", val_metrics)
if "test_metrics" in globals():
    print("Test:", test_metrics)
print("\nBest checkpoint path:", checkpoint_path)
print("Best epoch:", best_epoch)
print("Best val_top1:", best_val_top1)
```
- [C15]
```python
import json
import re
from datetime import datetime, timezone
def _collect_output_texts(cell):
    texts = []
    for out in cell.get("outputs", []):
        out_type = out.get("output_type")
        if out_type == "stream":
            t = out.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            texts.append(str(t))
        elif out_type in {"execute_result", "display_data"}:
            data = out.get("data", {})
            for key in ["text/plain", "text/markdown"]:
                if key in data:
                    t = data[key]
                    if isinstance(t, list):
```

## wikiart_style_classification_test6_max_accuracy.ipynb

- [M0] # WikiArt Style Classification — Test 6 (Highest Accuracy Notebook) This notebook is made to push accuracy as high as possible, based on what worked best in your tests 1 to 5. ### What is improved here (easy words) - Strong model: **ViT-Base 384** pretrained w...
- [M3] ## 2) Full High-Accuracy Pipeline Code This cell contains the full training pipeline (same strategy as the best script, but now directly in notebook form).
- [C4]
```python
@dataclass
class TrainConfig:
    model_name: str = "vit_base_patch16_384.augreg_in21k_ft_in1k"
    image_size: int = 384
    batch_size: int = 12
    effective_batch_size: int = 48
    head_epochs: int = 2
    ft_epochs: int = 24
    patience: int = 8
    warmup_epochs: int = 2
    head_lr: float = 6e-4
    ft_backbone_lr: float = 1e-5
    ft_head_lr: float = 3e-5
    weight_decay: float = 8e-5
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 0.8
    mix_probability: float = 0.75
    label_smoothing: float = 0.12
```
- [C5]
```python
# Progress-print version of run_epoch (overrides previous function)
# It prints status every N batches, so you can see training is moving.
def run_epoch(model, loader, criterion, optimizer, device, scaler, num_classes, cfg, ema, is_train, accum_steps):
    model.train(is_train)
    total_loss = total_top1 = total_top5 = 0.0
    total_samples = 0
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    progress_every = max(1, int(getattr(cfg, "progress_every", 100)))
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if is_train:
                images_mixed, soft_targets = mixup_cutmix(
                    images, targets, cfg.mixup_alpha, cfg.cutmix_alpha, cfg.mix_probability, num_classes
                )
```
- [C6]
```python
# AMP compatibility patch: removes deprecation warnings from older torch.cuda.amp calls
# Run this once before the training cell.
if hasattr(torch, "amp") and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
    def _amp_grad_scaler_compat(*args, **kwargs):
        kwargs.setdefault("device", "cuda")
        return torch.amp.GradScaler(*args, **kwargs)
    def _amp_autocast_compat(*args, **kwargs):
        kwargs.setdefault("device_type", "cuda")
        return torch.amp.autocast(*args, **kwargs)
    torch.cuda.amp.GradScaler = _amp_grad_scaler_compat
    torch.cuda.amp.autocast = _amp_autocast_compat
    print("AMP compatibility patch active (torch.cuda.amp -> torch.amp).")
else:
    print("torch.amp not available in this environment; no patch applied.")
```
- [M7] ## 3) Run Training (Best Settings) Run this cell to start the **real high-accuracy training**. Tip: - Keep `head_epochs=2` and `ft_epochs=24` first. - If you still have GPU time after that, increase `ft_epochs` to 28 or 32 for a final push.
- [C8]
```python
project_root = Path.cwd()
cfg = TrainConfig(
    model_name="vit_base_patch16_384.augreg_in21k_ft_in1k",
    image_size=384,
    batch_size=12,
    effective_batch_size=48,
    head_epochs=2,
    ft_epochs=24,
    patience=8,
    warmup_epochs=2,
    head_lr=6e-4,
    ft_backbone_lr=1e-5,
    ft_head_lr=3e-5,
    weight_decay=8e-5,
    mixup_alpha=0.4,
    cutmix_alpha=0.8,
    mix_probability=0.75,
    label_smoothing=0.12,
```
- [M9] ## 4) Quick Result Check (Simple View) This gives a clean summary after training.
- [C10]
```python
print("History rows:", len(history_df))
if len(history_df) > 0:
    best_idx = history_df["val_top1"].idxmax()
    print("\nBest validation row:")
    display(history_df.loc[[best_idx]])
    print("\nLast 5 epochs:")
    display(history_df.tail(5))
print("\nFinal Validation:", final_val)
print("Final Test:", final_test)
results_dir = Path.cwd() / "models" / "results"
print("\nSaved files:")
print("-", results_dir / "wikiart_test6_history.csv")
print("-", results_dir / "wikiart_tests_1_to_6_summary.csv")
print("-", Path.cwd() / "models" / "wikiart_test6_best.pt")
```