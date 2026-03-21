
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
```
- [M9] ## 5. Custom Dataset Class This dataset reads image paths from CSV rows and returns `(image_tensor, label)`. It also handles missing/corrupt images by retrying another sample.
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
```
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
```
- [M4] ## 3. Locate the WikiArt Dataset Automatically To keep this notebook flexible, we will search for the dataset folder and CSV files inside the project directory instead of hardcoding exact paths.
- [C5]
```python
def find_project_root(start: Path) -> Path:
    """Walk upward until we find a folder that looks like the project root."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "datasets").exists() and (candidate / "README.md").exists():
            return candidate
    return start.resolve()
project_root = find_project_root(Path.cwd())
wikiart_dir = project_root / "datasets" / "Wikiart"
if not wikiart_dir.exists():
    raise FileNotFoundError(f"Could not find WikiArt directory at: {wikiart_dir}")
print(f"Project root: {project_root}")
print(f"WikiArt dir:  {wikiart_dir}")
```
- [M6] ### Understand the CSV format The WikiArt style CSV files use rows like: `style_folder/image_name.jpg,label_id` There is no header row, so we will load with custom column names.
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
- [M12] ## 6. Create a Custom Dataset Loader Now we define a custom PyTorch `Dataset`. Each item will return: - transformed image tensor - numeric label
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
```
- [M9] ## 5. Custom Dataset Class This dataset reads one row from the CSV and returns: - transformed image tensor - numeric class label
- [C10]
```python
class WikiArtStyleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["relative_path"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
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
```
- [M3] ## 2. Dataset Loading and Verification This section reuses the working auto-discovery approach from the previous notebook.
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
```
- [M8] ## 4. Dataset and DataLoader Creation We reuse the custom dataset approach and filter out missing/unreadable images before training.
- [C9]
```python
class WikiArtStyleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["relative_path"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
```
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
```
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
```
- [M18] ## 8. Final Evaluation We report Top-1 and Top-5 on validation and on test split (if created).
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
```
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
```
- [M2] # WikiArt Style Classification - Test 4 (Improved Pipeline) This notebook is a stronger experimental version of `test3`. Main upgrades in this version: - two-stage training (freeze backbone -> fine-tune all layers) - stronger backbone (`ConvNeXt-Tiny`) - class...
- [M4] ## 2. Dataset Loading and Verification This section reuses the working auto-discovery approach from the previous notebook.
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
```
- [C7]
```python
class WikiArtStyleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["relative_path"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
```
- [M8] ## 5. Model Setup (Fast + Strong) We keep a strong pretrained backbone but tune hyperparameters for better **accuracy-per-time**. Key differences from test4: - fewer epochs + earlier unfreeze - cosine LR decay instead of plateau-only reduction - exponential mo...
- [M9] ## 6. Training Improvements This notebook uses: - two-stage fine-tuning (head -> full model) - class-weighted label-smoothed cross-entropy - MixUp/CutMix regularization - cosine LR schedule - EMA model averaging - early stopping on validation Top-1
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
```

## datasets/Artbench/prepare_artbench_local.ipynb

- [M0] # Prepare ArtBench images locally (with updated CSV) This notebook will do 3 things: 1. Read your original `ArtBench-10 (1).csv` file. 2. Download each image into local folders by style label (like your WikiArt structure). 3. Create **new CSV files** with loca...
- [M1] ## Step 1 — Imports and paths This cell: - imports the tools we need, - finds the ArtBench CSV file, - sets output locations for images and new CSV files.
- [C2]
```python
from pathlib import Path
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import re
import time
def find_artbench_csv() -> Path:
    candidates = [
        Path('datasets/Artbench/ArtBench-10 (1).csv'),
        Path('ArtBench-10 (1).csv'),
        Path('../Artbench/ArtBench-10 (1).csv'),
    ]
    for candidate in candidates:
```
- [M3] ## Step 2 — Read the CSV We load the data and quickly check important columns (`name`, `url`, `label`).
- [C4]
```python
df = pd.read_csv(CSV_PATH)
required_columns = {'name', 'url', 'label'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f'Missing required columns: {missing}')
print(f'Rows: {len(df):,}')
print(f'Unique labels: {df["label"].nunique()}')
display(df.head(3))
display(df['label'].value_counts())
```
- [M5] ## Step 3 — Helper functions These functions: - create safe folder/file names, - download one image with retries, - return both success status and local path.
- [C6]
```python
def sanitize_folder_name(value: str) -> str:
    text = str(value).strip()
    text = text.replace(' ', '_')
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'unknown'
def pick_filename(row_name: str, url: str) -> str:
    row_name = str(row_name).strip()
    if row_name and Path(row_name).suffix:
        return Path(row_name).name
    url_path_name = Path(unquote(urlparse(str(url)).path)).name
    if url_path_name and Path(url_path_name).suffix:
        return url_path_name
    return 'image.jpg'
```
- [M7] ## Step 4 — Download all images into label folders This can take a while because there are many images. Good to know: - It is resumable: already downloaded files are skipped. - It prints progress while running.
- [C8]
```python
records = df.to_dict(orient='records')
total = len(records)
results = []
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_one, i, row): i for i, row in enumerate(records)}
    for done_count, future in enumerate(as_completed(futures), start=1):
        results.append(future.result())
        if done_count % 250 == 0 or done_count == total:
            print(f'Processed {done_count:,}/{total:,}')
result_df = pd.DataFrame(results).sort_values('index')
success_count = int(result_df['ok'].sum())
fail_count = int((~result_df['ok']).sum())
print(f'✅ Success: {success_count:,}')
```
- [M9] ## Step 5 — Save new CSV files (original stays unchanged) We create: - `ArtBench-10-local-paths.csv` (full table with updated `url` column), - `ArtBench-10-local-simple.csv` (WikiArt-like: `relative_path,label_id`), - `ArtBench-10-download-failures.csv` (only ...
- [C10]
```python
df_out = df.copy()
df_out['source_url'] = df_out['url']
df_out['url'] = result_df['local_path'].values
df_out['download_ok'] = result_df['ok'].values
df_out['download_error'] = result_df['error'].values
df_out.to_csv(OUTPUT_CSV, index=False)
failures = df_out[~df_out['download_ok']].copy()
if len(failures) > 0:
    failures.to_csv(FAILURES_CSV, index=False)
    print(f'Failure list saved to: {FAILURES_CSV}')
else:
    print('No failures found.')
label_order = sorted(df_out['label'].dropna().astype(str).unique())
label_to_id = {label: i for i, label in enumerate(label_order)}
```
- [M11] ## Done After running all cells: - images are in `datasets/Artbench/images/<label>/<filename>`, - your original CSV is untouched, - you have one full updated CSV + one WikiArt-like simple CSV.

## datasets/ArtEmis/preprocess_artemis_wikiart_paths.ipynb

- [M0] # Preprocess ArtEmis: Link to WikiArt Image Paths This notebook links `painting` and `anchor_painting` from `Contrastive.csv` to relative image paths inside the WikiArt dataset. What this notebook does: 1. Loads `datasets/ArtEmis/Contrastive.csv` 2. Scans all ...
- [C1]
```python
from pathlib import Path
import pandas as pd
def find_project_root(start: Path) -> Path:
    """Walk up from the current directory until the expected dataset layout is found."""
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        has_artemis = (candidate / "datasets" / "ArtEmis" / "Contrastive.csv").exists()
        has_wikiart = (candidate / "datasets" / "Wikiart").exists()
        if has_artemis and has_wikiart:
            return candidate
    raise FileNotFoundError(
        "Could not locate project root containing datasets/ArtEmis/Contrastive.csv and datasets/Wikiart"
    )
# Resolve project root robustly, even if the notebook is run from datasets/ArtEmis.
```
- [C2]
```python
df = pd.read_csv(contrastive_csv)
print("Loaded rows:", len(df))
print("Columns:", list(df.columns))
display(df.head(3))
```
- [C3]
```python
# File types to include when scanning WikiArt images.
valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
image_files = [
    p for p in wikiart_root.rglob("*")
    if p.is_file() and p.suffix.lower() in valid_ext
]
print(f"Found {len(image_files):,} image files inside WikiArt.")
# Map: filename stem -> list of relative paths (relative to datasets folder).
name_to_paths = {}
for img_path in image_files:
    stem = img_path.stem.strip().lower()
    rel_to_datasets = img_path.relative_to(project_root / "datasets")
    name_to_paths.setdefault(stem, []).append(rel_to_datasets.as_posix())
print(f"Unique filename stems mapped: {len(name_to_paths):,}")
```
- [C4]
```python
def resolve_image_path(name: str):
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    matches = name_to_paths.get(key, [])
    if not matches:
        return None
    # If duplicates exist, keep first path for deterministic output.
    return sorted(matches)[0]
df["painting_path"] = df["painting"].apply(resolve_image_path)
df["anchor_painting_path"] = df["anchor_painting"].apply(resolve_image_path)
missing_painting = df["painting_path"].isna().sum()
missing_anchor = df["anchor_painting_path"].isna().sum()
print("Missing painting_path values:", int(missing_painting))
```
- [C5]
```python
df.to_csv(output_csv, index=False)
print(f"Saved new file: {output_csv}")
print(f"Exported {len(df):,} rows after removing rows with missing painting or anchor paths.")
print("Done. Original columns were preserved, and new path columns were added.")
```