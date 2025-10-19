# train_surgwound_pytorch.py
import os
import json
import base64
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging
import time

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# SETUP LOGGING
# -------------------------
def setup_logging():
    """Setup comprehensive logging for troubleshooting and tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# -------------------------
# SETTINGS - tweak as needed
# -------------------------
FILE_PATH = "merged_oversampled_dataset.csv"
BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MOBILENETV3_LARGE = False  # True = large, False = small (faster)
CHECKPOINT_PATH = "best_surgwound_mobilenetv3.pt"
TEST_SPLIT = 0.10
VAL_SPLIT = 0.15  # portion of remaining after test
SEED = 42

logger.info(f"Initializing training with settings:")
logger.info(f"  JSON_PATH: {FILE_PATH}")
logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"  IMAGE_SIZE: {IMAGE_SIZE}")
logger.info(f"  EPOCHS: {EPOCHS}")
logger.info(f"  LR: {LR}")
logger.info(f"  DEVICE: {DEVICE}")
logger.info(f"  USE_MOBILENETV3_LARGE: {USE_MOBILENETV3_LARGE}")
logger.info(f"  TEST_SPLIT: {TEST_SPLIT}, VAL_SPLIT: {VAL_SPLIT}")
logger.info(f"  SEED: {SEED}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# LABEL SPACES (from you)
# -------------------------
LABEL_SPACE = {
    "Healing Status": ["Healed", "Not Healed"],
    "Exudate Type": ["Non-existent", "Serous", "Sanguineous", "Purulent", "Seropurulent", "Uncertain"],
    "Erythema": ["Non-existent", "Existent", "Uncertain"],
    "Edema": ["Non-existent", "Existent", "Uncertain"],
    "Infection Risk Assessment": ["Low", "Medium", "High"],
    "Urgency Level": ["Home Care (Green): Manage with routine care",
                      "Clinic Visit (Yellow): Requires professional evaluation within 48 hours",
                      "Emergency Care (Red): Seek immediate medical attention"]
}

# Map JSON 'field' to short keys
FIELD_KEY_MAP = {
    "Healing Status": "healing_status",
    "Exudate Type": "exudate_type",
    "Erythema": "erythema",
    "Edema": "edema",
    "Infection Risk Assessment": "infection_risk",
    "Urgency Level": "urgency"
}

logger.info("Building label mappings...")

# Build mapping dicts used during encoding/decoding
label2idx = {}
idx2label = {}
for long_field, options in LABEL_SPACE.items():
    key = FIELD_KEY_MAP[long_field]
    label2idx[key] = {opt: i for i, opt in enumerate(options)}
    idx2label[key] = {i: opt for i, opt in enumerate(options)}
    logger.info(f"  {key}: {len(options)} classes - {options}")


# -------------------------
# 1) Load CSV Data
# -------------------------
def load_csv_data(csv_path):
    logger.info(f"Loading CSV data from {csv_path}...")
    start_time = time.time()

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} rows from CSV in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise

    # Ensure expected columns exist
    required_cols = ["image"] + list(LABEL_SPACE.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns in CSV: {missing}")
    else:
        logger.info("All required columns present in CSV")

    # Check for missing values
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.any():
        logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
    else:
        logger.info("No missing values found in required columns")

    # Rename columns to match FIELD_KEY_MAP if needed
    rename_count = 0
    for long_field, short_field in FIELD_KEY_MAP.items():
        if long_field in df.columns:
            df.rename(columns={long_field: short_field}, inplace=True)
            rename_count += 1
    logger.info(f"Renamed {rename_count} columns to short field names")

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {list(df.columns)}")

    return df


df = load_csv_data(FILE_PATH)


# -------------------------
# 2) Encode labels to integer indices (fill missing with 'Uncertain' or index 0)
# -------------------------
def encode_label(key, text):
    mapping = label2idx[key]
    if text in mapping:
        return mapping[text]
    # try strip/normalize
    for k in mapping:
        if str(text).strip().lower() == str(k).strip().lower():
            return mapping[k]
    # fallback: try to find 'Uncertain' else 0
    if "Uncertain" in mapping:
        return mapping["Uncertain"]
    return 0


logger.info("Encoding labels to integer indices...")
start_time = time.time()

for key in label2idx:
    idx_col = f"{key}_idx"
    original_counts = df[key].value_counts()
    df[idx_col] = df[key].apply(lambda x: encode_label(key, x))
    encoded_counts = df[idx_col].value_counts().sort_index()

    logger.info(f"  {key} encoding distribution:")
    for idx_val, count in encoded_counts.items():
        label_name = idx2label[key].get(idx_val, "Unknown")
        logger.info(f"    {idx_val} ({label_name}): {count} samples")

    # Log any encoding issues
    unknown_mask = df[key].isnull() | (df[idx_col] == 0)
    unknown_count = unknown_mask.sum()
    if unknown_count > 0:
        logger.warning(f"    {unknown_count} samples encoded as fallback (0) for {key}")

logger.info(f"Label encoding completed in {time.time() - start_time:.2f}s")


# -------------------------
# 3) Custom PyTorch Dataset (decodes base64 on the fly)
# -------------------------
class SurgWoundDataset(Dataset):
    def __init__(self, df, image_size=IMAGE_SIZE, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.image_size = image_size
        self.keys = list(label2idx.keys())
        self.decode_errors = 0
        logger.info(f"Initialized SurgWoundDataset with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _decode_image(self, b64str):
        if not b64str or pd.isna(b64str):
            logger.debug("Empty or NaN image string, using fallback black image")
            return Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        try:
            img = Image.open(BytesIO(base64.b64decode(b64str))).convert("RGB")
            return img
        except Exception as e:
            self.decode_errors += 1
            # try fallback if b64 is actually a filename path
            try:
                img = Image.open(b64str).convert("RGB")
                logger.debug(f"Successfully loaded image from file path: {b64str}")
                return img
            except Exception as e2:
                logger.warning(f"Image decode error (fallback also failed): {e}, {e2}")
                return Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._decode_image(row["image"])
        if self.transforms:
            img = self.transforms(img)

        # make sure all labels are integers
        labels = {}
        for k in self.keys:
            val = row.get(f"{k}_idx", 0)
            if isinstance(val, str):
                try:
                    val = int(val)  # convert string to int
                except ValueError:
                    logger.warning(f"Invalid label value '{val}' for {k} at index {idx}, using 0")
                    val = 0
            labels[k] = val
        return img, labels

    def get_decode_stats(self):
        return self.decode_errors


# -------------------------
# 4) Transforms
# -------------------------
logger.info("Setting up data transforms...")
train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

logger.info("Train transforms: Resize, RandomCrop, Flip, ColorJitter, Normalize")
logger.info("Val/Test transforms: Resize, CenterCrop, Normalize")

# -------------------------
# 5) Train/val/test split and DataLoaders
# -------------------------
logger.info("Performing train/val/test split...")
n = len(df)
n_test = int(TEST_SPLIT * n)
n_remain = n - n_test
n_val = int(VAL_SPLIT * n_remain)
n_train = n_remain - n_val

logger.info(f"Dataset sizes - Total: {n}, Train: {n_train}, Val: {n_val}, Test: {n_test}")

dataset_full = SurgWoundDataset(df, transforms=val_tfms)  # default transforms for splitting
# We'll split indices then create datasets with appropriate transforms
indices = list(range(n))
np.random.shuffle(indices)

test_idx = indices[:n_test]
val_idx = indices[n_test:n_test + n_val]
train_idx = indices[n_test + n_val:]

train_df = df.loc[train_idx].reset_index(drop=True)
val_df = df.loc[val_idx].reset_index(drop=True)
test_df = df.loc[test_idx].reset_index(drop=True)

logger.info("Creating datasets with appropriate transforms...")
train_ds = SurgWoundDataset(train_df, transforms=train_tfms)
val_ds = SurgWoundDataset(val_df, transforms=val_tfms)
test_ds = SurgWoundDataset(test_df, transforms=val_tfms)

logger.info("Creating data loaders...")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Log dataset statistics
logger.info(f"Final dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
logger.info(f"Train dataset decode errors: {train_ds.get_decode_stats()}")
logger.info(f"Val dataset decode errors: {val_ds.get_decode_stats()}")
logger.info(f"Test dataset decode errors: {test_ds.get_decode_stats()}")

# -------------------------
# 6) Multi-head MobileNetV3 model (PyTorch)
# -------------------------
logger.info("Initializing MultiHeadMobileNetV3 model...")
start_time = time.time()


class MultiHeadMobileNetV3(nn.Module):
    def __init__(self, use_large=False, pretrained=True):
        super().__init__()
        logger.info(f"Creating MobileNetV3 {'Large' if use_large else 'Small'} with pretrained={pretrained}")
        if use_large:
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        # Remove classifier head and use features + pooling
        self.features = self.backbone.features  # feature extractor
        # determine feature dim by running a dummy tensor through features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # get in_features from backbone classifier if available
        if hasattr(self.backbone, 'classifier'):
            # classifier typically: [nn.Linear(in_features, ...)]
            # for mobilenet_v3_small classifier[0] may be Linear with in_features
            try:
                in_features = self.backbone.classifier[0].in_features
            except Exception as e:
                logger.warning(f"Could not get in_features from classifier: {e}")
                # fallback estimate
                in_features = 576 if not use_large else 960
        else:
            in_features = 576 if not use_large else 960

        logger.info(f"Feature dimension: {in_features}")

        # small shared head
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Create classification heads (output logits per head)
        self.heads = nn.ModuleDict()
        for key in label2idx:
            n_classes = len(label2idx[key])
            self.heads[key] = nn.Linear(512, n_classes)
            logger.info(f"  Added head '{key}' with {n_classes} output classes")

    def forward(self, x):
        x = self.features(x)  # (B, C, H, W)
        x = self.pool(x).view(x.size(0), -1)  # (B, C)
        x = self.shared_fc(x)
        out = {}
        for k, head in self.heads.items():
            out[k] = head(x)
        return out


model = MultiHeadMobileNetV3(use_large=USE_MOBILENETV3_LARGE, pretrained=True).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Model initialized in {time.time() - start_time:.2f}s")
logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

# -------------------------
# 7) Loss, optimizer, helper functions
# -------------------------
logger.info("Setting up loss functions and optimizer...")
# CrossEntropyLoss expects class indices (not one-hot)
criterions = {k: nn.CrossEntropyLoss() for k in label2idx}  # can set weights per head if needed
optimizer = optim.Adam(model.parameters(), lr=LR)

logger.info(f"Using CrossEntropyLoss for {len(criterions)} heads")
logger.info(f"Using Adam optimizer with LR={LR}")


def compute_loss(outputs, targets):
    # outputs: dict head -> logits (B, C)
    # targets: dict head -> (B,) long
    total_loss = 0.0
    losses = {}
    for k in outputs:
        loss = criterions[k](outputs[k], targets[k])
        losses[k] = loss.item()
        total_loss += loss
    return total_loss, losses


# convert batch targets dict of ints into tensor longs for each head
def to_target_tensors(batch_targets, device=DEVICE):
    # batch_targets is dict of lists/ints per key
    out = {}
    for k in batch_targets:
        out[k] = torch.tensor(batch_targets[k], dtype=torch.long, device=device)
    return out


# helper to extract predictions and true labels lists for sklearn
def collect_preds_trues(all_preds, all_trues):
    # each is list per batch of arrays; convert to flatten lists
    preds = {}
    trues = {}
    for k in all_preds:
        preds[k] = np.concatenate(all_preds[k], axis=0)
        trues[k] = np.concatenate(all_trues[k], axis=0)
    return preds, trues


# -------------------------
# 8) Training & validation loops
# -------------------------
if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()  # Optional, but safe for Windows

    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    training_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'=' * 50}")

        # Training phase
        model.train()
        train_losses = []
        train_losses_heads = defaultdict(float)
        batch_times = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        batch_start = time.time()

        for batch_idx, (imgs, labels) in enumerate(pbar):
            batch_load_time = time.time() - batch_start

            imgs = imgs.to(DEVICE)




            # DEBUG: Check label structure
            logger.info(f"Labels type: {type(labels)}")
            logger.info(f"Labels keys: {list(labels.keys())}")

            # Print sample values
            logger.info("Sample label values:")
            for k in list(labels.keys())[:2]:  # Just first 2 to avoid spam
                tensor = labels[k]
                values = tensor.cpu().numpy()[:5]  # First 5 values
                logger.info(f"  {k}: {values}")

            # Create batch_targets by moving existing tensors to device
            batch_targets = {
                k: labels[k].to(DEVICE)
                for k in label2idx
            }
















            optimizer.zero_grad()
            forward_start = time.time()
            outputs = model(imgs)
            forward_time = time.time() - forward_start

            loss, losses_heads = compute_loss(outputs, batch_targets)

            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start

            train_losses.append(loss.item())
            for k, v in losses_heads.items():
                train_losses_heads[k] += v

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if batch_idx % 10 == 0:  # Log every 10 batches
                avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else batch_time
                logger.debug(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f}, "
                             f"batch_time={batch_time:.3f}s, load_time={batch_load_time:.3f}s, "
                             f"forward_time={forward_time:.3f}s, backward_time={backward_time:.3f}s")

            pbar.set_postfix({"loss": np.mean(train_losses)})
            batch_start = time.time()

        # Log training statistics
        epoch_train_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        avg_train_loss = np.mean(train_losses)

        logger.info(f"Epoch {epoch} Training completed in {epoch_train_time:.2f}s")
        logger.info(f"Average batch time: {avg_batch_time:.3f}s")
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        # Log per-head training losses
        for k in train_losses_heads:
            avg_head_loss = train_losses_heads[k] / len(train_loader)
            logger.info(f"  {k} training loss: {avg_head_loss:.4f}")

        # Validation phase
        model.eval()
        val_losses = []
        val_losses_heads = defaultdict(float)
        all_preds = {k: [] for k in label2idx}
        all_trues = {k: [] for k in label2idx}

        val_start_time = time.time()
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                imgs = imgs.to(DEVICE)
                batch_targets = {
                    k: labels[k].to(DEVICE)
                    for k in label2idx
                }
                outputs = model(imgs)
                loss, losses_heads = compute_loss(outputs, batch_targets)
                val_losses.append(loss.item())
                for k, v in losses_heads.items():
                    val_losses_heads[k] += v

                # collect preds
                for k in outputs:
                    preds = outputs[k].argmax(dim=1).cpu().numpy()
                    trues = batch_targets[k].cpu().numpy()
                    all_preds[k].append(preds)
                    all_trues[k].append(trues)

        val_time = time.time() - val_start_time
        mean_val_loss = np.mean(val_losses) if val_losses else 0.0

        logger.info(f"Epoch {epoch} Validation completed in {val_time:.2f}s")
        logger.info(f"Training loss: {avg_train_loss:.4f}, Validation loss: {mean_val_loss:.4f}")

        # Calculate and log metrics
        preds_flat, trues_flat = collect_preds_trues(all_preds, all_trues)
        for k in label2idx:
            if len(trues_flat[k]) == 0:
                logger.warning(f"No validation samples for head: {k}")
                continue

            accuracy = (preds_flat[k] == trues_flat[k]).mean()
            logger.info(f"--- Head: {k} - Accuracy: {accuracy:.4f} ---")

            # Detailed classification report
            try:
                report = classification_report(
                    trues_flat[k],
                    preds_flat[k],
                    target_names=[idx2label[k][i] for i in range(len(idx2label[k]))],
                    zero_division=0,
                    output_dict=True
                )

                # Log per-class metrics
                for class_name, metrics in report.items():
                    if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                        if class_name == 'accuracy':
                            logger.info(f"  Overall Accuracy: {metrics:.4f}")
                        else:
                            logger.info(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                                        f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
                    elif isinstance(metrics, dict):
                        logger.info(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
            except Exception as e:
                logger.error(f"Error generating classification report for {k}: {e}")

        # Save best model
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': mean_val_loss,
                'train_loss': avg_train_loss,
                'label_mappings': {'label2idx': label2idx, 'idx2label': idx2label}
            }, CHECKPOINT_PATH)
            logger.info(f"Saved best checkpoint to {CHECKPOINT_PATH} with val_loss={mean_val_loss:.4f}")
        else:
            logger.info(f"Validation loss {mean_val_loss:.4f} not improved from {best_val_loss:.4f}")

        epoch_total_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} total time: {epoch_total_time:.2f}s")

    total_training_time = time.time() - training_start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time / 60:.2f} minutes)")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"{'=' * 50}")

# -------------------------
# 9) Final evaluation on test set (load best checkpoint)
# -------------------------
logger.info("\nStarting final evaluation on test set...")
if os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt.get('epoch', 'unknown')
        best_val_loss = ckpt.get('val_loss', 'unknown')
        logger.info(f"Loaded best checkpoint from epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.info("Using current model weights for testing")
else:
    logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH}, using current model weights")

model.eval()
all_preds = {k: [] for k in label2idx}
all_trues = {k: [] for k in label2idx}

test_start_time = time.time()
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="Test")):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        for k in outputs:
            preds = outputs[k].argmax(dim=1).cpu().numpy()
            trues = labels[k].cpu().numpy()


            all_preds[k].append(preds)
            all_trues[k].append(trues)

test_time = time.time() - test_start_time
logger.info(f"Test inference completed in {test_time:.2f}s")

preds_flat, trues_flat = collect_preds_trues(all_preds, all_trues)

logger.info("\n" + "=" * 60)
logger.info("FINAL TEST RESULTS")
logger.info("=" * 60)

for k in label2idx:
    if len(trues_flat[k]) == 0:
        logger.warning(f"No test samples for head: {k}")
        continue

    accuracy = (preds_flat[k] == trues_flat[k]).mean()
    logger.info(f"\n=== Test Head: {k} (Accuracy: {accuracy:.4f}) ===")

    try:
        print(classification_report(trues_flat[k], preds_flat[k],
                                    target_names=[idx2label[k][i] for i in range(len(idx2label[k]))],
                                    zero_division=0))

        cm = confusion_matrix(trues_flat[k], preds_flat[k])
        logger.info(f"Confusion matrix shape: {cm.shape}")
        logger.info("Confusion matrix:\n" + str(cm))

    except Exception as e:
        logger.error(f"Error generating test report for {k}: {e}")

# -------------------------
# 10) Grad-CAM example (requires pytorch-grad-cam)
# -------------------------
# pip install pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2

    logger.info("Setting up Grad-CAM visualization...")

    # Pick last conv layer in mobilenet backbone
    target_module = None
    for m in model.features.modules():
        if isinstance(m, nn.Conv2d):
            target_module = m

    if target_module is None:
        logger.warning("Could not find Conv2d layer for Grad-CAM")
    else:
        logger.info(f"Using target module: {target_module}")


    def make_gradcam_on_sample(df_row_index=0, head_key='healing_status'):
        logger.info(f"Generating Grad-CAM for sample {df_row_index}, head {head_key}")
        try:
            # get row from test_df
            row = test_df.reset_index(drop=True).iloc[df_row_index]
            b64 = row["image"]
            pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            img_np = np.array(pil).astype(np.float32) / 255.0
            input_tensor = val_tfms(pil).unsqueeze(0).to(DEVICE)

            cam = GradCAM(model=model, target_layers=[target_module], use_cuda=(DEVICE.type == 'cuda'))
            outputs = model(input_tensor)
            pred_idx = int(outputs[head_key].argmax(dim=1).item())
            true_label = row.get(f"{head_key}_idx", "unknown")

            logger.info(f"Prediction: {pred_idx} ({idx2label[head_key].get(pred_idx, 'unknown')}), "
                        f"True: {true_label} ({idx2label[head_key].get(true_label, 'unknown')})")

            grayscale_cam = cam(input_tensor=input_tensor, targets=[None])[0]
            grayscale_cam_resized = cv2.resize(grayscale_cam, (IMAGE_SIZE, IMAGE_SIZE))
            visualization = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)

            logger.info("Grad-CAM visualization completed successfully")
            return visualization, pred_idx, true_label

        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")
            return None, None, None

    # Example call (uncomment to run):
    # logger.info("Running Grad-CAM example...")
    # make_gradcam_on_sample(df_row_index=0, head_key='healing_status')

except Exception as e:
    logger.warning(f"Grad-CAM setup failed; install 'pytorch-grad-cam' to enable. Error: {e}")

# -------------------------
# FINAL SUMMARY
# -------------------------
logger.info("\n" + "=" * 60)
logger.info("TRAINING COMPLETED SUCCESSFULLY")
logger.info("=" * 60)
logger.info(f"Total training time: {total_training_time:.2f}s")
logger.info(f"Best model saved to: {CHECKPOINT_PATH}")
logger.info(f"Final test evaluation completed")
logger.info("Check 'training.log' for detailed logs")

# -------------------------
# END
# -------------------------