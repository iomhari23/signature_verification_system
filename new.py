import os
import math
import time
import random
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, List, Dict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from PIL.ImageOps import scale
from skimage import filters, transform
from skimage.io import imread
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class Config:
    SEED = 42
    CANVAS_SIZE: Tuple[int, int] = (952, 1360)
    MODEL_INPUT_SIZE: Tuple[int, int] = (150, 220)
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    MAX_GRAD_NORM = 1000
    PRINT_FREQ = 100
    EARLY_STOP_PATIENCE = 5
    FC_DIM = 512
    FEATURE_DIM = 6
    FEATURE_PROJ_DIM = 128
    SCHEDULER_T_MAX = 4
    SCHEDULER_ETA_MIN = 1e-5
    OUTPUT_DIR = "./"
    MODEL_SAVE_NAME = "siamese_best_loss.pt"

def set_seed(seed: int = Config.SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

def build_logger(log_path: str = Config.OUTPUT_DIR + "train.log") -> logging.Logger:
    logger = logging.getLogger("signature_matcher")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

LOGGER = build_logger()

def load_dataframes(
    train_csv: str,
    test_csv: str,
    train_img_root: str,
    test_img_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _load(csv_path: str, img_root: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        original_cols = df.columns.tolist()
        df = df.rename(columns={
            original_cols[0]: "real_rel",
            original_cols[1]: "forged_rel",
            original_cols[2]: "label",
        })

        df["real_path"]   = df["real_rel"].apply(lambda p: os.path.join(img_root, p))
        df["forged_path"] = df["forged_rel"].apply(lambda p: os.path.join(img_root, p))

        return df[["label", "real_path", "forged_path"]]

    train_df = _load(train_csv, train_img_root)
    test_df  = _load(test_csv,  test_img_root)

    LOGGER.info(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    LOGGER.info(f"Train label distribution:\n{train_df['label'].value_counts(normalize=True)}")

    return train_df, test_df

def load_signature_image(path: str) -> np.ndarray:
    return img_as_ubyte(imread(path, as_gray=True))


def normalize_image(
    img: np.ndarray,
    canvas_size: Tuple[int, int] = Config.CANVAS_SIZE,
) -> np.ndarray:

    img = img.astype(np.uint8)
    max_rows, max_cols = canvas_size
    blurred = filters.gaussian(img, sigma=2, preserve_range=True)
    threshold = filters.threshold_otsu(blurred)
    binary    = blurred > threshold
    ink_rows, ink_cols = np.where(binary == 0)
    if ink_rows.size == 0:
        canvas = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
        return canvas
    r_center = int(ink_rows.mean() - ink_rows.min())
    c_center = int(ink_cols.mean() - ink_cols.min())

    cropped   = img[ink_rows.min(): ink_rows.max(),
                    ink_cols.min(): ink_cols.max()]
    img_h, img_w = cropped.shape
    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center
    r_start = max(0, min(r_start, max_rows - img_h))
    c_start = max(0, min(c_start, max_cols - img_w))

    canvas = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    canvas[r_start:r_start + img_h, c_start:c_start + img_w] = cropped
    canvas[canvas > threshold] = 255  # FIX: was canvas(canvas > threshold).astype(np.uint8) * 255

    return canvas


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    src_h,    src_w    = img.shape
    ratio_w = src_w / target_w
    ratio_h = src_h / target_h

    if ratio_w > ratio_h:
        new_h = target_h
        new_w = int(round(src_w / ratio_h))
    else:
        new_w = target_w
        new_h = int(round(src_h / ratio_w))

    resized = transform.resize(img, (new_h, new_w),
                               mode="constant",
                               anti_aliasing=True,
                               preserve_range=True).astype(np.uint8)
    if ratio_w > ratio_h:
        start = (new_w - target_w) // 2
        return resized[:, start:start + target_w]
    else:
        start = (new_h - target_h) // 2
        return resized[start:start + target_h, :]


def crop_center(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h_in, w_in   = img.shape
    h_out, w_out = size
    r_start = (h_in - h_out) // 2
    c_start = (w_in - w_out) // 2
    return img[r_start:r_start + h_out, c_start:c_start + w_out]


def preprocess_signature(
    img: np.ndarray,
    canvas_size: Tuple[int, int] = Config.CANVAS_SIZE,
    output_size: Tuple[int, int] = Config.MODEL_INPUT_SIZE,
) -> np.ndarray:
    centered = normalize_image(img, canvas_size)
    inverted = 255 - centered                        # black bg, white ink
    resized  = resize_image(inverted, output_size)
    return resized

def _binarize(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    img = cv2.resize(img, (600, 300))

    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _skeleton(binary: np.ndarray) -> np.ndarray:
    return skeletonize(binary > 0).astype(np.uint8)


def _count_endpoints_and_junctions(
    skel: np.ndarray,
) -> Tuple[int, int]:
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1],
    ])
    neighbor_count = convolve(skel, kernel)

    endpoints  = int(np.sum(neighbor_count == 11))
    junctions  = int(np.sum(neighbor_count >= 13))
    return endpoints, junctions


def extract_topological_features(path: str) -> torch.Tensor:
    binary = _binarize(path)
    skel   = _skeleton(binary)

    endpoints, junctions = _count_endpoints_and_junctions(skel)
    stroke_length = int(np.sum(skel))
    num_components, _ = cv2.connectedComponents(binary)
    num_components -= 1
    coords = np.column_stack(np.where(binary > 0))
    coords = coords[:, ::-1]
    if coords.size > 0:
        x, y, w, h = cv2.boundingRect(coords)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    else:
        aspect_ratio = 0.0
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    avg_stroke_width = float(np.mean(dist_map[dist_map > 0])) if np.any(dist_map > 0) else 0.0

    raw_features = np.array([
        endpoints,
        junctions,
        stroke_length,
        num_components,
        aspect_ratio,
        avg_stroke_width,
    ], dtype=np.float32)

    # FIX: removed redundant mean/std variables; added epsilon to avoid divide-by-zero
    mean = raw_features.mean()
    std  = raw_features.std() + 1e-8
    normalized = (raw_features - mean) / std

    return torch.tensor(normalized, dtype=torch.float32)


class SignaturePairDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        canvas_size: Tuple[int, int] = Config.CANVAS_SIZE,
        output_size: Tuple[int, int] = Config.MODEL_INPUT_SIZE,
    ) -> None:
        self.real_paths   = df["real_path"].values
        self.forged_paths = df["forged_path"].values
        self.labels       = df["label"].values
        self.canvas_size  = canvas_size
        self.output_size  = output_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        real_path   = self.real_paths[idx]
        forged_path = self.forged_paths[idx]
        real_img   = load_signature_image(real_path)
        forged_img = load_signature_image(forged_path)
        real_processed   = preprocess_signature(real_img,   self.canvas_size, self.output_size)
        forged_processed = preprocess_signature(forged_img, self.canvas_size, self.output_size)
        real_features   = extract_topological_features(real_path)
        forged_features = extract_topological_features(forged_path)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return (
            torch.tensor(real_processed,   dtype=torch.float32),
            torch.tensor(forged_processed, dtype=torch.float32),
            real_features,
            forged_features,
            label,
        )


def _conv_bn_mish(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    pad: int = 0,
) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ("bn",   nn.BatchNorm2d(out_channels)),
        ("mish", nn.Mish()),
    ]))


def _linear_bn_mish(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ("fc",   nn.Linear(in_features, out_features, bias=False)),
        ("bn",   nn.BatchNorm1d(out_features)),
        ("mish", nn.Mish()),
    ]))


class SigNetBackbone(nn.Module):

    EMBEDDING_DIM = 2048

    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(OrderedDict([
            ("conv1",    _conv_bn_mish(1, 96, 11, stride=4)),
            ("maxpool1", nn.MaxPool2d(3, 2)),
            ("conv2",    _conv_bn_mish(96, 256, 5, pad=2)),
            ("maxpool2", nn.MaxPool2d(3, 2)),
            ("conv3",    _conv_bn_mish(256, 384, 3, pad=1)),
            ("conv4",    _conv_bn_mish(384, 384, 3, pad=1)),
            ("conv5",    _conv_bn_mish(384, 256, 3, pad=1)),
            ("maxpool3", nn.MaxPool2d(3, 2)),
        ]))
        self.fc_layers = nn.Sequential(OrderedDict([
            ("fc1", _linear_bn_mish(256 * 3 * 5, self.EMBEDDING_DIM)),
            ("fc2", _linear_bn_mish(self.EMBEDDING_DIM, self.EMBEDDING_DIM)),
        ]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)    # flatten: (B, 256*3*5)
        x = self.fc_layers(x)
        return x


class SiameseSignatureModel(nn.Module):

    def __init__(self, pretrained_weights_path: str) -> None:
        super().__init__()
        self.cnn_backbone = SigNetBackbone()
        state_dict, _, _ = torch.load(pretrained_weights_path, map_location="cpu")
        self.cnn_backbone.load_state_dict(state_dict)
        LOGGER.info("Loaded pretrained SigNet weights.")
        self.topo_projector = nn.Linear(Config.FEATURE_DIM, Config.FEATURE_PROJ_DIM)
        fused_dim = SigNetBackbone.EMBEDDING_DIM + Config.FEATURE_PROJ_DIM
        self.projection_head = nn.Linear(fused_dim, 2)
        self.classifier = nn.Linear(4, 1)

    def _embed(
        self,
        img: torch.Tensor,
        topo: torch.Tensor,
    ) -> torch.Tensor:
        img_normalized = img.view(-1, 1, *Config.MODEL_INPUT_SIZE).float().div(255)

        cnn_emb  = self.cnn_backbone(img_normalized)      # (B, 2048)
        topo_emb = self.topo_projector(topo)               # (B, 128)

        fused = torch.cat([cnn_emb, topo_emb], dim=1)     # (B, 2176)
        return self.projection_head(fused)                 # (B, 2)

    def forward(
        self,
        real_img:    torch.Tensor,
        forged_img:  torch.Tensor,
        real_topo:   torch.Tensor,
        forged_topo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1 = self._embed(real_img,   real_topo)
        emb2 = self._embed(forged_img, forged_topo)
        pair_features = torch.cat([emb1, emb2], dim=1)
        logit = self.classifier(pair_features)

        return emb1, emb2, logit


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1:   torch.Tensor,
        emb2:   torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        similarity = F.cosine_similarity(
            F.normalize(emb1, dim=1),
            F.normalize(emb2, dim=1),
        )
        # FIX: labels are flipped relative to standard contrastive loss
        # label=1 means forged (different), label=0 means genuine (same)
        genuine_loss = labels * similarity.pow(2)
        forged_loss  = (1 - labels) * torch.clamp(self.margin - similarity, min=0.0).pow(2)
        return torch.mean(genuine_loss + forged_loss)


class AverageMeter:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def get_scheduler(optimizer: torch.optim.Optimizer) -> CosineAnnealingLR:
    return CosineAnnealingLR(
        optimizer,
        T_max=Config.SCHEDULER_T_MAX,
        eta_min=Config.SCHEDULER_ETA_MIN,
    )


def train_one_epoch(
    epoch:       int,
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterions:  List[nn.Module],
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    device:      torch.device,
) -> float:

    model.train()
    loss_meter = AverageMeter()
    contrastive_loss_fn, bce_loss_fn = criterions
    start_time = time.time()

    for step, (real_img, forged_img, real_topo, forged_topo, labels) in enumerate(loader):
        real_img    = real_img.to(device)
        forged_img  = forged_img.to(device)
        real_topo   = real_topo.to(device)
        forged_topo = forged_topo.to(device)
        labels      = labels.to(device)

        batch_size = labels.size(0)
        emb1, emb2, logits = model(real_img, forged_img, real_topo, forged_topo)
        loss_contrastive = contrastive_loss_fn(emb1, emb2, labels)
        loss_bce = bce_loss_fn(logits.squeeze(1), labels)
        total_loss = (loss_contrastive + loss_bce) / 2.0
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.MAX_GRAD_NORM
        )
        optimizer.step()
        loss_meter.update(total_loss.item(), batch_size)

        if step % Config.PRINT_FREQ == 0 or step == len(loader) - 1:
            elapsed   = time.time() - start_time
            fraction  = (step + 1) / len(loader)
            remaining = elapsed / fraction - elapsed
            current_lr = scheduler.get_last_lr()[0]

            LOGGER.info(
                f"Epoch [{epoch}][{step}/{len(loader)}] "
                f"Loss: {loss_meter.val:.4f} (avg {loss_meter.avg:.4f})  "
                f"Grad: {grad_norm:.4f}  LR: {current_lr:.6f}  "
                f"ETA: {format_time(remaining)}"
            )

    return loss_meter.avg

def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for real_img, forged_img, real_topo, forged_topo, labels in loader:

            real_img    = real_img.to(device)
            forged_img  = forged_img.to(device)
            real_topo   = real_topo.to(device)
            forged_topo = forged_topo.to(device)

            _, _, logits = model(real_img, forged_img, real_topo, forged_topo)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds   = [1 if p >= 0.5 else 0 for p in all_probs]
    acc     = accuracy_score(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": acc, "auc": roc_auc}

def main() -> None:
    set_seed(Config.SEED)
    DATA_ROOT = "C:\\dataset\\sign"
    WEIGHTS   = "C:\\dataset\\sign\\signet.pth"

    train_df, test_df = load_dataframes(
        train_csv      = f"{DATA_ROOT}/train_data.csv",
        test_csv       = f"{DATA_ROOT}/test_data.csv",
        train_img_root = f"{DATA_ROOT}/train",
        test_img_root  = f"{DATA_ROOT}/test",
    )

    train_dataset = SignaturePairDataset(train_df)
    test_dataset  = SignaturePairDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model     = SiameseSignatureModel(pretrained_weights_path=WEIGHTS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE,
                     weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)

    contrastive_loss = ContrastiveLoss(margin=1.0)
    bce_loss         = nn.BCEWithLogitsLoss()
    best_loss          = float("inf")
    epochs_no_improve  = 0

    for epoch in range(Config.EPOCHS):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"EPOCH {epoch + 1} / {Config.EPOCHS}")
        LOGGER.info(f"{'='*60}")

        avg_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterions=[contrastive_loss, bce_loss],
            scheduler=scheduler,
            device=DEVICE,
        )

        scheduler.step()

        LOGGER.info(f"Epoch {epoch+1} — avg train loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_path = os.path.join(Config.OUTPUT_DIR, Config.MODEL_SAVE_NAME)
            torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)
            LOGGER.info(f"  Saved new best model (loss={best_loss:.4f}) → {save_path}")
        else:
            epochs_no_improve += 1
            LOGGER.info(f"  No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
            LOGGER.info("Early stopping triggered.")
            break

    LOGGER.info("Training complete.")

def predict_pair(
    model:       nn.Module,
    img_path_1:  str,
    img_path_2:  str,
    device:      torch.device,
    threshold:   float = 0.5,
) -> Dict:
    model.eval()

    def _prepare(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = load_signature_image(path)
        processed = preprocess_signature(raw)
        img_tensor  = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
        topo_tensor = extract_topological_features(path).unsqueeze(0)
        return img_tensor.to(device), topo_tensor.to(device)

    img1, topo1 = _prepare(img_path_1)
    img2, topo2 = _prepare(img_path_2)

    with torch.no_grad():
        emb1, emb2, logit = model(img1, img2, topo1, topo2)

    confidence = torch.sigmoid(logit).item()
    similarity = F.cosine_similarity(emb1, emb2).item()
    verdict    = "Forged" if confidence >= threshold else "Genuine"

    return {
        "verdict":    verdict,
        "confidence": round(confidence, 4),
        "similarity": round(similarity, 4),
    }

def plot_embedding_distances(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    genuine_distances = []
    forged_distances  = []

    with torch.no_grad():
        for real_img, forged_img, real_topo, forged_topo, labels in loader:

            real_img    = real_img.to(device)
            forged_img  = forged_img.to(device)
            real_topo   = real_topo.to(device)
            forged_topo = forged_topo.to(device)

            emb1, emb2, _ = model(real_img, forged_img, real_topo, forged_topo)
            dist = F.pairwise_distance(emb1, emb2).cpu().item()

            if labels.item() == 0:
                genuine_distances.append(dist)
            else:
                forged_distances.append(dist)

    plt.figure(figsize=(8, 5))
    plt.hist(genuine_distances, bins=40, alpha=0.6, label="Genuine", color="steelblue")
    plt.hist(forged_distances,  bins=40, alpha=0.6, label="Forged",  color="tomato")
    plt.xlabel("Pairwise Distance in Embedding Space")
    plt.ylabel("Number of Pairs")
    plt.title("Embedding Distance Distribution — Genuine vs Forged")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for real_img, forged_img, real_topo, forged_topo, labels in loader:

            real_img    = real_img.to(device)
            forged_img  = forged_img.to(device)
            real_topo   = real_topo.to(device)
            forged_topo = forged_topo.to(device)

            _, _, logit = model(real_img, forged_img, real_topo, forged_topo)
            all_probs.append(torch.sigmoid(logit).cpu().item())
            all_labels.append(labels.item())

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Signature Verification")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
