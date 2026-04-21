# =============================================================================
# SIGNATURE MATCHING SYSTEM — Full Rewrite
# =============================================================================
# WHAT THIS FILE DOES:
#   Trains a Siamese Neural Network to verify whether two signatures are genuine
#   or forged. It fuses two sources of information:
#     1. Deep visual features from a CNN backbone (SigNet)
#     2. Hand-crafted topological features (stroke endpoints, junctions, etc.)
#
# HOW TO RUN (on Kaggle):
#   Simply run each cell top-to-bottom. GPU is strongly recommended.
#
# DATASET USED:
#   robinreni/signature-verification-dataset  (via kagglehub)
#   medali1992/pretrained-signature-weights   (SigNet pretrained weights)
# =============================================================================


# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
# We group imports by category so it is easy to find what comes from where.

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

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

# This allows loading images that are slightly corrupted (missing end bytes).
# Without this, such images would crash the DataLoader mid-training.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Detect GPU. Training on CPU is possible but will be very slow.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# SECTION 2 — CONFIGURATION
# =============================================================================
# WHY USE A CONFIG CLASS?
#   Keeping every hyperparameter in one place means you never have to hunt
#   through the code to change the learning rate or batch size. It also makes
#   experiment tracking much cleaner — you can log the whole config to W&B
#   in one line.

class Config:
    # ---- Reproducibility ----
    SEED = 42

    # ---- Image sizes ----
    # CANVAS_SIZE: a large blank canvas that each signature is centered onto
    #   before resizing. Making it large avoids clipping big signatures.
    CANVAS_SIZE: Tuple[int, int] = (952, 1360)

    # MODEL_INPUT_SIZE: the exact (H, W) that SigNet expects.
    # The original SigNet paper uses 150 × 220.
    MODEL_INPUT_SIZE: Tuple[int, int] = (150, 220)

    # ---- Training ----
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    MAX_GRAD_NORM = 1000          # gradient clipping ceiling
    PRINT_FREQ = 100              # print loss every N steps
    EARLY_STOP_PATIENCE = 5      # stop if no improvement for this many epochs

    # ---- Model ----
    FC_DIM = 512                 # size of the 2-D projection head output
    FEATURE_DIM = 6              # number of topological features per signature
    FEATURE_PROJ_DIM = 128       # project 6 → 128 before concatenating with CNN emb

    # ---- Scheduler ----
    # CosineAnnealingLR smoothly decays the LR from LEARNING_RATE down to
    # ETA_MIN over T_MAX epochs, then restarts. This helps the model escape
    # local minima at the start of each restart.
    SCHEDULER_T_MAX = 4
    SCHEDULER_ETA_MIN = 1e-5

    # ---- Output ----
    OUTPUT_DIR = "./"
    MODEL_SAVE_NAME = "siamese_best_loss.pt"

    # ---- W&B ----
    WANDB_PROJECT = "signature-matching"
    WANDB_RUN_NAME = "siamese-v1"


# =============================================================================
# SECTION 3 — REPRODUCIBILITY SETUP
# =============================================================================
# WHY DO WE NEED THIS?
#   Neural network training involves many random operations: weight
#   initialization, data shuffling, dropout, etc. Without fixing the random
#   seed, every run gives different results, making it impossible to fairly
#   compare experiments. Seeding all sources of randomness (Python, NumPy,
#   PyTorch CPU and GPU) ensures results are reproducible.

def set_seed(seed: int = Config.SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


# =============================================================================
# SECTION 4 — LOGGING
# =============================================================================
# WHY USE logging INSTEAD OF print?
#   - Logs can be written to a file and the console at the same time.
#   - Log levels (DEBUG, INFO, WARNING, ERROR) let you filter noise.
#   - In production, you can redirect logs to monitoring systems.
#   - print() output can get lost or interleaved in multi-process training.

def build_logger(log_path: str = Config.OUTPUT_DIR + "train.log") -> logging.Logger:
    """
    Create a logger that writes to both stdout and a log file.

    Args:
        log_path: Path to the output .log file.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("signature_matcher")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")

    # Console handler — shows messages in the terminal / notebook output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    # File handler — writes every message to a persistent .log file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

LOGGER = build_logger()


# =============================================================================
# SECTION 5 — DATA LOADING
# =============================================================================
# The dataset CSV has two image path columns and a label column.
# Label = 0  →  the pair (real, real) — a genuine match
# Label = 1  →  the pair (real, forged) — a forged signature

def load_dataframes(
    train_csv: str,
    test_csv: str,
    train_img_root: str,
    test_img_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the train/test CSVs, adding full image path columns.

    Args:
        train_csv:      Path to train_data.csv
        test_csv:       Path to test_data.csv
        train_img_root: Root folder for training images
        test_img_root:  Root folder for test images

    Returns:
        (train_df, test_df) — DataFrames with columns:
            label, real_path, forged_path
    """
    def _load(csv_path: str, img_root: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        # The CSV uses the first image path as the column header.
        # We rename to something readable.
        original_cols = df.columns.tolist()
        df = df.rename(columns={
            original_cols[0]: "real_rel",    # relative path to real image
            original_cols[1]: "forged_rel",  # relative path to forged image
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


# =============================================================================
# SECTION 6 — IMAGE PREPROCESSING
# =============================================================================
# WHY PREPROCESS SIGNATURES?
#   Raw signature images vary wildly in scale, position, background brightness,
#   and noise. If we feed them directly to the network, it will spend capacity
#   learning "this person signs on white paper" rather than "this person has a
#   looped g". Preprocessing normalizes away everything except the ink.
#
# PIPELINE OVERVIEW:
#   load_signature_image()
#       → normalize_image()       center on canvas, remove background noise
#       → resize_image()          scale to target height/width
#       → crop_center()           final crop to exact model input size


def load_signature_image(path: str) -> np.ndarray:
    """
    Load a signature image as an 8-bit grayscale numpy array.

    Args:
        path: File path to the signature image (PNG, JPG, etc.)

    Returns:
        np.ndarray of shape (H, W), dtype uint8, values in [0, 255].
    """
    return img_as_ubyte(imread(path, as_gray=True))


def normalize_image(
    img: np.ndarray,
    canvas_size: Tuple[int, int] = Config.CANVAS_SIZE,
) -> np.ndarray:
    """
    Center a signature on a blank canvas and remove background noise.

    Steps:
      1. Blur lightly to suppress tiny specks.
      2. Find OTSU threshold to separate ink from background.
      3. Find the bounding box of all ink pixels.
      4. Place the cropped signature in the center of a white canvas.
      5. Set all pixels above the threshold back to 255 (clean background).

    Args:
        img:         Grayscale uint8 image of the signature.
        canvas_size: (H, W) of the output canvas. Should be larger than
                     any expected signature to avoid clipping.

    Returns:
        np.ndarray of shape canvas_size, dtype uint8.
    """
    img = img.astype(np.uint8)
    max_rows, max_cols = canvas_size

    # Step 1 — blur to remove salt-and-pepper noise before thresholding
    blurred = filters.gaussian(img, sigma=2, preserve_range=True)

    # Step 2 — OTSU finds the best threshold automatically by maximizing
    # inter-class variance. Works without needing a fixed value.
    threshold = filters.threshold_otsu(img)
    binary    = blurred > threshold      # True = background, False = ink

    # Step 3 — locate ink pixels (where binary is False)
    ink_rows, ink_cols = np.where(binary == 0)
    if ink_rows.size == 0:
        # Edge case: completely blank image — return as-is on canvas
        canvas = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
        return canvas

    # Center of mass of the ink, relative to its bounding box
    r_center = int(ink_rows.mean() - ink_rows.min())
    c_center = int(ink_cols.mean() - ink_cols.min())

    # Tight crop around the ink
    cropped   = img[ink_rows.min(): ink_rows.max(),
                    ink_cols.min(): ink_cols.max()]
    img_h, img_w = cropped.shape

    # Step 4 — place cropped signature at the center of the canvas
    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    # Clamp so we never go out of bounds
    r_start = max(0, min(r_start, max_rows - img_h))
    c_start = max(0, min(c_start, max_cols - img_w))

    canvas = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    canvas[r_start:r_start + img_h, c_start:c_start + img_w] = cropped

    # Step 5 — clean background: anything brighter than threshold → pure white
    canvas[canvas > threshold] = 255

    return canvas


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to fit within `size` without stretching (aspect-preserving).

    The image is scaled so the *smaller* ratio dimension matches exactly,
    then the other dimension is center-cropped.

    Args:
        img:  Grayscale uint8 image.
        size: Target (H, W).

    Returns:
        np.ndarray of exactly shape `size`, dtype uint8.
    """
    target_h, target_w = size
    src_h,    src_w    = img.shape

    # Which dimension is the bottleneck?
    ratio_w = src_w / target_w
    ratio_h = src_h / target_h

    if ratio_w > ratio_h:
        # Width is the bottleneck — match height exactly, width will overshoot
        new_h = target_h
        new_w = int(round(src_w / ratio_h))
    else:
        new_w = target_w
        new_h = int(round(src_h / ratio_w))

    resized = transform.resize(img, (new_h, new_w),
                               mode="constant",
                               anti_aliasing=True,
                               preserve_range=True).astype(np.uint8)

    # Center-crop the overshooting dimension
    if ratio_w > ratio_h:
        start = (new_w - target_w) // 2
        return resized[:, start:start + target_w]
    else:
        start = (new_h - target_h) // 2
        return resized[start:start + target_h, :]


def crop_center(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the exact center of an image.

    Useful as a final step after resizing to guarantee the output is exactly
    the model's expected input size.

    Args:
        img:  Image array of shape (H_in, W_in).
        size: Desired output (H_out, W_out). Must be ≤ input dimensions.

    Returns:
        np.ndarray of shape `size`.
    """
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
    """
    Full preprocessing pipeline: center → invert → resize → crop.

    WHY INVERT?
        SigNet was trained on white-ink-on-black-background images.
        Most real-world scans are black-ink-on-white. Inverting aligns
        our data with what the pretrained network expects.

    Args:
        img:         Raw grayscale signature image.
        canvas_size: Size of the centering canvas.
        output_size: Final (H, W) fed to the model.

    Returns:
        np.ndarray of shape `output_size`, dtype uint8.
    """
    centered = normalize_image(img, canvas_size)
    inverted = 255 - centered                        # black bg, white ink
    resized  = resize_image(inverted, output_size)
    return resized


# =============================================================================
# SECTION 7 — TOPOLOGICAL FEATURE EXTRACTION
# =============================================================================
# WHY TOPOLOGICAL FEATURES?
#   CNNs are great at texture and local patterns, but signatures also have
#   structural properties: how many pen lifts? how many loops? how long are
#   the strokes? These are hard for a CNN to capture directly but easy to
#   compute analytically. We use 6 such features:
#
#   1. endpoints  — pen-lift points (degree-1 skeleton pixels)
#   2. junctions  — crossing/branching points (degree≥3 skeleton pixels)
#   3. stroke_length — total ink pixels in the skeleton (proxy for complexity)
#   4. components — number of disconnected ink regions (pen lifts)
#   5. aspect_ratio — width / height of the bounding box
#   6. avg_stroke_width — mean distance-transform value (how thick the strokes are)

def _binarize(path: str) -> np.ndarray:
    """
    Load and binarize a signature image using OTSU thresholding.

    Returns a uint8 binary image (255 = ink, 0 = background),
    resized to a fixed canvas for consistent feature computation.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")

    # Resize to a standard size so feature values are scale-independent
    img = cv2.resize(img, (600, 300))

    # OTSU + invert so ink = 255
    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _skeleton(binary: np.ndarray) -> np.ndarray:
    """
    Thin the ink regions down to 1-pixel-wide skeleton lines.

    Skeletonization preserves the topology of the signature (endpoints,
    junctions, loops) while removing thickness variation, making features
    more robust to pen pressure differences.

    Returns:
        uint8 array where 1 = skeleton pixel, 0 = background.
    """
    return skeletonize(binary > 0).astype(np.uint8)


def _count_endpoints_and_junctions(
    skel: np.ndarray,
) -> Tuple[int, int]:
    """
    Count pen endpoints and junction points in the skeleton.

    HOW IT WORKS:
        For each skeleton pixel, count its 8-connected neighbors.
        - Exactly 1 neighbor  → endpoint (pen lift or start)
        - 3 or more neighbors → junction (crossing or branching)

    We use a convolution trick: convolve with a 3×3 kernel where the
    center is 10 and all neighbors are 1. The result at each skeleton
    pixel = 10 + (number of neighbors). So:
        value == 11  → 1 neighbor  → endpoint
        value >= 13  → 3+ neighbors → junction

    Args:
        skel: Binary skeleton array (1 = skeleton, 0 = background).

    Returns:
        (endpoints, junctions) as integers.
    """
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
    """
    Compute a 6-dimensional topological feature vector for a signature image.

    Features are normalized to zero mean / unit variance so they enter the
    neural network on the same scale as each other (and roughly the same
    scale as the CNN embeddings after batch norm).

    Args:
        path: Path to the signature image file.

    Returns:
        torch.Tensor of shape (6,), dtype float32, normalized.
    """
    binary = _binarize(path)
    skel   = _skeleton(binary)

    endpoints, junctions = _count_endpoints_and_junctions(skel)

    # Total skeleton pixels ≈ total stroke length in pixels
    stroke_length = int(np.sum(skel))

    # Connected components of the binary image ≈ number of separate pen strokes
    num_components, _ = cv2.connectedComponents(binary)

    # Bounding box aspect ratio: wide signatures look different from tall ones
    coords = np.column_stack(np.where(binary > 0))
    if coords.size > 0:
        x, y, w, h = cv2.boundingRect(coords)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    else:
        aspect_ratio = 0.0

    # Average stroke width via distance transform:
    # dist[p] = distance from pixel p to the nearest background pixel.
    # Averaging over ink pixels gives mean half-width.
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

    # Z-score normalization: (x - mean) / std
    # This is critical — without it, stroke_length (~5000) would dominate
    # endpoints (~50), making most features invisible to the linear layer.
    mean = raw_features.mean()
    std  = raw_features.std() + 1e-8    # +epsilon prevents division by zero
    normalized = (raw_features - mean) / std

    return torch.tensor(normalized, dtype=torch.float32)


# =============================================================================
# SECTION 8 — DATASET
# =============================================================================
# WHY A CUSTOM DATASET?
#   PyTorch's DataLoader expects a Dataset object with __len__ and __getitem__.
#   Our dataset returns a pair of images (real + forged/genuine) along with
#   their topological features and a binary label.
#
# LABEL CONVENTION:
#   label = 0  →  genuine pair  (real vs. real from same person)
#   label = 1  →  forged pair   (real vs. forged from different person)

class SignaturePairDataset(Dataset):
    """
    Dataset that yields (real_image, comparison_image, topo_feat_1,
    topo_feat_2, label) tuples for training a Siamese network.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        canvas_size: Tuple[int, int] = Config.CANVAS_SIZE,
        output_size: Tuple[int, int] = Config.MODEL_INPUT_SIZE,
    ) -> None:
        """
        Args:
            df:          DataFrame with columns [label, real_path, forged_path].
            canvas_size: Canvas for centering during preprocessing.
            output_size: Final image size fed to the model.
        """
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

        # Load raw grayscale images
        real_img   = load_signature_image(real_path)
        forged_img = load_signature_image(forged_path)

        # Run the full preprocessing pipeline on both images
        real_processed   = preprocess_signature(real_img,   self.canvas_size, self.output_size)
        forged_processed = preprocess_signature(forged_img, self.canvas_size, self.output_size)

        # Extract topological features (independently for each image)
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


# =============================================================================
# SECTION 9 — MODEL ARCHITECTURE
# =============================================================================
#
# OVERVIEW OF THE ARCHITECTURE:
#
#   [Real Image]  → SigNet CNN → 2048-d embedding ──┐
#                                                     cat → 2176-d → Linear → 2-d (emb1)
#   [Real Topo Features] → Linear(6→128) ───────────┘
#
#   [Forged Image] → SigNet CNN → 2048-d embedding ─┐
#                                                     cat → 2176-d → Linear → 2-d (emb2)
#   [Forged Topo Features] → Linear(6→128) ──────────┘
#
#   concat(emb1, emb2) → [4-d] → Linear → logit → sigmoid → probability
#
# WHY FUSE TOPOLOGY?
#   The CNN sees pixels. Topology sees structure. A forger might copy the
#   general shape but produce different numbers of pen lifts or crossings.
#   Fusing both gives the model two independent signals to work with.
#
# WHY PROJECT TO 2-D?
#   A 2-D projection makes the embeddings easy to visualize (scatter plot)
#   and forces the model to learn a maximally compact representation.


def _conv_bn_mish(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    pad: int = 0,
) -> nn.Sequential:
    """
    Conv2d → BatchNorm2d → Mish activation block.

    WHY MISH?
        Mish (f(x) = x·tanh(softplus(x))) is a smooth, non-monotonic
        activation that outperforms ReLU on many vision tasks. It avoids
        the "dying ReLU" problem and has a bounded negative region.
    """
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ("bn",   nn.BatchNorm2d(out_channels)),
        ("mish", nn.Mish()),
    ]))


def _linear_bn_mish(in_features: int, out_features: int) -> nn.Sequential:
    """
    Linear → BatchNorm1d → Mish activation block.

    WHY NO BIAS IN LINEAR?
        BatchNorm has its own learnable bias (beta). Adding a linear bias
        before BN is redundant — BN subtracts the batch mean anyway.
    """
    return nn.Sequential(OrderedDict([
        ("fc",   nn.Linear(in_features, out_features, bias=False)),
        ("bn",   nn.BatchNorm1d(out_features)),
        ("mish", nn.Mish()),
    ]))


class SigNetBackbone(nn.Module):
    """
    CNN feature extractor for signature images.

    Architecture:
        5 convolutional layers (with batch norm and Mish) → 3 max-pools
        → 2 fully connected layers → 2048-d feature vector

    This is the SigNet architecture from:
        "Learning Signature Representations for Signature Verification"
        (Hafemann et al., 2017, arXiv:1705.05787)

    We load pretrained weights, so this acts as a strong feature extractor
    from the start rather than learning from scratch.
    """

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

        # After the conv stack, feature maps are 256 × 3 × 5 = 3840 values.
        # Two FC layers compress these to a 2048-d embedding.
        self.fc_layers = nn.Sequential(OrderedDict([
            ("fc1", _linear_bn_mish(256 * 3 * 5, self.EMBEDDING_DIM)),
            ("fc2", _linear_bn_mish(self.EMBEDDING_DIM, self.EMBEDDING_DIM)),
        ]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (B, 1, 150, 220), values in [0, 1].

        Returns:
            Tensor of shape (B, 2048) — the signature embedding.
        """
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)    # flatten: (B, 256*3*5)
        x = self.fc_layers(x)
        return x


class SiameseSignatureModel(nn.Module):
    """
    Full Siamese model for signature verification.

    Combines:
        - SigNetBackbone (pretrained CNN) for visual features
        - A small Linear layer for topological features
        - A 2-D projection head to create visualizable embeddings
        - A final classifier that takes the concatenated 2-D pair

    Inputs:  (real_img, forged_img, real_topo, forged_topo)
    Outputs: (embedding1, embedding2, logit)
        - embedding1/2: 2-D embeddings, useful for scatter visualization
        - logit: raw score before sigmoid (use BCEWithLogitsLoss with this)
    """

    def __init__(self, pretrained_weights_path: str) -> None:
        """
        Args:
            pretrained_weights_path: Path to the .pth file with SigNet weights.
        """
        super().__init__()

        # ---- CNN backbone (loaded with pretrained weights) ----
        self.cnn_backbone = SigNetBackbone()
        state_dict, _, _ = torch.load(pretrained_weights_path, map_location="cpu")
        self.cnn_backbone.load_state_dict(state_dict)
        LOGGER.info("Loaded pretrained SigNet weights.")

        # ---- Topological feature projector ----
        # Projects 6 topological features → 128-d so they are comparable
        # in scale to the 2048-d CNN embedding when concatenated.
        self.topo_projector = nn.Linear(Config.FEATURE_DIM, Config.FEATURE_PROJ_DIM)

        # ---- 2-D projection head ----
        # Takes (2048 + 128) = 2176-d fused embedding → 2-d
        fused_dim = SigNetBackbone.EMBEDDING_DIM + Config.FEATURE_PROJ_DIM
        self.projection_head = nn.Linear(fused_dim, 2)

        # ---- Binary classifier ----
        # Takes concat(emb1, emb2) = 4-d → 1 logit
        self.classifier = nn.Linear(4, 1)

    def _embed(
        self,
        img: torch.Tensor,
        topo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce a 2-D embedding for one image in the pair.

        Args:
            img:  Tensor (B, H, W) — raw pixel values (0-255).
            topo: Tensor (B, 6)   — normalized topological features.

        Returns:
            Tensor (B, 2) — the 2-D embedding.
        """
        # Normalize pixels to [0, 1] and add channel dimension → (B, 1, H, W)
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
        """
        Forward pass through the full Siamese network.

        Returns:
            emb1:  2-D embedding of the real signature     (B, 2)
            emb2:  2-D embedding of the forged signature   (B, 2)
            logit: raw classification score                (B, 1)
        """
        emb1 = self._embed(real_img,   real_topo)     # (B, 2)
        emb2 = self._embed(forged_img, forged_topo)   # (B, 2)

        # Concatenate the two 2-D embeddings → (B, 4) → (B, 1)
        pair_features = torch.cat([emb1, emb2], dim=1)
        logit = self.classifier(pair_features)

        return emb1, emb2, logit


# =============================================================================
# SECTION 10 — LOSS FUNCTIONS
# =============================================================================
#
# We use TWO losses simultaneously:
#
#   Loss 1 — ContrastiveLoss (on the 2-D embeddings)
#       Pushes genuine pairs CLOSE together in embedding space.
#       Pulls forged pairs APART (up to a margin).
#       This shapes the geometry of the embedding space directly.
#
#   Loss 2 — BCEWithLogitsLoss (on the classifier logit)
#       Standard binary cross-entropy. Trains the final classifier
#       to distinguish genuine from forged based on the embeddings.
#
#   Total loss = (Loss1 + Loss2) / 2
#
# WHY TWO LOSSES?
#   Each loss has a different inductive bias. Contrastive loss optimizes the
#   embedding geometry; BCE optimizes the decision boundary. Together they
#   make the model both well-calibrated and well-separated.


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    For a genuine pair  (label=0): loss = similarity²
        → minimize cosine similarity → push embeddings apart? No — wait.

    NOTE ON SIGN CONVENTION:
        We use cosine_similarity (higher = more similar).
        For genuine pairs (label=0), we WANT high similarity → minimize (1-label)*sim²
        For forged  pairs (label=1), we WANT low similarity  → minimize label*(margin-sim)²

    Reference: Hadsell, Chopra, LeCun (2006) — generalized here to cosine space.

    Args:
        margin: Similarity below which forged pairs incur no penalty. Default=1.0.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1:   torch.Tensor,
        emb2:   torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb1:   (B, D) embeddings for the first image in each pair.
            emb2:   (B, D) embeddings for the second image in each pair.
            labels: (B,) — 0 = genuine, 1 = forged.

        Returns:
            Scalar loss tensor.
        """
        # Normalize so cosine similarity is the right metric
        similarity = F.cosine_similarity(
            F.normalize(emb1, dim=1),
            F.normalize(emb2, dim=1),
        )

        # Genuine pairs: penalize if similarity is low
        genuine_loss = (1 - labels) * similarity.pow(2)

        # Forged pairs: penalize if similarity is still high (above margin)
        forged_loss  = labels * torch.clamp(self.margin - similarity, min=0.0).pow(2)

        return torch.mean(genuine_loss + forged_loss)


# =============================================================================
# SECTION 11 — TRAINING UTILITIES
# =============================================================================

class AverageMeter:
    """
    Tracks a running average of a scalar (e.g., loss or accuracy).

    Useful for printing smooth per-epoch averages rather than noisy per-batch values.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Args:
            val: New value to incorporate.
            n:   Batch size (so we weight by sample count, not step count).
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable 'Xm Ys' string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def get_scheduler(optimizer: torch.optim.Optimizer) -> CosineAnnealingLR:
    """
    Build the learning rate scheduler.

    We use CosineAnnealingLR which decays the LR as a cosine curve from
    LEARNING_RATE down to ETA_MIN over T_MAX epochs, then restarts.
    This provides:
      - Fast initial progress (high LR)
      - Fine-grained convergence near the minimum (low LR)
      - Periodic restarts to escape local minima
    """
    return CosineAnnealingLR(
        optimizer,
        T_max=Config.SCHEDULER_T_MAX,
        eta_min=Config.SCHEDULER_ETA_MIN,
    )


# =============================================================================
# SECTION 12 — TRAINING LOOP
# =============================================================================

def train_one_epoch(
    epoch:       int,
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterions:  List[nn.Module],
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    device:      torch.device,
) -> float:
    """
    Run one full training epoch.

    Args:
        epoch:      Current epoch index (0-based).
        model:      The SiameseSignatureModel.
        loader:     DataLoader yielding (real_img, forged_img, real_topo,
                    forged_topo, label) batches.
        optimizer:  Adam optimizer.
        criterions: [ContrastiveLoss, BCEWithLogitsLoss]
        scheduler:  LR scheduler (stepped per epoch, not per batch).
        device:     "cuda" or "cpu".

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    contrastive_loss_fn, bce_loss_fn = criterions
    start_time = time.time()

    for step, (real_img, forged_img, real_topo, forged_topo, labels) in enumerate(loader):

        # Move all tensors to the target device
        real_img    = real_img.to(device)
        forged_img  = forged_img.to(device)
        real_topo   = real_topo.to(device)
        forged_topo = forged_topo.to(device)
        labels      = labels.to(device)

        batch_size = labels.size(0)

        # ---- Forward pass ----
        emb1, emb2, logits = model(real_img, forged_img, real_topo, forged_topo)

        # ---- Compute losses ----
        # Contrastive loss: shapes the embedding space geometry
        loss_contrastive = contrastive_loss_fn(emb1, emb2, labels)

        # BCE loss: trains the binary classifier on top of the embeddings
        loss_bce = bce_loss_fn(logits.squeeze(1), labels)

        # Average the two losses — they are on comparable scales
        total_loss = (loss_contrastive + loss_bce) / 2.0

        # ---- Backward pass ----
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping: prevents exploding gradients. If the gradient
        # norm exceeds MAX_GRAD_NORM, all gradients are rescaled so the
        # total norm equals MAX_GRAD_NORM.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.MAX_GRAD_NORM
        )

        optimizer.step()

        # ---- Logging ----
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

            wandb.log({
                "step_loss": loss_meter.val,
                "learning_rate": current_lr,
            })

    return loss_meter.avg


# =============================================================================
# SECTION 13 — EVALUATION
# =============================================================================

def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run inference on a DataLoader and return evaluation metrics.

    Metrics returned:
        - accuracy:  fraction of correctly classified pairs
        - auc:       area under the ROC curve (threshold-independent)

    Args:
        model:  Trained SiameseSignatureModel (will be set to eval mode).
        loader: DataLoader for the evaluation split.
        device: "cuda" or "cpu".

    Returns:
        Dict with keys "accuracy" and "auc".
    """
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

            # sigmoid converts the raw logit to a probability in (0, 1)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds   = [1 if p >= 0.5 else 0 for p in all_probs]
    acc     = accuracy_score(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": acc, "auc": roc_auc}


# =============================================================================
# SECTION 14 — MAIN TRAINING SCRIPT
# =============================================================================

def main() -> None:
    """
    End-to-end training pipeline.

    1. Load and inspect data
    2. Build datasets and dataloaders
    3. Initialize model, optimizer, scheduler, losses
    4. Train for Config.EPOCHS epochs
    5. Save the best model checkpoint
    """

    # ---- W&B setup ----
    # IMPORTANT: never hardcode the API key. Use an environment variable.
    # Set it with:  export WANDB_API_KEY="your_key_here"
    # Or on Kaggle: Add-ons → Secrets → Add WANDB_API_KEY
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        LOGGER.warning("WANDB_API_KEY not set — running W&B in offline mode.")
        wandb.login(anonymous="must")

    wandb.init(
        project=Config.WANDB_PROJECT,
        name=Config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(Config).items() if not k.startswith("_")},
    )

    set_seed(Config.SEED)

    # ---- Data ----
    DATA_ROOT = "../input/datasets/robinreni/signature-verification-dataset/sign_data"
    WEIGHTS   = "../input/datasets/medali1992/pretrained-signature-weights/signet.pth"

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
        drop_last=True,    # avoids BatchNorm errors on tiny last batches
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Model ----
    model     = SiameseSignatureModel(pretrained_weights_path=WEIGHTS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE,
                     weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)

    contrastive_loss = ContrastiveLoss(margin=1.0)
    bce_loss         = nn.BCEWithLogitsLoss()

    # ---- Training loop ----
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

        scheduler.step()    # advance LR schedule after each epoch

        LOGGER.info(f"Epoch {epoch+1} — avg train loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        # ---- Save checkpoint if improved ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_path = os.path.join(Config.OUTPUT_DIR, Config.MODEL_SAVE_NAME)
            torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)
            LOGGER.info(f"  Saved new best model (loss={best_loss:.4f}) → {save_path}")
        else:
            epochs_no_improve += 1
            LOGGER.info(f"  No improvement for {epochs_no_improve} epoch(s).")

        # ---- Early stopping ----
        if epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
            LOGGER.info("Early stopping triggered.")
            break

    wandb.finish()
    LOGGER.info("Training complete.")


# =============================================================================
# SECTION 15 — INFERENCE (single pair)
# =============================================================================

def predict_pair(
    model:       nn.Module,
    img_path_1:  str,
    img_path_2:  str,
    device:      torch.device,
    threshold:   float = 0.5,
) -> Dict:
    """
    Predict whether two signatures belong to the same person.

    Args:
        model:      Trained SiameseSignatureModel.
        img_path_1: Path to the first (reference) signature.
        img_path_2: Path to the second (query) signature.
        device:     "cuda" or "cpu".
        threshold:  Probability above which we call it "Forged".

    Returns:
        Dict with keys:
            verdict:     "Genuine" or "Forged"
            confidence:  probability in (0, 1)
            similarity:  cosine similarity of the 2-D embeddings
    """
    model.eval()

    def _prepare(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load, preprocess, and extract features for one image."""
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


# =============================================================================
# SECTION 16 — VISUALIZATIONS
# =============================================================================

def plot_embedding_distances(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Plot the distribution of pairwise distances for genuine vs forged pairs.

    A well-trained model should show two clearly separated histograms:
    - Genuine pairs: small distances (embeddings close together)
    - Forged pairs:  large distances (embeddings far apart)
    """
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
    """
    Plot the ROC curve and print the AUC score.

    The ROC curve shows the tradeoff between:
    - True Positive Rate  (correctly catching forgeries)
    - False Positive Rate (incorrectly flagging genuine signatures)

    A perfect model has AUC = 1.0. Random guessing gives AUC = 0.5.
    """
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


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
