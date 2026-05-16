from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
# IF forking the workers starts creating issues, try changing the start method to spawn (safer on CUDA)
#import torch.multiprocessing as mp
#mp.set_start_method("spawn")

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet

"""
Trains a 1D Convolutional Neural Network (CNN) on top of frozen RiNALMo
embeddings to REGRESS per-window RNA structure scores as continuous values.

CHANGES from train_cnn_rinalmo_initial.py:
    1. TEST SET: A held-out test set (TEST_FRAC) is split off BEFORE
       train/val splitting, so it is never seen during training or
       hyperparameter tuning.
    2. METADATA ONE-HOT ENCODING: Experimental conditions (Method, Reagent,
       Condition, Specificity) are one-hot encoded and concatenated with
       the RiNALMo embeddings before the CNN head. This lets the model
       learn condition-dependent score behaviour.
    3. REGRESSION with WMSE: Instead of classification, the model outputs
       a single continuous score prediction. The loss function is a
       Weighted Mean Squared Error (WMSE) that prioritises getting
       low-reactivity (near-zero) scores right, because these are the
       most informative for RNA structure (low reactivity = likely paired).
    4. SCORE PREPROCESSING:
       - Scores below 0 are clipped to 0 (negative scores are artifacts)
       - Hard outlier cutoff at SCORE_CLIP_MAX (99th percentile, default 10.0)
       - Scores are then scaled to [0, 1] by dividing by SCORE_CLIP_MAX
    5. WMSE WEIGHTING: Uses w(y) = 1 + WMSE_ALPHA * (1 - y), so samples
       with y near 0 (paired/structured) get weight ~(1 + WMSE_ALPHA),
       while samples with y near 1 (unpaired) get weight ~1. This makes
       the model care more about accurately predicting structured regions.

Pipeline overview:
    1. Load processed CSV files (one per bacterial species)
    2. Clip and scale scores to [0, 1]
    3. One-hot encode experimental metadata and concatenate with
       RiNALMo embeddings
    4. Embed each RNA sequence using RiNALMo (frozen feature extractor)
    5. Train a lightweight CNN regressor with WMSE loss
    6. Save checkpoints every epoch for resumability on HPC (SPARTAN)
    7. Evaluate on held-out test set after training completes

- RiNALMo 'micro' (35M params, D=480) is used instead of 'giga'
      (600M params, D=1280) because with ~13M sequences, on-the-fly
      embedding with giga takes >1hr per epoch. Micro is ~10x faster.
    - GPU embedding is done in the main process, NOT in DataLoader
      workers, because CUDA cannot be re-initialised in forked
      subprocesses (a PyTorch/Linux limitation).
    - Epoch-level subsampling (default 10%) enables fast iteration while
      covering the full dataset across multiple epochs.

Author: Zofia Witkowski-Blake
Project: Predicting RNA structure scores using CNNs and RiNALMo embeddings
University of Melbourne, MSc Bioinformatics    
"""

#Tuneable constants
MODEL_SIZE   = "micro"           # "micro" | "giga"
WEIGHTS_PATH = "weights/rinalmo_micro_pretrained.pt"  # path to pretrained RiNALMo weights
BATCH_SIZE   = 256               # number of sequences per training step: safe for micro on A100 & not OOM
SUBSET_FRAC  = 0.1              # fraction of training set used per epoch (1.0 = use all data)
                                # 0.10 means ~1.3M sequences per epoch out of ~13M total
EPOCHS       = 20                # number of training epochs
LR           = 1e-3              # initial learning rate for AdamW optimiser
WEIGHT_DECAY = 1e-2              # L2 regularisation strength in AdamW

# CHANGED: split fractions now include a test set
TEST_FRAC    = 0.10              # fraction of data held out for final test (never seen during training)
VAL_FRAC     = 0.10              # fraction of REMAINING data held out for validation

SEED         = 0                 # random seed for reproducibility
NUM_WORKERS  = 4                 # CPU workers for DataLoader prefetching
USE_AMP      = True              # automatic mixed precision (float16) on GPU
CHECKPOINT_DIR = "checkpoints_regress"   # directory to save per-epoch model weights

# ADDED: Score preprocessing constants for regression
# Scores below 0 are clipped to 0 (negative scores are experimental artifacts).
# Scores above SCORE_CLIP_MAX are clipped to SCORE_CLIP_MAX (hard outlier cutoff).
# After clipping, scores are divided by SCORE_CLIP_MAX to scale into [0, 1].
# Looking at the score distributions across species:
#   - Most scores cluster between 0 and ~3
#   - 99th percentile across pooled data is approximately 10
#   - Max values range up to 738 (extreme outliers in synechococcus DMS-seq)
# A cutoff of 10.0 retains >99% of the data's dynamic range while removing
# extreme outliers that would dominate MSE and distort gradients.
SCORE_CLIP_MAX = 10.0

# ADDED: WMSE weighting parameter
# w(y) = 1 + WMSE_ALPHA * (1 - y)
# When WMSE_ALPHA = 4.0:
#   y = 0.0 (structured/paired)   -> weight = 5.0  (5x emphasis)
#   y = 0.5 (intermediate)        -> weight = 3.0
#   y = 1.0 (unpaired/flexible)   -> weight = 1.0  (baseline)
# This prioritises getting low-reactivity predictions right because
# misclassifying a paired nucleotide as unpaired is more costly for
# downstream RNA folding than the reverse error.
WMSE_ALPHA = 4.0

# Resume-from-checkpoint settings. RESUME_FROM = checkpoint path to resume an interrupted run
RESUME_FROM  = "checkpoints_regress/latest.pt"              # can be None or checkpoints/...
TOTAL_EPOCHS = 40                # total epochs INCLUDING already-completed ones

# CHANGED: Define the categorical metadata columns and their known categories
# for one-hot encoding. These must match the values present in the processed CSVs.
# Unknown/missing values get a dedicated "unknown" column.
META_CATEGORIES = {
    "Method": [
        "SHAPE-seq", "SHAPE-MaP", "DMS-seq", "Cotranscriptional_SHAPE-seq",
        "Lead-seq", "icSHAPE",
    ],
    "Reagent": [
        "DMS", "1M7", "BZCN", "NAI", "NIC", "2A3", "1M4",
        "B5", "I5", "6A3", "LEAD(II)", "HYDROXYL_RADICAL",
    ],
    "Condition": ["in_vivo", "in_vitro", "ex_vivo"],
    "Specificity": ["transcriptome-wide", "targeted"],
}


def build_onehot_vector(meta: Dict, categories: Dict[str, List[str]]) -> torch.Tensor:
    """
    ADDED: Converts categorical metadata into a single one-hot vector.

    For each metadata field (Method, Reagent, Condition, Specificity), creates
    a sub-vector of length len(known_categories) + 1 (the +1 is for "unknown"
    or missing values). Concatenates all sub-vectors into one flat tensor.

    Parameters:
    meta : Dict
        Metadata dictionary from BacterialScoresDataset.__getitem__,
        e.g. {"Method": "DMS-seq", "Reagent": "DMS", ...}
    categories : Dict[str, List[str]]
        Mapping of field name -> list of known category strings.

    Returns:
    torch.Tensor
        1D float32 tensor of length sum(len(cats)+1 for cats in categories.values())
    """
    parts = []
    for field, cats in categories.items():
        n = len(cats) + 1  # +1 for unknown/missing
        vec = torch.zeros(n, dtype=torch.float32)
        val = meta.get(field, None)
        if val is not None and val in cats:
            vec[cats.index(val)] = 1.0
        else:
            vec[-1] = 1.0  # mark as unknown/missing
        parts.append(vec)
    return torch.cat(parts)


def get_onehot_dim(categories: Dict[str, List[str]]) -> int:
    """
    ADDED: Returns the total dimensionality of the one-hot metadata vector.
    Each field contributes len(known_categories) + 1 dimensions.
    """
    return sum(len(cats) + 1 for cats in categories.values())


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = WMSE_ALPHA) -> torch.Tensor:
    """
    ADDED: Weighted Mean Squared Error loss using torch.nn.functional.mse_loss.

    Computes per-sample weights based on target value:
        w(y) = 1 + alpha * (1 - y)
    
    So samples with target near 0 (structured/paired nucleotides) receive
    higher weight than samples with target near 1 (unpaired/flexible).

    The rationale is that low-reactivity scores carry the most structural
    information — a nucleotide with near-zero reactivity is very likely
    base-paired, and getting that prediction wrong has a larger downstream
    impact on RNA folding accuracy than errors on high-reactivity positions.

    Implementation:
        1. Compute element-wise MSE via F.mse_loss(reduction='none')
        2. Multiply each sample's squared error by its weight
        3. Return the mean of the weighted errors

    Parameters:
    pred : torch.Tensor
        Predicted scores [B], values in [0, 1].
    target : torch.Tensor
        True scores [B], values in [0, 1] (after clipping and scaling).
    alpha : float
        Weight amplification factor. Higher alpha = stronger emphasis
        on low-score samples. Default is WMSE_ALPHA (4.0).

    Returns:
    torch.Tensor
        Scalar weighted MSE loss.
    """
    # Per-sample unweighted squared error
    mse_per_sample = F.mse_loss(pred, target, reduction='none')  # [B]
    # Per-sample weight: higher for low target values
    weights = 1.0 + alpha * (1.0 - target)  # [B], range [1, 1+alpha]
    # Weighted mean
    return (weights * mse_per_sample).mean()



@dataclass(frozen=True)
class CSVDatasetSpec:
    """
    Immutable container pairing a CSV file path with its species name.
    frozen=True prevents accidental modification after creation.
    """
    path: str
    species: str


class BacterialScoresDataset(Dataset):
    """
    Custom PyTorch Dataset that loads processed CSV files for multiple
    bacterial species and serves (sequence, score, metadata) tuples.

    CHANGED: This is now a regression dataset. Instead of binning scores
    into classes, the raw scores are clipped and scaled to [0, 1]:
      1. Scores < 0 are set to 0 (negative values are experimental artifacts)
      2. Scores > SCORE_CLIP_MAX are set to SCORE_CLIP_MAX (hard outlier cutoff)
      3. Clipped scores are divided by SCORE_CLIP_MAX to get [0, 1] range

    keep_meta is True by default because metadata is needed for one-hot encoding.

    Parameters:
    specs : List[CSVDatasetSpec]
        List of (path, species) pairs for each species CSV file.
    seq_col : str
        Name of the column containing RNA sequences.
    target_col : str
        Name of the column containing raw structure scores.
    keep_meta : bool
        Whether to return metadata columns (species, method, etc.).
    dropna : bool
        Whether to drop rows with missing sequence or score values.
    shuffle : bool
        Whether to shuffle the dataset after loading.
    seed : int
        Random seed for shuffling (ensures reproducibility).
    """

    def __init__(
        self,
        specs: List[CSVDatasetSpec],
        seq_col: str = "Seq",
        target_col: str = "Score",
        keep_meta: bool = True,          # CHANGED: default True for metadata encoding
        dropna: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        # Load and concatenate all species CSVs into one DataFrame
        dfs = []
        for s in specs:
            # low_memory=False prevents DtypeWarning on mixed-type columns
            df = pd.read_csv(s.path, low_memory=False)
            df["Species"] = s.species
            dfs.append(df)
        big = pd.concat(dfs, ignore_index=True)

        # Verify required columns exist. This indicates database error
        required = [seq_col, target_col, "Species"]
        missing = [c for c in required if c not in big.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(big.columns)}")
        
        # Remove rows where sequence or score is missing. 
        if dropna:
            big = big.dropna(subset=[seq_col, target_col]).reset_index(drop=True)

        # Shuffle with fixed seed for reproducibility across runs
        if shuffle:
            big = big.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.df = big
        self.seq_col = seq_col
        self.target_col = target_col
        self.keep_meta = keep_meta

        self.meta_cols = [
            "Species", "Method", "Reagent", "Temp", "Condition", "Specificity",
            "Coord", "Study ID", "Paper"
        ]
        self.meta_cols = [c for c in self.meta_cols if c in self.df.columns]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, Dict]:
        """
        Returns one sample as (sequence_string, scaled_score, metadata_dict).

        CHANGED: Instead of binning into classes, the score is clipped and
        scaled to [0, 1] for regression:
          1. Clip negatives to 0
          2. Clip above SCORE_CLIP_MAX
          3. Divide by SCORE_CLIP_MAX
        """
        row   = self.df.iloc[idx]
        seq   = str(row[self.seq_col])
        score = float(row[self.target_col])

        # CHANGED: Clip and scale score for regression
        # Scores below 0 are experimental artifacts -> clamp to 0
        score = max(score, 0.0)
        # Hard outlier cutoff -> clamp to SCORE_CLIP_MAX
        score = min(score, SCORE_CLIP_MAX)
        # Scale to [0, 1]
        score = score / SCORE_CLIP_MAX

        # dtype=torch.float32 for regression target (not int64 like classification)
        y = torch.tensor(score, dtype=torch.float32)

        meta: Dict = {}
        if self.keep_meta:
            for c in self.meta_cols:
                meta[c] = row[c]
        return seq, y, meta


# RiNALMo embedder

class RinalmoBatchEmbedder(nn.Module):
    """
    Wraps RiNALMo and converts raw RNA strings to per-token embeddings.
    Model is frozen by default, only the CNN head will be trained.
    Output shape: [B, L, D]  (B=batch, L=seq length, D=embedding dim)

    Parameters:
    weights_path : str
        Path to the pre-trained RiNALMo weights file.
    device : torch.device
        GPU or CPU device to run the model on.
    model_size : str
        RiNALMo variant: "micro" or "giga".
    use_amp : bool
        Whether to use automatic mixed precision (float16) for speed.
        -> automatic mixed precision speeds up deep learning by 
        automatically using both 16-bit (half-precision) and 32-bit 
        (single-precision) floating-point types to speed up training
    freeze : bool
        Whether to freeze all RiNALMo parameters (no gradient updates).
    """

    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        model_size: str = "micro",
        use_amp: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.device  = device
        self.use_amp = use_amp

        # Load RiNALMo architecture and pre-trained weights
        config  = model_config(model_size)
        model   = RiNALMo(config)
        alphabet = Alphabet(**config["alphabet"])

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()

        # Freeze all parameters
        if freeze:
            for p in model.parameters():
                p.requires_grad = False

        self.model    = model
        self.alphabet = alphabet

    @torch.no_grad()

    def forward(self, seqs: List[str]) -> torch.Tensor:
        """
        This function takes a list of RNA strings 
        and returns embeddings [B, L, D] as a tensor
        """
        # Tokenise: convert nucleotide characters to integer IDs
        toks   = self.alphabet.batch_tokenize(seqs)
        tokens = torch.tensor(toks, dtype=torch.int64, device=self.device)
        # Run through RiNALMo with optional mixed precision (speed)
        if self.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = self.model(tokens)
        else:
            out = self.model(tokens)
        # Extract the representation layer (per-nucleotide embeddings)
        return out["representation"]  # [B, L, D]


# Collation and batch embedding:
# CPU based data loading in forked worker processes, and GPU based embedding

def cpu_collate(batch):
    """
    Collate function for DataLoader workers (CPU-only processes).
    Assembles raw sequences, regression targets, and metadata from individual samples.

    CHANGED: Now also collates metadata dicts and one-hot vectors for metadata
    encoding. Returns (seqs, targets, onehot_meta).

    NO GPU operations. GPU embedding happens afterwards in the main process via embed_batch().

    Parameters:
    batch : list of (str, torch.Tensor, dict)
        Individual samples from BacterialScoresDataset.__getitem__

    Returns:
    (List[str], torch.Tensor, torch.Tensor)
        Sequences as strings, targets as CPU float32 tensor [B],
        one-hot metadata [B, meta_dim].
    """
    seqs, ys, metas = zip(*batch)
    y = torch.stack(ys, dim=0)  # [B] float32 regression targets on CPU
    # ADDED: build one-hot metadata vectors for each sample in batch
    onehot_list = [build_onehot_vector(m, META_CATEGORIES) for m in metas]
    onehot = torch.stack(onehot_list, dim=0)  # [B, meta_dim]
    return list(seqs), y, onehot


def embed_batch(seqs: List[str], y: torch.Tensor, onehot: torch.Tensor,
                embedder: RinalmoBatchEmbedder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs RiNALMo in the main process (where CUDA is initialised).
    Called inside the training/eval loop, not in the DataLoader.
    The embedding output is transposed from [B, L, D] to [B, D, L] because
    PyTorch's Conv1d expects input as [batch, channels, length].

    CHANGED: Now also moves the one-hot metadata tensor to GPU.

    Parameters:
    seqs : List[str]
        RNA sequences from cpu_collate.
    y : torch.Tensor
        Regression targets from cpu_collate (CPU tensor).
    onehot : torch.Tensor
        One-hot encoded metadata [B, meta_dim] from cpu_collate (CPU tensor).
    embedder : RinalmoBatchEmbedder
        The frozen RiNALMo model.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor)
        X: embedded sequences [B, D, L] on GPU
        y: regression targets [B] on GPU
        onehot: metadata [B, meta_dim] on GPU
    """
    emb = embedder(seqs)                        # [B, L, D] on GPU
    X   = emb.transpose(1, 2).contiguous()      # [B, D, L] — Conv1d format
    y   = y.to(X.device, non_blocking=True)     # move targets to GPU
    # ADDED: move metadata to GPU
    onehot = onehot.to(X.device, non_blocking=True)
    return X, y, onehot


# CHANGED: CNN regressor (outputs a single scalar, not class logits)
class SimpleCNNRegressor(nn.Module):
    """
    3-layer 1D CNN that reads per-position embeddings and predicts
    a continuous structure score in [0, 1].

    CHANGED from classifier version:
    - Final layer outputs 1 value instead of NUM_CLASSES
    - Sigmoid activation constrains output to [0, 1]
    - One-hot metadata is concatenated after pooling

    Architecture:
      [B, D, L]
        -> Conv1d(D->256, k=7) + ReLU   # detects 7-nt local patterns
        -> Conv1d(256->128, k=5) + ReLU  # combines nearby patterns
        -> Conv1d(128->64,  k=3) + ReLU  # refines
        -> AdaptiveAvgPool1d(1)           # collapses variable seq len -> fixed size
        -> Flatten -> [B, 64]
        -> Concat with one-hot metadata -> [B, 64 + meta_dim]
        -> Linear(64 + meta_dim -> 1)     # single score prediction
        -> Sigmoid                         # constrain to [0, 1]

    Parameters:
    d_model : int
        Embedding dimension from RiNALMo (480 for micro, 1280 for giga).
    meta_dim : int
        Dimensionality of the one-hot metadata vector (from get_onehot_dim).
    """

    def __init__(self, d_model: int, meta_dim: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(d_model, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          #  [B, 64, 1]
            nn.Flatten(),                      #  [B, 64]
        )
        # CHANGED: output is 1 scalar (regression) instead of NUM_CLASSES
        self.head = nn.Sequential(
            nn.Linear(64 + meta_dim, 1),      # [B, 1]
            nn.Sigmoid(),                      # constrain to [0, 1]
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        CHANGED: Outputs a single predicted score per sample.

        Parameters:
        x : torch.Tensor
            RiNALMo embeddings [B, D, L] in Conv1d format.
        meta : torch.Tensor
            One-hot encoded metadata [B, meta_dim].

        Returns:
        torch.Tensor
            Predicted scores [B] in [0, 1].
        """
        h = self.pool(self.feat(x))           # [B, 64]
        h = torch.cat([h, meta], dim=1)       # [B, 64 + meta_dim]
        return self.head(h).squeeze(-1)       # [B] — squeeze out the last dim


# CHANGED: Regression metrics (replacing classification accuracy)

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    ADDED: Pearson correlation coefficient between predictions and targets.
    Measures how well the model captures relative trends in structure
    across a transcript, even if absolute values differ.

    Returns NaN if either vector has zero variance (e.g. all same value).

    Parameters:
    pred : torch.Tensor
        Predicted scores [B].
    target : torch.Tensor
        True scores [B].

    Returns:
    float
        Pearson r in [-1, 1], or NaN.
    """
    p = pred.float().cpu()
    t = target.float().cpu()
    p_mean = p.mean()
    t_mean = t.mean()
    p_dev = p - p_mean
    t_dev = t - t_mean
    num = (p_dev * t_dev).sum()
    den = (p_dev.pow(2).sum().sqrt() * t_dev.pow(2).sum().sqrt())
    if den < 1e-8:
        return float("nan")
    return float(num / den)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    ADDED: Root Mean Squared Error between predictions and targets.
    More interpretable than MSE because it's in the same units as
    the target variable (scaled scores in [0, 1]).

    Parameters:
    pred : torch.Tensor
        Predicted scores [B].
    target : torch.Tensor
        True scores [B].

    Returns:
    float
        RMSE value.
    """
    return float(F.mse_loss(pred, target).sqrt().cpu())



# Train & eval

def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp) -> Tuple[float, float, float, float, float]:
    """
    Train the CNN regressor for one epoch.
    For each batch:
        1. cpu_collate returns raw sequences + regression targets + one-hot metadata
        2. embed_batch runs RiNALMo on GPU (main process) to get embeddings
        3. Forward pass through CNN with metadata to predicted score
        4. Compute WMSE loss (weighted MSE prioritising low scores)
        5. Backward pass to update weights

    CHANGED: Uses WMSE loss instead of CrossEntropyLoss. Collects predictions
    for Pearson r and RMSE instead of classification accuracy.

    Parameters:
    model : SimpleCNNRegressor
        The CNN regression model being trained.
    loader : DataLoader
        Training data loader (yields batches from cpu_collate).
    embedder : RinalmoBatchEmbedder
        Frozen RiNALMo for embedding sequences.
    opt : torch.optim.Optimizer
        AdamW optimiser.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    device : torch.device
        GPU device.
    use_amp : bool
        Whether to use automatic mixed precision.

    Returns:
    (avg_wmse_loss, pearson_r, rmse, elapsed_seconds, sequences_per_second)
    """

    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    # CHANGED: unpack onehot metadata from collate
    for seqs, y_cpu, onehot_cpu in loader:
        # GPU embedding + metadata transfer
        X, y, onehot = embed_batch(seqs, y_cpu, onehot_cpu, embedder)

        opt.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                pred = model(X, onehot)                         # CHANGED: regression output [B]
                loss = weighted_mse_loss(pred, y, WMSE_ALPHA)   # CHANGED: WMSE loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(X, onehot)                             # CHANGED: regression output [B]
            loss = weighted_mse_loss(pred, y, WMSE_ALPHA)       # CHANGED: WMSE loss
            loss.backward()
            opt.step()

        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_preds.append(pred.detach())
        all_targets.append(y.detach())
        n_seqs += bs

    elapsed      = time.perf_counter() - t_start
    cat_preds    = torch.cat(all_preds)
    cat_targets  = torch.cat(all_targets)
    n            = cat_targets.shape[0]
    avg_loss     = total_loss / max(n, 1)
    r            = pearson_r(cat_preds, cat_targets)
    err          = rmse(cat_preds, cat_targets)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, r, err, elapsed, seqs_per_sec


@torch.no_grad() # No gradient calculation as this is a validation function
def eval_one_epoch(model, loader, embedder, device) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the CNN regressor on the validation or test set.

    CHANGED: Uses WMSE loss and regression metrics (Pearson r, RMSE)
    instead of classification accuracy.

    Returns:
    (avg_wmse_loss, pearson_r, rmse, elapsed_seconds, sequences_per_second)
    """
    import time
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    # CHANGED: unpack onehot metadata from collate
    for seqs, y_cpu, onehot_cpu in loader:
        X, y, onehot = embed_batch(seqs, y_cpu, onehot_cpu, embedder)
        pred = model(X, onehot)                             # CHANGED: regression output
        loss = weighted_mse_loss(pred, y, WMSE_ALPHA)       # CHANGED: WMSE loss
        bs   = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_preds.append(pred)
        all_targets.append(y)
        n_seqs += bs

    elapsed      = time.perf_counter() - t_start
    cat_preds    = torch.cat(all_preds)
    cat_targets  = torch.cat(all_targets)
    n            = cat_targets.shape[0]
    avg_loss     = total_loss / max(n, 1)
    r            = pearson_r(cat_preds, cat_targets)
    err          = rmse(cat_preds, cat_targets)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, r, err, elapsed, seqs_per_sec


def epoch_sampler(indices: np.ndarray, frac: float, rng: np.random.Generator) -> SubsetRandomSampler:
    """
    Randomly selects `frac` fraction of `indices` without replacement.
    Called once per epoch so each epoch sees a fresh random subset.
    Over 10 epochs with frac=0.1, the full dataset is covered approximately once.
    """
    n      = max(1, int(len(indices) * frac))
    chosen = rng.choice(indices, size=n, replace=False)
    return SubsetRandomSampler(chosen.tolist())


"""
Main section, calling functions
"""

def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ADDED: compute and display metadata dimensionality
    meta_dim = get_onehot_dim(META_CATEGORIES)
    print(f"device: {device}  |  model_size: {MODEL_SIZE}  |  batch: {BATCH_SIZE}  |  subset: {SUBSET_FRAC*100:.0f}%")
    print(f"metadata one-hot dim: {meta_dim}")
    print(f"regression mode: scores clipped to [0, {SCORE_CLIP_MAX}] then scaled to [0, 1]")
    print(f"WMSE alpha: {WMSE_ALPHA} (weight at y=0: {1+WMSE_ALPHA:.1f}x, at y=1: 1.0x)")

    # Dataset location
    specs = [
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_cereus_processed.csv",              "b_cereus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/s_enterica_processed.csv",            "s_enterica"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/synechococcus_processed.csv",         "synechococcus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_subtilis_processed.csv",            "b_subtilis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/p_putida_processed.csv",              "p_putida"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/y_pseudotuberculosis_processed.csv",  "y_pseudotuberculosis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/e_coli_processed.csv",                "e_coli"),
    ]

    # Load dataset
    # CHANGED: keep_meta=True so metadata is available for one-hot encoding
    ds = BacterialScoresDataset(
        specs=specs, seq_col="Seq", target_col="Score",
        keep_meta=True, dropna=True, shuffle=True, seed=SEED,
    )
    
    print(f"total sequences: {len(ds):,}")

    # CHANGED: Three-way split: test -> val -> train
    # Test set is split off first and NEVER used during training or tuning.
    # This prevents any data leakage into final evaluation.
    n_test  = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val   = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    # Use a fixed generator so splits are reproducible across runs
    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=split_gen
    )

    # IMPORTANT: use positional indices 0..n_train-1, NOT train_ds.indices.
    train_positions = np.arange(n_train)
    print(f"train: {n_train:,}  val: {n_val:,}  test: {n_test:,}  "
          f"(~{int(n_train*SUBSET_FRAC):,} used per epoch)")

    # Initialise RiNALMo embedder
    embedder = RinalmoBatchEmbedder(
        weights_path=WEIGHTS_PATH,
        device=device,
        model_size=MODEL_SIZE,
        use_amp=USE_AMP,
        freeze=True,
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=cpu_collate, drop_last=False,
    )

    # ADDED: Test DataLoader (only used after training completes)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=cpu_collate, drop_last=False,
    )

    # Check embedding dimension
    sample_seqs = [ds[i][0] for i in range(4)]  # 4 raw sequences
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)       # [4, L, D]
    d_model = sample_emb.shape[2]
    L       = sample_emb.shape[1]
    print(f"d_model: {d_model}  L (sample): {L}")
    del sample_emb

    # CHANGED: Initialise regression model (not classifier)
    model   = SimpleCNNRegressor(d_model=d_model, meta_dim=meta_dim).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_epochs  = TOTAL_EPOCHS if RESUME_FROM else EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5) 
    scaler    = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    # Optionally starting from saved checkpoint
    start_epoch   = 1
    best_val_loss = float("inf")

    if RESUME_FROM:
        print(f"resuming from checkpoint: {RESUME_FROM}")
        ckpt_loaded = torch.load(RESUME_FROM, map_location=device)
        # ADDED: verify checkpoint belongs to this model type
        ckpt_type = ckpt_loaded.get("model_type", None)
        if ckpt_type is not None and ckpt_type != "regress":
            raise ValueError(
                f"Checkpoint model_type is '{ckpt_type}', expected 'regress'. "
                f"Are you loading classifier weights by mistake?"
            )
        model.load_state_dict(ckpt_loaded["model"])
        opt.load_state_dict(ckpt_loaded["opt"])
        scheduler.load_state_dict(ckpt_loaded["scheduler"])
        start_epoch   = ckpt_loaded["epoch"] + 1
        best_val_loss = ckpt_loaded.get("val_loss", float("inf"))
        print(f"  resumed at epoch {start_epoch}  "
              f"(best val_loss so far: {best_val_loss:.4f})")
        # Advance RNG past completed epochs to avoid repeating subsets
        rng = np.random.default_rng(SEED + start_epoch)
    else:
        rng = np.random.default_rng(SEED)

    end_epoch = (TOTAL_EPOCHS if RESUME_FROM else EPOCHS) + 1

    # ADDED: accumulate per-epoch metrics for JSON logging and plotting
    metrics_history = []

    # Training loop
    for ep in range(start_epoch, end_epoch):

        # Fresh random subset of training data for this epoch
        sampler      = epoch_sampler(train_positions, SUBSET_FRAC, rng)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False,
            collate_fn=cpu_collate, drop_last=True,
        )

        # CHANGED: regression metrics instead of classification accuracy
        tr_loss, tr_r, tr_rmse, tr_secs, tr_sps = train_one_epoch(
            model, train_loader, embedder, opt, scaler, device, USE_AMP
        )
        va_loss, va_r, va_rmse, va_secs, va_sps = eval_one_epoch(
            model, val_loader, embedder, device
        )
        scheduler.step()

        epoch_secs = tr_secs + va_secs
        full_tr_est        = tr_secs / max(SUBSET_FRAC, 1e-6)
        full_epoch_est_hrs = (full_tr_est + va_secs) / 3600

        current_lr = scheduler.get_last_lr()[0]
        # CHANGED: print regression metrics instead of accuracy
        print(
            f"epoch {ep:02d} | "
            f"train wmse={tr_loss:.4f} r={tr_r:.4f} rmse={tr_rmse:.4f} | "
            f"val wmse={va_loss:.4f} r={va_r:.4f} rmse={va_rmse:.4f} | "
            f"lr={current_lr:.2e}"
        )
        print(
            f"         timing | "
            f"train {tr_secs/60:.1f}min ({tr_sps:.0f} seq/s) | "
            f"val {va_secs/60:.1f}min ({va_sps:.0f} seq/s) | "
            f"epoch total {epoch_secs/60:.1f}min"
        )
        print(
            f"         projection | "
            f"full-dataset epoch ~{full_epoch_est_hrs:.2f}hrs | "
            f"{EPOCHS} epochs ~{full_epoch_est_hrs*EPOCHS:.1f}hrs total "
            f"-> recommend --time=0-{int(full_epoch_est_hrs*EPOCHS*1.3)+1}:0:00"  # 30% buffer
        )

        # ADDED: log metrics for this epoch to history for later plotting
        epoch_metrics = {
            "epoch": ep,
            "train_wmse": tr_loss,
            "train_r": tr_r,
            "train_rmse": tr_rmse,
            "val_wmse": va_loss,
            "val_r": va_r,
            "val_rmse": va_rmse,
            "lr": current_lr,
            "train_secs": tr_secs,
            "val_secs": va_secs,
            "train_seqs_per_sec": tr_sps,
            "val_seqs_per_sec": va_sps,
        }
        # Replace NaN with None for JSON serialisation
        epoch_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in epoch_metrics.items()}
        metrics_history.append(epoch_metrics)

        # Write metrics JSON after every epoch (overwrites) so it's
        # available even if the job is killed mid-training
        metrics_path = f"{CHECKPOINT_DIR}/metrics_regress.json"
        with open(metrics_path, "w") as f:
            json.dump({"epochs": metrics_history, "test": None}, f, indent=2)

        # Each epoch overwrites 'latest.pt' so disk usage stays small
        # ADDED: "model_type" tag prevents accidentally loading classifier
        # weights into the regressor (or vice versa). Checked on resume.
        ckpt = {
            "model_type": "regress",        # ADDED: safety tag
            "epoch":    ep,
            "model":    model.state_dict(),
            "opt":      opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": va_loss,
            "val_r":    va_r,
            "val_rmse": va_rmse,
        }
        torch.save(ckpt, f"{CHECKPOINT_DIR}/latest.pt")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(ckpt, f"{CHECKPOINT_DIR}/best.pt")
            print(f"  -> new best val_wmse={best_val_loss:.4f} r={va_r:.4f} rmse={va_rmse:.4f}, saved best.pt")

    # ADDED: Final evaluation on held-out test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    # Load best model for test evaluation
    best_ckpt = torch.load(f"{CHECKPOINT_DIR}/best.pt", map_location=device)
    # ADDED: verify checkpoint belongs to this model type
    if best_ckpt.get("model_type", None) not in (None, "regress"):
        raise ValueError("best.pt model_type mismatch — wrong checkpoint directory?")
    model.load_state_dict(best_ckpt["model"])
    te_loss, te_r, te_rmse, te_secs, te_sps = eval_one_epoch(
        model, test_loader, embedder, device
    )
    print(f"test wmse={te_loss:.4f}  r={te_r:.4f}  rmse={te_rmse:.4f}")
    print(f"test eval time: {te_secs/60:.1f}min ({te_sps:.0f} seq/s)")
    print("="*60)

    # ADDED: append test results to metrics JSON
    test_metrics = {
        "test_wmse": te_loss,
        "test_r": te_r,
        "test_rmse": te_rmse,
    }
    test_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in test_metrics.items()}
    final_output = {"epochs": metrics_history, "test": test_metrics}
    metrics_path = f"{CHECKPOINT_DIR}/metrics_regress.json"
    with open(metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nTraining complete.")
    print(f"Best val WMSE: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
