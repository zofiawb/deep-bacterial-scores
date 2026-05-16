from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
# IF forking the workers starts creating issues, try changing the start method to spawn (safer on CUDA)
#import torch.multiprocessing as mp
#mp.set_start_method("spawn")

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet

"""
Trains a 1D Convolutional Neural Network (CNN) on top of frozen RiNALMo
embeddings to classify per-window RNA structure scores into three structural
classes: Unstructured, Intermediate, and Structured.

CHANGES from train_cnn_rinalmo_classify_normonly.py:
    1. PER-METHOD REACTIVITY NORMALISATION: Replaces z-score normalisation
       with a reactivity-standard normalisation pipeline applied separately
       to each probing Method. The pipeline:
         a) Remove RL-seq scores (hydroxyl radical footprinting, different
            chemistry from other probing methods).
         b) Remove control samples with no reagent (NaN Method AND NaN Reagent).
         c) Clip negative reactivity scores to 0 (negatives are artefacts).
         d) Remove outliers above Q3 + 1.5*IQR (extreme values that distort
            the effective maximum calculation).
         e) Compute effective maximum = mean of the top 8% of values within
            each Method group (robust estimator of the maximum).
         f) Divide all values by the effective maximum so scores are on a
            0-to-~1 scale within each Method.
       This normalisation ensures that scores from different probing
       technologies (DMS-seq, SHAPE-MaP, etc.) are on comparable scales
       without requiring z-scoring, which can distort the biological
       interpretation of the bins.
    2. CLASSIFICATION BINS: Fixed thresholds on normalised scores:
         0 = "Structured" (score < 0.3) — low reactivity, base-paired
         1 = "Intermediate" (0.3 <= score <= 0.7) — moderate reactivity
         2 = "Unstructured" (score > 0.7) — high reactivity, flexible/unpaired
       These thresholds are biologically interpretable because the per-method
       normalisation puts all scores on a 0-to-1 scale first.
    3. WEIGHTED RANDOM SAMPLING: Uses WeightedRandomSampler to oversample
       minority classes so each training batch contains roughly equal
       numbers of each class. This replaces the epoch_sampler and addresses
       class imbalance (the majority class typically dominates training,
       causing the model to ignore minority classes).
    4. NO METADATA PREDICTORS: Unlike the normmeta_v2 version, this script
       does NOT feed one-hot encoded experimental conditions to the model.
       The CNN receives only sequence embeddings. This tests whether the
       normalisation alone is sufficient to handle cross-method variation,
       or whether the model still benefits from knowing the experimental
       context explicitly.

This is the NORMALISED + NO METADATA version (v2) for the ablation study.
Compare with:
    - train_cnn_rinalmo_classify.py               (unnormalised + metadata, baseline)
    - train_cnn_rinalmo_classify_normmeta_v2.py    (normalised + metadata, v2)

Pipeline overview:
    1. Load processed CSV files (one per bacterial species) containing
       RNA sequences and their experimentally-derived structure scores
    2. Per-method reactivity normalisation (clip, outlier removal,
       effective-max scaling)
    3. Bin normalised scores into 3 discrete classes using fixed thresholds
    4. Embed each RNA sequence using RiNALMo (a pre-trained RNA language
       model) as a frozen feature extractor 
    5. Train a lightweight CNN classifier on the embeddings ONLY,
       using WeightedRandomSampler for class balance
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
    - Per-class accuracy is tracked alongside overall accuracy to detect 
      class imbalance issues (e.g. model only predicting the majority class).

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

# Split fractions
TEST_FRAC    = 0.10              # fraction of data held out for final test (never seen during training)
VAL_FRAC     = 0.10              # fraction of REMAINING data held out for validation

SEED         = 0                 # random seed for reproducibility
NUM_WORKERS  = 4                 # CPU workers for DataLoader prefetching
USE_AMP      = True              # automatic mixed precision (float16) on GPU
CHECKPOINT_DIR = "checkpoints_classify_normonly_v2"   # separate directory for this experiment

# CHANGED: Fixed classification bins on normalised (0-to-1) scores.
# After per-method normalisation, scores are on a 0-to-~1 scale, so
# these thresholds have direct biological meaning:
#   < 0.3  = low reactivity  -> structured / base-paired
#   0.3-0.7 = moderate       -> intermediate
#   > 0.7  = high reactivity -> unstructured / flexible
BIN_LOW  = 0.3
BIN_HIGH = 0.7
NUM_CLASSES = 3
CLASS_NAMES = {
    0: "Structured: < 0.3",
    1: "Intermediate: 0.3 to 0.7",
    2: "Unstructured: > 0.7",
}

# Methods to include (RL-seq is excluded in the normalisation step)
VALID_METHODS = [
    "Cotranscriptional_SHAPE-seq", "DMS-MaPseq", "DMS-seq",
    "Lead-seq", "RL-Seq", "SHAPE-MaP", "SHAPE-seq",
]

# Top percentage of values used to compute the effective maximum
EFFECTIVE_MAX_TOP_PCT = 0.08

# Resume-from-checkpoint settings
RESUME_FROM  = "checkpoints_classify_normonly_v2/latest.pt"              # can be None or "checkpoints_classify_normonly_v2/latest.pt"
TOTAL_EPOCHS = 40                # total epochs INCLUDING already-completed ones


# CHANGED: Per-method reactivity normalisation (replaces z-score)

def normalise_scores_by_method(df: pd.DataFrame, score_col: str,
                               method_col: str = "Method",
                               reagent_col: str = "Reagent",
                               top_pct: float = EFFECTIVE_MAX_TOP_PCT) -> pd.DataFrame:
    """
    Per-method reactivity normalisation pipeline.

    This function normalises raw structure-probing scores so that values
    from different experimental methods (DMS-seq, SHAPE-MaP, Lead-seq, etc.)
    are on a comparable 0-to-~1 scale. Each Method is normalised independently
    because different probing chemistries produce scores on fundamentally
    different scales.

    The pipeline for each Method group:
        1. Remove RL-seq scores entirely (hydroxyl radical footprinting uses
           different chemistry and is not comparable to other methods).
        2. Remove control samples: rows where BOTH Method and Reagent are NaN
           indicate measurements taken without any chemical reagent, i.e.
           negative controls that don't reflect true reactivity.
        3. Clip negative values to 0: negative reactivity scores are
           experimental artefacts (reactivity can't physically be negative).
        4. Remove upper outliers: compute Q3 + 1.5*IQR for each Method and
           discard values above this threshold. These extreme outliers would
           inflate the effective maximum and compress the useful score range.
        5. Compute effective maximum: the mean of the top 8% of remaining
           values within each Method. This is more robust than the raw max
           (which could be a single noisy measurement).
        6. Divide all values by the effective maximum: this puts scores on
           a 0-to-~1 scale where ~1 means "as reactive as the most reactive
           nucleotides measured by this method".

    Parameters:
    df : pd.DataFrame
        Full dataset with raw scores and metadata columns.
    score_col : str
        Name of the column containing raw structure scores.
    method_col : str
        Column identifying the probing method (default "Method").
    reagent_col : str
        Column identifying the reagent (default "Reagent").
    top_pct : float
        Fraction of top values used to compute effective maximum (default 0.08).

    Returns:
    pd.DataFrame
        Copy of df with an additional "Score_normalised" column and
        removed/filtered rows. Original "Score" column is preserved
        for reference.
    """
    df = df.copy()
    n_initial = len(df)

    print(f"\n{'='*60}")
    print(f"PER-METHOD REACTIVITY NORMALISATION")
    print(f"{'='*60}")

    # Step 1: Remove RL-seq scores
    # RL-seq uses hydroxyl radical footprinting which probes the sugar-phosphate
    # backbone, while other methods probe base accessibility. The two measurement
    # types are not directly comparable.
    rl_mask = df[method_col].str.strip().str.lower() == "rl-seq"
    n_rl = rl_mask.sum()
    df = df[~rl_mask].reset_index(drop=True)
    print(f"Step 1: Removed {n_rl:,} RL-seq scores ({100*n_rl/max(n_initial,1):.2f}%)")

    # Step 2: Remove control samples (NaN Method AND NaN Reagent)
    # These are measurements taken without any chemical reagent — they serve as
    # negative controls in the original experiments but don't represent real
    # structure-probing reactivity.
    control_mask = df[method_col].isna() & df[reagent_col].isna()
    n_controls = control_mask.sum()
    df = df[~control_mask].reset_index(drop=True)
    print(f"Step 2: Removed {n_controls:,} control samples (NaN method + NaN reagent)")

    # Step 3: Clip negative scores to 0
    # Negative reactivity values are artefacts of background subtraction in the
    # experimental pipeline. True chemical reactivity cannot be negative.
    n_negative = (df[score_col] < 0).sum()
    df[score_col] = df[score_col].clip(lower=0)
    print(f"Step 3: Clipped {n_negative:,} negative scores to 0")

    # Steps 4-6: Per-method outlier removal + effective-max normalisation
    # Each Method is processed independently because different chemistries
    # produce values on different scales.
    normalised_dfs = []
    methods_in_data = df[method_col].dropna().unique()
    print(f"\nProcessing {len(methods_in_data)} methods:")

    for method in sorted(methods_in_data):
        method_mask = df[method_col] == method
        mdf = df[method_mask].copy()
        n_method_initial = len(mdf)
        scores = mdf[score_col].values

        # Step 4: IQR outlier removal
        # The interquartile range (IQR) is Q3 - Q1. Values above Q3 + 1.5*IQR
        # are considered extreme outliers. Removing these prevents a handful of
        # very high values from inflating the effective maximum.
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        outlier_mask = scores > upper_fence
        n_outliers = outlier_mask.sum()
        pct_outliers = 100 * n_outliers / max(len(scores), 1)

        # Remove outliers
        mdf = mdf[~outlier_mask].reset_index(drop=True)
        scores_clean = mdf[score_col].values

        # Step 5: Compute effective maximum (mean of top 8%)
        # Using the mean of the top fraction is more robust than the absolute
        # maximum, which could be a single noisy measurement. The top 8%
        # represents the "highly reactive" tail of the distribution.
        n_top = max(1, int(len(scores_clean) * top_pct))
        # np.partition is O(n) vs O(n log n) for full sort — faster for large arrays
        top_values = np.partition(scores_clean, -n_top)[-n_top:]
        effective_max = np.mean(top_values)

        # Step 6: Divide by effective maximum
        # This rescales so that ~1.0 means "as reactive as the top 8% of
        # nucleotides measured by this method". Values slightly above 1.0 are
        # possible and expected for individual high-reactivity nucleotides.
        if effective_max > 0:
            mdf["Score_normalised"] = mdf[score_col] / effective_max
        else:
            # Edge case: all scores are 0 (degenerate group)
            mdf["Score_normalised"] = 0.0

        normalised_dfs.append(mdf)

        print(f"  {method}:")
        print(f"    n={n_method_initial:,} -> {len(mdf):,} after outlier removal "
              f"({n_outliers:,} removed, {pct_outliers:.1f}%)")
        print(f"    IQR fence: Q3={q3:.4f} + 1.5*IQR={1.5*iqr:.4f} = {upper_fence:.4f}")
        print(f"    effective max (mean of top {top_pct*100:.0f}%): {effective_max:.4f}")
        if len(mdf) > 0:
            ns = mdf["Score_normalised"]
            print(f"    normalised range: [{ns.min():.4f}, {ns.max():.4f}], "
                  f"median={ns.median():.4f}")

    # Combine all normalised methods back together
    df = pd.concat(normalised_dfs, ignore_index=True)

    # Summary statistics
    n_final = len(df)
    ns = df["Score_normalised"]
    print(f"\nOverall normalised score distribution:")
    print(f"  total samples: {n_initial:,} -> {n_final:,} "
          f"({n_initial - n_final:,} removed, {100*(n_initial-n_final)/max(n_initial,1):.1f}%)")
    print(f"  mean:   {ns.mean():.4f}")
    print(f"  std:    {ns.std():.4f}")
    print(f"  min:    {ns.min():.4f}")
    print(f"  25%:    {ns.quantile(0.25):.4f}")
    print(f"  50%:    {ns.quantile(0.50):.4f}")
    print(f"  75%:    {ns.quantile(0.75):.4f}")
    print(f"  max:    {ns.max():.4f}")
    print(f"{'='*60}\n")

    return df


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
    bacterial species and serves (sequence, class_label, metadata) tuples.

    CHANGED: Scores are normalised using the per-method reactivity
    normalisation pipeline (clip negatives, IQR outlier removal,
    effective-max scaling) before binning. Bin thresholds are fixed at
    0.3 and 0.7 on the normalised 0-to-1 scale.

    Also computes per-sample weights for WeightedRandomSampler to
    address class imbalance during training.

    Parameters:
    specs : List[CSVDatasetSpec]
        List of (path, species) pairs for each species CSV file.
    seq_col : str
        Name of the column containing RNA sequences.
    target_col : str
        Name of the column containing raw structure scores.
    keep_meta : bool
        Whether to return metadata columns (species, method, etc.).
        Kept True so metadata is available for logging even though the
        model doesn't use it.
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
        keep_meta: bool = True,
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

        # CHANGED: Apply per-method reactivity normalisation
        big = normalise_scores_by_method(
            big, score_col=target_col,
            method_col="Method", reagent_col="Reagent",
            top_pct=EFFECTIVE_MAX_TOP_PCT,
        )

        # CHANGED: Bin using fixed thresholds on normalised scores
        labels = np.where(
            big["Score_normalised"] < BIN_LOW, 0,
            np.where(big["Score_normalised"] > BIN_HIGH, 2, 1)
        )
        big["_class_label"] = labels

        # Count and print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        print(f"Class distribution after normalisation + binning:")
        for u, c in zip(unique, counts):
            print(f"  class {u} ({CLASS_NAMES.get(u, '?')}): {c:,} ({100*c/total:.1f}%)")

        # ADDED: Compute per-sample weights for WeightedRandomSampler
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        self.sample_weights = torch.tensor(
            [class_weights[l] for l in labels], dtype=torch.float64
        )
        print(f"WeightedRandomSampler: class weights = {class_weights.tolist()}")

        # Shuffle with fixed seed for reproducibility across runs
        if shuffle:
            perm = np.random.RandomState(seed).permutation(len(big))
            big = big.iloc[perm].reset_index(drop=True)
            self.sample_weights = self.sample_weights[perm]

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
        Returns one sample as (sequence_string, class_label, metadata_dict).

        Uses pre-computed class labels from the normalised scores
        with fixed thresholds: < 0.3 = structured, 0.3-0.7 = intermediate,
        > 0.7 = unstructured.
        """
        row   = self.df.iloc[idx]
        seq   = str(row[self.seq_col])

        # Use pre-computed class label
        label = int(row["_class_label"])
        y = torch.tensor(label, dtype=torch.long)

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
        toks   = self.alphabet.batch_tokenize(seqs)
        tokens = torch.tensor(toks, dtype=torch.int64, device=self.device)
        if self.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = self.model(tokens)
        else:
            out = self.model(tokens)
        return out["representation"]  # [B, L, D]


# Collation and batch embedding:
# CPU based data loading in forked worker processes, and GPU based embedding

def cpu_collate(batch):
    """
    Collate function for DataLoader workers (CPU-only processes).
    Assembles raw sequences and class labels from individual samples.

    CHANGED (normonly): No metadata one-hot encoding. Returns a dummy
    zero tensor for the onehot slot to keep the function signature
    compatible with embed_batch, but it is not used by the model.

    Parameters:
    batch : list of (str, torch.Tensor, dict)
        Individual samples from BacterialScoresDataset.__getitem__

    Returns:
    (List[str], torch.Tensor, torch.Tensor)
        Sequences as strings, labels as CPU tensor, dummy onehot [B, 0].
    """
    seqs, ys, metas = zip(*batch)
    y = torch.stack(ys, dim=0)  # [B] int64 class labels on CPU
    # No metadata — return empty tensor to keep interface consistent
    dummy_onehot = torch.zeros(len(seqs), 0, dtype=torch.float32)
    return list(seqs), y, dummy_onehot


def embed_batch(seqs: List[str], y: torch.Tensor, onehot: torch.Tensor,
                embedder: RinalmoBatchEmbedder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs RiNALMo in the main process (where CUDA is initialised).
    Called inside the training/eval loop, not in the DataLoader.
    The embedding output is transposed from [B, L, D] to [B, D, L] because
    PyTorch's Conv1d expects input as [batch, channels, length].

    Parameters:
    seqs : List[str]
        RNA sequences from cpu_collate.
    y : torch.Tensor
        Class labels from cpu_collate (CPU tensor).
    onehot : torch.Tensor
        Dummy metadata [B, 0] from cpu_collate (CPU tensor).
    embedder : RinalmoBatchEmbedder
        The frozen RiNALMo model.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor)
        X: embedded sequences [B, D, L] on GPU
        y: class labels [B] on GPU
        onehot: dummy metadata [B, 0] on GPU
    """
    emb = embedder(seqs)                        # [B, L, D] on GPU
    X   = emb.transpose(1, 2).contiguous()      # [B, D, L] — Conv1d format
    y   = y.to(X.device, non_blocking=True)     # move labels to GPU
    onehot = onehot.to(X.device, non_blocking=True)
    return X, y, onehot


# CNN classifier WITHOUT metadata (normonly ablation)
class SimpleCNNClassifier(nn.Module):
    """
    3-layer 1D CNN that reads per-position embeddings and predicts
    one of 3 structural classes (structured / intermediate / unstructured).

    CHANGED (normonly): No metadata concatenation. The model receives
    only sequence embeddings from RiNALMo. This tests whether per-method
    normalisation alone is sufficient, without the model needing to know
    which experimental method produced each sample.

    Architecture:
      [B, D, L]
        -> Conv1d(D->256, k=7) + ReLU   # detects 7-nt local patterns
        -> Conv1d(256->128, k=5) + ReLU  # combines nearby patterns
        -> Conv1d(128->64,  k=3) + ReLU  # refines
        -> AdaptiveAvgPool1d(1)           # collapses variable seq len -> fixed size
        -> Flatten -> [B, 64]
        -> Linear(64 -> 3)               # 3 output logits, one per class

    Output: raw logits [B, 3]. CrossEntropyLoss applies softmax internally,
    so no softmax here during training.

    Parameters:
    d_model : int
        Embedding dimension from RiNALMo (480 for micro, 1280 for giga).
    meta_dim : int
        Ignored in this version (kept for interface compatibility).
    num_classes : int
        Number of output classes (default 3).
    """

    def __init__(self, d_model: int, meta_dim: int = 0, num_classes: int = 3):
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
        # No metadata, so input is just 64 CNN features
        self.head = nn.Linear(64, num_classes)  # [B, 3]

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequence embeddings only.

        meta argument is accepted but ignored to keep the interface
        consistent with the metadata version. This avoids needing to
        change the training loop.

        Parameters:
        x : torch.Tensor
            RiNALMo embeddings [B, D, L] in Conv1d format.
        meta : torch.Tensor
            Ignored (dummy tensor [B, 0]).

        Returns:
        torch.Tensor
            Raw logits [B, num_classes].
        """
        h = self.pool(self.feat(x))       # [B, 64]
        return self.head(h)               # [B, num_classes]


# Metrics

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of sequences whose predicted class (argmax of logits) matches
    the true class label. Range: 0.0 (all wrong) to 1.0 (all correct).
    """
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().cpu())

def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                       num_classes: int = 3) -> List[float]:
    """
    Per-class accuracy: for each class, what fraction of samples truly
    belonging to that class were correctly predicted.

      0 = structured, 1 = intermediate, 2 = unstructured

    Returns:
    List[float]
        Accuracy for each class. NaN if class is absent in the batch.
    """
    preds = logits.argmax(dim=1)
    accs  = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            accs.append(float("nan"))
        else:
            accs.append(float((preds[mask] == c).float().mean().cpu()))
    return accs



# Train & eval

def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp) -> Tuple[float, float, List[float], float, float]:
    """
    Train the CNN classifier for one epoch.

    Returns:
    (avg_loss, accuracy, per_class_accuracies, elapsed_seconds, sequences_per_second)
    """

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, onehot_cpu in loader:
        X, y, onehot = embed_batch(seqs, y_cpu, onehot_cpu, embedder)

        opt.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(X, onehot)
                loss   = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X, onehot)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()

        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
        n_seqs += bs

    elapsed      = time.perf_counter() - t_start
    cat_logits   = torch.cat(all_logits)
    cat_labels   = torch.cat(all_labels)
    n            = cat_labels.shape[0]
    avg_loss     = total_loss / max(n, 1)
    acc          = accuracy(cat_logits, cat_labels)
    cls_accs     = per_class_accuracy(cat_logits, cat_labels, NUM_CLASSES)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, acc, cls_accs, elapsed, seqs_per_sec


@torch.no_grad()
def eval_one_epoch(model, loader, embedder, device) -> Tuple[float, float, List[float], float, float]:
    """
    Evaluates the CNN classifier on the validation or test set.

    Returns:
    (avg_loss, overall_accuracy, per_class_accuracies, elapsed_seconds, sequences_per_second)
    """
    import time
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, onehot_cpu in loader:
        X, y, onehot = embed_batch(seqs, y_cpu, onehot_cpu, embedder)
        logits = model(X, onehot)
        loss   = loss_fn(logits, y)
        bs     = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits)
        all_labels.append(y)
        n_seqs += bs

    elapsed      = time.perf_counter() - t_start
    cat_logits   = torch.cat(all_logits)
    cat_labels   = torch.cat(all_labels)
    n            = cat_labels.shape[0]
    avg_loss     = total_loss / max(n, 1)
    acc          = accuracy(cat_logits, cat_labels)
    cls_accs     = per_class_accuracy(cat_logits, cat_labels, NUM_CLASSES)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, acc, cls_accs, elapsed, seqs_per_sec


# CHANGED: Replaced epoch_sampler with weighted_epoch_sampler

def weighted_epoch_sampler(train_ds, full_sample_weights: torch.Tensor,
                           frac: float, rng: np.random.Generator) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that draws `frac` fraction of the
    training set WITH replacement, where each sample's draw probability
    is proportional to the inverse of its class frequency.

    This achieves two goals at once:
      1. Epoch subsampling: only `frac` of the data is used per epoch
         (same as the old epoch_sampler), keeping each epoch fast.
      2. Class balancing: minority classes are oversampled so each batch
         contains roughly equal numbers of each class, preventing the
         model from ignoring rare classes.

    WeightedRandomSampler draws WITH replacement by default. This means
    minority-class samples will appear multiple times per epoch (that's
    the whole point — oversampling). Majority-class samples may appear
    zero or one times. Over many epochs, all samples are seen.

    Parameters:
    train_ds : Subset
        The training split (from random_split). We need its .indices
        to look up the correct sample weights from the full dataset.
    full_sample_weights : torch.Tensor
        Per-sample weights for the FULL dataset (computed in
        BacterialScoresDataset.__init__).
    frac : float
        Fraction of the training set to draw per epoch.
    rng : np.random.Generator
        Random number generator (unused here — PyTorch handles its own
        RNG for WeightedRandomSampler, but kept for interface consistency).

    Returns:
    WeightedRandomSampler
        Sampler that draws `n_draw` samples with class-balanced probabilities.
    """
    train_indices = train_ds.indices
    train_weights = full_sample_weights[train_indices]
    n_draw = max(1, int(len(train_indices) * frac))
    return WeightedRandomSampler(
        weights=train_weights,
        num_samples=n_draw,
        replacement=True,
    )


"""
Main section, calling functions
"""

def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_dim = 0  # No metadata in this version
    print(f"device: {device}  |  model_size: {MODEL_SIZE}  |  batch: {BATCH_SIZE}  |  subset: {SUBSET_FRAC*100:.0f}%")
    print(f"metadata: NONE (normonly ablation)")
    print(f"EXPERIMENT: per-method reactivity normalisation + NO metadata + weighted sampling")
    print(f"bin thresholds: structured < {BIN_LOW} <= intermediate <= {BIN_HIGH} < unstructured")

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

    # Load dataset (normalisation happens inside the constructor)
    ds = BacterialScoresDataset(
        specs=specs, seq_col="Seq", target_col="Score",
        keep_meta=True, dropna=True, shuffle=True, seed=SEED,
    )
    
    print(f"total sequences (after normalisation filtering): {len(ds):,}")

    # Three-way split: test -> val -> train
    n_test  = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val   = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=split_gen
    )

    print(f"train: {n_train:,}  val: {n_val:,}  test: {n_test:,}  "
          f"(~{int(n_train*SUBSET_FRAC):,} drawn per epoch with weighted sampling)")

    # Initialise RiNALMo embedder
    embedder = RinalmoBatchEmbedder(
        weights_path=WEIGHTS_PATH,
        device=device,
        model_size=MODEL_SIZE,
        use_amp=USE_AMP,
        freeze=True,
    )

    # Validation DataLoader (no weighted sampling — evaluate on true distribution)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=cpu_collate, drop_last=False,
    )

    # Test DataLoader (only used after training completes)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=cpu_collate, drop_last=False,
    )

    # Check embedding dimension
    sample_seqs = [ds[i][0] for i in range(4)]
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)
    d_model = sample_emb.shape[2]
    L       = sample_emb.shape[1]
    print(f"d_model: {d_model}  L (sample): {L}")
    del sample_emb

    # Initialise model WITHOUT metadata
    model   = SimpleCNNClassifier(d_model=d_model, meta_dim=0, num_classes=NUM_CLASSES).to(device)
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
        ckpt_type = ckpt_loaded.get("model_type", None)
        if ckpt_type is not None and ckpt_type != "classify_normonly_v2":
            raise ValueError(
                f"Checkpoint model_type is '{ckpt_type}', expected 'classify_normonly_v2'. "
                f"Are you loading the wrong checkpoint?"
            )
        model.load_state_dict(ckpt_loaded["model"])
        opt.load_state_dict(ckpt_loaded["opt"])
        scheduler.load_state_dict(ckpt_loaded["scheduler"])
        start_epoch   = ckpt_loaded["epoch"] + 1
        best_val_loss = ckpt_loaded.get("val_loss", float("inf"))
        print(f"  resumed at epoch {start_epoch}  "
              f"(best val_loss so far: {best_val_loss:.4f})")
        rng = np.random.default_rng(SEED + start_epoch)
    else:
        rng = np.random.default_rng(SEED)

    end_epoch = (TOTAL_EPOCHS if RESUME_FROM else EPOCHS) + 1

    # Accumulate per-epoch metrics
    metrics_history = []

    # Training loop
    for ep in range(start_epoch, end_epoch):

        # CHANGED: Use WeightedRandomSampler for class-balanced subsampling
        sampler = weighted_epoch_sampler(train_ds, ds.sample_weights, SUBSET_FRAC, rng)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False,
            collate_fn=cpu_collate, drop_last=True,
        )

        tr_loss, tr_acc, tr_cls_accs, tr_secs, tr_sps = train_one_epoch(model, train_loader, embedder, opt, scaler, device, USE_AMP)
        va_loss, va_acc, va_cls_accs, va_secs, va_sps = eval_one_epoch(model, val_loader, embedder, device)
        scheduler.step()

        epoch_secs = tr_secs + va_secs
        full_tr_est        = tr_secs / max(SUBSET_FRAC, 1e-6)
        full_epoch_est_hrs = (full_tr_est + va_secs) / 3600

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {ep:02d} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        cls_str = " | ".join(
            f"{'nan' if np.isnan(a) else f'{a:.3f}'}"
            for a in va_cls_accs
        )
        print(
            f"         per-class val acc "
            f"[structured | inter | unstructured]: {cls_str}"
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
            f"-> recommend --time=0-{int(full_epoch_est_hrs*EPOCHS*1.3)+1}:0:00"
        )

        # Log metrics
        epoch_metrics = {
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_acc_structured": tr_cls_accs[0],
            "train_acc_inter": tr_cls_accs[1],
            "train_acc_unstructured": tr_cls_accs[2],
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_acc_structured": va_cls_accs[0],
            "val_acc_inter": va_cls_accs[1],
            "val_acc_unstructured": va_cls_accs[2],
            "lr": current_lr,
            "train_secs": tr_secs,
            "val_secs": va_secs,
            "train_seqs_per_sec": tr_sps,
            "val_seqs_per_sec": va_sps,
        }
        epoch_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in epoch_metrics.items()}
        metrics_history.append(epoch_metrics)

        metrics_path = f"{CHECKPOINT_DIR}/metrics_classify.json"
        with open(metrics_path, "w") as f:
            json.dump({"epochs": metrics_history, "test": None}, f, indent=2)

        # Save checkpoints
        ckpt = {
            "model_type": "classify_normonly_v2",
            "epoch":    ep,
            "model":    model.state_dict(),
            "opt":      opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": va_loss,
            "val_acc":  va_acc,
        }
        torch.save(ckpt, f"{CHECKPOINT_DIR}/latest.pt")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(ckpt, f"{CHECKPOINT_DIR}/best.pt")
            print(f"  -> new best val_loss={best_val_loss:.4f} acc={va_acc:.4f}, saved best.pt")

    # Final evaluation on held-out test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    best_ckpt = torch.load(f"{CHECKPOINT_DIR}/best.pt", map_location=device)
    if best_ckpt.get("model_type", None) not in (None, "classify_normonly_v2"):
        raise ValueError("best.pt model_type mismatch — wrong checkpoint directory?")
    model.load_state_dict(best_ckpt["model"])
    te_loss, te_acc, te_cls_accs, te_secs, te_sps = eval_one_epoch(
        model, test_loader, embedder, device
    )
    cls_str = " | ".join(
        f"{'nan' if np.isnan(a) else f'{a:.3f}'}"
        for a in te_cls_accs
    )
    print(f"test loss={te_loss:.4f}  acc={te_acc:.4f}")
    print(f"per-class test acc [structured | inter | unstructured]: {cls_str}")
    print(f"test eval time: {te_secs/60:.1f}min ({te_sps:.0f} seq/s)")
    print("="*60)

    # Append test results to metrics JSON
    test_metrics = {
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_acc_structured": te_cls_accs[0],
        "test_acc_inter": te_cls_accs[1],
        "test_acc_unstructured": te_cls_accs[2],
    }
    test_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in test_metrics.items()}
    final_output = {"epochs": metrics_history, "test": test_metrics}
    metrics_path = f"{CHECKPOINT_DIR}/metrics_classify.json"
    with open(metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
