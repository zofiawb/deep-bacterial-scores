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

CHANGES from train_cnn_rinalmo_classify_normmeta.py:
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
    4. METADATA PREDICTORS: One-hot encoded experimental conditions are
       still concatenated with CNN features. Even though scores are now
       normalised, metadata may still carry useful information about
       experimental noise characteristics or biological context.

Pipeline overview:
    1. Load processed CSV files (one per bacterial species) containing
       RNA sequences and their experimentally-derived structure scores
    2. Per-method reactivity normalisation (clip, outlier removal,
       effective-max scaling)
    3. Bin normalised scores into 3 discrete classes using fixed thresholds
    4. One-hot encode experimental metadata and concatenate with
       RiNALMo embeddings
    5. Embed each RNA sequence using RiNALMo (a pre-trained RNA language
       model) as a frozen feature extractor 
    6. Train a lightweight CNN classifier on the embeddings + metadata,
       using WeightedRandomSampler for class balance
    7. Save checkpoints every epoch for resumability on HPC (SPARTAN)
    8. Evaluate on held-out test set after training completes

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
CHECKPOINT_DIR = "checkpoints_classify_normmeta_v2"   # separate directory for this experiment

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

# CHANGED: Normalisation is now per-Method only (not per Method+Reagent+Condition)
# Methods to include (RL-seq is excluded in the normalisation step)
VALID_METHODS = [
    "Cotranscriptional_SHAPE-seq", "DMS-MaPseq", "DMS-seq",
    "Lead-seq", "RL-Seq", "SHAPE-MaP", "SHAPE-seq",
]

# Top percentage of values used to compute the effective maximum
EFFECTIVE_MAX_TOP_PCT = 0.08

# Resume-from-checkpoint settings
RESUME_FROM  = "checkpoints_classify_normmeta_v2/latest.pt"             # can be None or "checkpoints_classify_normmeta_v2/latest.pt"
TOTAL_EPOCHS = 40                # total epochs INCLUDING already-completed ones

# Define the categorical metadata columns and their known categories
# for one-hot encoding. These must match the values present in the processed CSVs.
# Unknown/missing values get a dedicated "unknown" column.
# CHANGED: Updated Method list to match VALID_METHODS minus RL-Seq (which is removed),
# and added DMS-MaPseq. icSHAPE removed as it's not in the valid methods list.
META_CATEGORIES = {
    "Method": [
        "SHAPE-seq", "SHAPE-MaP", "DMS-seq", "DMS-MaPseq",
        "Cotranscriptional_SHAPE-seq", "Lead-seq",
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
    Converts categorical metadata into a single one-hot vector.

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
    Returns the total dimensionality of the one-hot metadata vector.
    Each field contributes len(known_categories) + 1 dimensions.
    """
    return sum(len(cats) + 1 for cats in categories.values())


# CHANGED: New per-method reactivity normalisation (replaces z-score)

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
        # Assign class labels based on normalised score
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

        # ADDED: Compute per-sample weights for WeightedRandomSampler.
        # Each sample gets weight = 1 / (number of samples in its class).
        # This means that when sampling, each CLASS has equal total weight,
        # so the sampler draws roughly equal numbers from each class per epoch.
        # This is equivalent to oversampling the minority classes.
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        # Avoid division by zero for empty classes
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        self.sample_weights = torch.tensor(
            [class_weights[l] for l in labels], dtype=torch.float64
        )
        print(f"WeightedRandomSampler: class weights = {class_weights.tolist()}")

        # Shuffle with fixed seed for reproducibility across runs
        if shuffle:
            # We need to shuffle both the df and the sample_weights in the same order
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

        CHANGED: Uses pre-computed class labels from the normalised scores
        with fixed thresholds: < 0.3 = structured, 0.3-0.7 = intermediate,
        > 0.7 = unstructured.
        """
        row   = self.df.iloc[idx]
        seq   = str(row[self.seq_col])

        # Use pre-computed class label
        label = int(row["_class_label"])
        # dtype=torch.long (int64) is required by CrossEntropyLoss
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
    Assembles raw sequences, class labels, and metadata from individual samples.

    Returns (seqs, labels, onehot_meta) — no GPU operations here.
    GPU embedding happens afterwards in the main process via embed_batch().

    Parameters:
    batch : list of (str, torch.Tensor, dict)
        Individual samples from BacterialScoresDataset.__getitem__

    Returns:
    (List[str], torch.Tensor, torch.Tensor)
        Sequences as strings, labels as CPU tensor, one-hot metadata [B, meta_dim].
    """
    seqs, ys, metas = zip(*batch)
    y = torch.stack(ys, dim=0)  # [B] int64 class labels on CPU
    # Build one-hot metadata vectors for each sample in batch
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

    Parameters:
    seqs : List[str]
        RNA sequences from cpu_collate.
    y : torch.Tensor
        Class labels from cpu_collate (CPU tensor).
    onehot : torch.Tensor
        One-hot encoded metadata [B, meta_dim] from cpu_collate (CPU tensor).
    embedder : RinalmoBatchEmbedder
        The frozen RiNALMo model.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor)
        X: embedded sequences [B, D, L] on GPU
        y: class labels [B] on GPU
        onehot: metadata [B, meta_dim] on GPU
    """
    emb = embedder(seqs)                        # [B, L, D] on GPU
    X   = emb.transpose(1, 2).contiguous()      # [B, D, L] — Conv1d format
    y   = y.to(X.device, non_blocking=True)     # move labels to GPU
    # Move metadata to GPU
    onehot = onehot.to(X.device, non_blocking=True)
    return X, y, onehot


# CNN classifier with metadata
class SimpleCNNClassifier(nn.Module):
    """
    3-layer 1D CNN that reads per-position embeddings and predicts
    one of 3 structural classes (structured / intermediate / unstructured).

    After global average pooling over the sequence dimension,
    the one-hot metadata vector is concatenated with the CNN features
    before the final linear layer. This lets the model learn how
    experimental conditions (method, reagent, etc.) affect score
    interpretation.

    Architecture:
      [B, D, L]
        -> Conv1d(D->256, k=7) + ReLU   # detects 7-nt local patterns
        -> Conv1d(256->128, k=5) + ReLU  # combines nearby patterns
        -> Conv1d(128->64,  k=3) + ReLU  # refines
        -> AdaptiveAvgPool1d(1)           # collapses variable seq len -> fixed size
        -> Flatten -> [B, 64]
        -> Concat with one-hot metadata -> [B, 64 + meta_dim]
        -> Linear(64 + meta_dim -> 3)     # 3 output logits, one per class

    Output: raw logits [B, 3]. CrossEntropyLoss applies softmax internally,
    so no softmax here during training.

    Parameters:
    d_model : int
        Embedding dimension from RiNALMo (480 for micro, 1280 for giga).
    meta_dim : int
        Dimensionality of the one-hot metadata vector (from get_onehot_dim).
    num_classes : int
        Number of output classes (default 3).
    """

    def __init__(self, d_model: int, meta_dim: int, num_classes: int = 3):
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
        # Linear layer input is 64 (CNN features) + meta_dim (one-hot metadata)
        self.head = nn.Linear(64 + meta_dim, num_classes)  # [B, 3]

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequence embeddings and metadata.

        Parameters:
        x : torch.Tensor
            RiNALMo embeddings [B, D, L] in Conv1d format.
        meta : torch.Tensor
            One-hot encoded metadata [B, meta_dim].

        Returns:
        torch.Tensor
            Raw logits [B, num_classes].
        """
        h = self.pool(self.feat(x))       # [B, 64]
        h = torch.cat([h, meta], dim=1)   # [B, 64 + meta_dim]
        return self.head(h)               # [B, num_classes]


# Metrics

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of sequences whose predicted class (argmax of logits) matches
    the true class label. Range: 0.0 (all wrong) to 1.0 (all correct).
    A random 3-class classifier would score ~0.33, so anything above that
    means the model has learned something. However, this needs to be checked 
    across the different classes. 

    Parameters:
    logits : torch.Tensor
        Raw model output [B, num_classes].
    labels : torch.Tensor
        True class labels [B].
    
    """
    preds = logits.argmax(dim=1)  # [B] class with highest logit
    return float((preds == labels).float().mean().cpu())

def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                       num_classes: int = 3) -> List[float]:
    """
    Per-class accuracy: for each class, what fraction of samples truly
    belonging to that class were correctly predicted.

    CHANGED: Class names updated for new binning scheme:
      0 = structured, 1 = intermediate, 2 = unstructured

    Parameters:
    logits : torch.Tensor
        Raw model output [B, num_classes].
    labels : torch.Tensor
        True class labels [B].
    num_classes : int
        Number of classes (default 3).

    Returns:
    List[float]
        Accuracy for each class. NaN if class is absent in the batch.
    """
    preds = logits.argmax(dim=1)
    accs  = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            accs.append(float("nan"))  # class absent in this batch
        else:
            accs.append(float((preds[mask] == c).float().mean().cpu()))
    return accs



# Train & eval

def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp) -> Tuple[float, float, List[float], float, float]:
    """
    Train the CNN classifier for one epoch.
    For each batch:
        1. cpu_collate returns raw sequences + labels + one-hot metadata
        2. embed_batch runs RiNALMo on GPU (main process) to get embeddings
        3. Forward pass through CNN with metadata to logits
        4. Compute CrossEntropyLoss
        5. Backward pass to update weights

    Parameters:
    model : SimpleCNNClassifier
        The CNN model being trained.
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
    (avg_loss, accuracy, per_class_accuracies, elapsed_seconds, sequences_per_second)
    """

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, onehot_cpu in loader:
        # GPU embedding + metadata transfer
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


@torch.no_grad() # No gradient calculation as this is a validation function
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
    # Get the dataset indices that belong to the training split
    train_indices = train_ds.indices
    # Extract the corresponding weights
    train_weights = full_sample_weights[train_indices]
    # Number of samples to draw this epoch
    n_draw = max(1, int(len(train_indices) * frac))
    return WeightedRandomSampler(
        weights=train_weights,
        num_samples=n_draw,
        replacement=True,  # required for oversampling minority classes
    )


"""
Main section, calling functions
"""

def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_dim = get_onehot_dim(META_CATEGORIES)
    print(f"device: {device}  |  model_size: {MODEL_SIZE}  |  batch: {BATCH_SIZE}  |  subset: {SUBSET_FRAC*100:.0f}%")
    print(f"metadata one-hot dim: {meta_dim}")
    print(f"EXPERIMENT: per-method reactivity normalisation + metadata predictors + weighted sampling")
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
    # Test set is split off first and NEVER used during training or tuning.
    n_test  = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val   = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    # Use a fixed generator so splits are reproducible across runs
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
    sample_seqs = [ds[i][0] for i in range(4)]  # 4 raw sequences
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)       # [4, L, D]
    d_model = sample_emb.shape[2]
    L       = sample_emb.shape[1]
    print(f"d_model: {d_model}  L (sample): {L}")
    del sample_emb

    # Initialise model with metadata dimension
    model   = SimpleCNNClassifier(d_model=d_model, meta_dim=meta_dim, num_classes=NUM_CLASSES).to(device)
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
        if ckpt_type is not None and ckpt_type != "classify_normmeta_v2":
            raise ValueError(
                f"Checkpoint model_type is '{ckpt_type}', expected 'classify_normmeta_v2'. "
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

    # Accumulate per-epoch metrics for JSON logging and plotting
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
        # CHANGED: Per-class accuracy labels updated for new binning scheme
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

        # Log metrics for this epoch
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

        # Write metrics JSON after every epoch
        metrics_path = f"{CHECKPOINT_DIR}/metrics_classify.json"
        with open(metrics_path, "w") as f:
            json.dump({"epochs": metrics_history, "test": None}, f, indent=2)

        # Save checkpoints
        ckpt = {
            "model_type": "classify_normmeta_v2",  # safety tag
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
    if best_ckpt.get("model_type", None) not in (None, "classify_normmeta_v2"):
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
