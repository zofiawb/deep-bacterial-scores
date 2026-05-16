from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
# IF forking the workers starts creating issues, try changing the start method to spawn (safer on CUDA)
#import torch.multiprocessing as mp
#mp.set_start_method("spawn")

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet

"""
Trains a 1D Convolutional Neural Network (CNN) on top of frozen RiNALMo
embeddings as a BINARY CLASSIFIER to distinguish structured (score == 0)
from unstructured (score > 0.7) RNA nucleotides.

CHANGES from train_cnn_rinalmo_classify_normmeta_v2.py:
    1. BINARY TASK: The intermediate class (0 < score <= 0.7) is discarded
       entirely. Only two classes remain:
         0 = "Structured"   (normalised score == 0, i.e. zero reactivity)
         1 = "Unstructured" (normalised score > 0.7, high reactivity)
       This removes the ambiguous middle ground and creates a cleaner
       separation between clearly paired and clearly flexible nucleotides.
    2. SUBSAMPLING MAJORITY CLASS: Instead of oversampling the minority
       class (unstructured), the majority class (structured, score == 0)
       is randomly subsampled down to match the minority class size. This
       avoids the duplicate-sample problem that comes with oversampling
       and ensures every training sample is unique.
    3. BINARY CROSS-ENTROPY LOSS: Replaces CrossEntropyLoss with
       BCEWithLogitsLoss. The model outputs a single logit per sample
       (not 2 logits), and the loss function applies sigmoid internally.
       Positive logit -> predicts unstructured; negative -> structured.
    4. EXPANDED METADATA: Now includes Species and Temp (temperature) in
       addition to Method, Reagent, Condition, and Specificity. Species
       is one-hot encoded like the other categoricals. Temp is a continuous
       value (25, 30, 37, 42°C, etc.) that is min-max normalised to [0,1]
       and concatenated as a single scalar feature.
    5. METRICS: Accuracy, precision, recall, F1, and AUROC are computed
       for the binary task. Per-class accuracy is replaced by these more
       informative binary metrics.

Pipeline overview:
    1. Load processed CSV files (one per bacterial species)
    2. Per-method reactivity normalisation (same as v2)
    3. Discard intermediate scores (0 < score <= 0.7)
    4. Subsample structured class to match unstructured class size
    5. One-hot encode metadata (Species, Method, Reagent, Condition,
       Specificity) + normalised Temp scalar
    6. Embed sequences with frozen RiNALMo
    7. Train binary CNN classifier with BCEWithLogitsLoss
    8. Evaluate with accuracy, precision, recall, F1, AUROC

Author: Zofia Witkowski-Blake
Project: Predicting RNA structure scores using CNNs and RiNALMo embeddings
University of Melbourne, MSc Bioinformatics    
"""

#Tuneable constants
MODEL_SIZE   = "micro"           # "micro" | "giga"
WEIGHTS_PATH = "weights/rinalmo_micro_pretrained.pt"
BATCH_SIZE   = 256
SUBSET_FRAC  = 0.1              # fraction of training set used per epoch
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-2

# Split fractions
TEST_FRAC    = 0.10
VAL_FRAC     = 0.10

SEED         = 0
NUM_WORKERS  = 4
USE_AMP      = True
CHECKPOINT_DIR = "checkpoints_binary_v3"

# Binary classification threshold on normalised scores
# Structured: score == 0 (zero reactivity after normalisation)
# Unstructured: score > 0.7 (high reactivity)
# Intermediate (0 < score <= 0.7) is DISCARDED
UNSTRUCTURED_THRESHOLD = 0.7

# Top percentage of values used to compute the effective maximum
EFFECTIVE_MAX_TOP_PCT = 0.08

# Resume-from-checkpoint settings
RESUME_FROM  = "checkpoints_binary_v3/latest.pt"
TOTAL_EPOCHS = 60

# CHANGED: Expanded metadata categories now include Species.
# Temp is handled separately as a continuous feature (not one-hot).
META_CATEGORIES = {
    "Species": [
        "b_cereus", "s_enterica", "synechococcus", "b_subtilis",
        "p_putida", "y_pseudotuberculosis", "e_coli",
    ],
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

# Known temperature values for min-max normalisation.
# Parsed from column names in processing.py: 25, 30, 37, 42, 80, 95°C.
# Min-max normalisation maps these to [0, 1].
TEMP_MIN = 25.0
TEMP_MAX = 95.0


def build_meta_vector(meta: Dict, categories: Dict[str, List[str]],
                      temp_min: float = TEMP_MIN, temp_max: float = TEMP_MAX) -> torch.Tensor:
    """
    Converts metadata into a feature vector combining one-hot encoded
    categoricals and a normalised continuous temperature value.

    CHANGED from build_onehot_vector: Now also handles Species (one-hot)
    and Temp (min-max normalised scalar). The output is a single 1D tensor
    ready for concatenation with CNN features.

    For each categorical field (Species, Method, Reagent, Condition,
    Specificity), creates a sub-vector of length len(known_categories) + 1
    (the +1 is for unknown/missing). Temp is appended as a single float
    in [0, 1] (0.5 if missing, which corresponds to ~60°C — a neutral
    midpoint that avoids biasing toward any extreme).

    Parameters:
    meta : Dict
        Metadata dictionary from __getitem__,
        e.g. {"Species": "e_coli", "Method": "DMS-seq", "Temp": 37, ...}
    categories : Dict[str, List[str]]
        Mapping of field name -> list of known category strings.
    temp_min, temp_max : float
        Range for min-max normalising temperature values.

    Returns:
    torch.Tensor
        1D float32 tensor of length sum(len(cats)+1) + 1 (for Temp).
    """
    parts = []
    for field, cats in categories.items():
        n = len(cats) + 1  # +1 for unknown/missing
        vec = torch.zeros(n, dtype=torch.float32)
        val = meta.get(field, None)
        # Handle pandas NA / NaN for string comparison
        if val is not None and not (isinstance(val, float) and np.isnan(val)) and val in cats:
            vec[cats.index(val)] = 1.0
        else:
            vec[-1] = 1.0  # mark as unknown/missing
        parts.append(vec)

    # Temp: continuous, min-max normalised to [0, 1]
    temp_val = meta.get("Temp", None)
    if temp_val is not None and not (isinstance(temp_val, float) and np.isnan(temp_val)):
        try:
            t = float(temp_val)
            t_norm = (t - temp_min) / max(temp_max - temp_min, 1e-6)
            t_norm = max(0.0, min(1.0, t_norm))  # clamp to [0, 1]
        except (ValueError, TypeError):
            t_norm = 0.5  # unknown temperature -> midpoint
    else:
        t_norm = 0.5  # missing -> midpoint
    parts.append(torch.tensor([t_norm], dtype=torch.float32))

    return torch.cat(parts)


def get_meta_dim(categories: Dict[str, List[str]]) -> int:
    """
    Returns total dimensionality of the metadata feature vector:
    sum of (len(cats)+1) for each categorical field, plus 1 for Temp.
    """
    return sum(len(cats) + 1 for cats in categories.values()) + 1  # +1 for Temp


# Per-method reactivity normalisation (identical to v2)

def normalise_scores_by_method(df: pd.DataFrame, score_col: str,
                               method_col: str = "Method",
                               reagent_col: str = "Reagent",
                               top_pct: float = EFFECTIVE_MAX_TOP_PCT) -> pd.DataFrame:
    """
    Per-method reactivity normalisation pipeline.

    Normalises raw structure-probing scores so that values from different
    experimental methods are on a comparable 0-to-~1 scale. Each Method
    is normalised independently.

    Pipeline per Method group:
        1. Remove RL-seq scores (hydroxyl radical, different chemistry).
        2. Remove controls (NaN Method AND NaN Reagent).
        3. Clip negative values to 0 (experimental artefacts).
        4. Remove upper outliers > Q3 + 1.5*IQR.
        5. Compute effective max = mean of top 8% of values.
        6. Divide all values by effective max.

    Returns df with "Score_normalised" column added.
    """
    df = df.copy()
    n_initial = len(df)

    print(f"\n{'='*60}")
    print(f"PER-METHOD REACTIVITY NORMALISATION")
    print(f"{'='*60}")

    # Step 1: Remove RL-seq
    rl_mask = df[method_col].str.strip().str.lower() == "rl-seq"
    n_rl = rl_mask.sum()
    df = df[~rl_mask].reset_index(drop=True)
    print(f"Step 1: Removed {n_rl:,} RL-seq scores ({100*n_rl/max(n_initial,1):.2f}%)")

    # Step 2: Remove controls (NaN Method AND NaN Reagent)
    control_mask = df[method_col].isna() & df[reagent_col].isna()
    n_controls = control_mask.sum()
    df = df[~control_mask].reset_index(drop=True)
    print(f"Step 2: Removed {n_controls:,} control samples (NaN method + NaN reagent)")

    # Step 3: Clip negative scores to 0
    n_negative = (df[score_col] < 0).sum()
    df[score_col] = df[score_col].clip(lower=0)
    print(f"Step 3: Clipped {n_negative:,} negative scores to 0")

    # Steps 4-6: Per-method outlier removal + effective-max normalisation
    normalised_dfs = []
    methods_in_data = df[method_col].dropna().unique()
    print(f"\nProcessing {len(methods_in_data)} methods:")

    for method in sorted(methods_in_data):
        method_mask = df[method_col] == method
        mdf = df[method_mask].copy()
        n_method_initial = len(mdf)
        scores = mdf[score_col].values

        # Step 4: IQR outlier removal
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        outlier_mask = scores > upper_fence
        n_outliers = outlier_mask.sum()
        pct_outliers = 100 * n_outliers / max(len(scores), 1)

        mdf = mdf[~outlier_mask].reset_index(drop=True)
        scores_clean = mdf[score_col].values

        # Step 5: Compute effective maximum (mean of top 8%)
        n_top = max(1, int(len(scores_clean) * top_pct))
        top_values = np.partition(scores_clean, -n_top)[-n_top:]
        effective_max = np.mean(top_values)

        # Step 6: Divide by effective maximum
        if effective_max > 0:
            mdf["Score_normalised"] = mdf[score_col] / effective_max
        else:
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

    df = pd.concat(normalised_dfs, ignore_index=True)

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
    """Immutable container pairing a CSV file path with its species name."""
    path: str
    species: str


class BinaryScoresDataset(Dataset):
    """
    PyTorch Dataset for the binary structured-vs-unstructured task.

    After per-method normalisation, only two groups of samples are kept:
      - Structured:   normalised score == 0 (zero reactivity)
      - Unstructured: normalised score > 0.7 (high reactivity)

    All intermediate scores (0 < score <= 0.7) are discarded. The
    structured class (typically much larger) is then randomly subsampled
    to match the unstructured class size, yielding a balanced dataset.

    Subsampling the majority class rather than oversampling the minority
    means every sample in the dataset is unique — there are no duplicate
    rows. This avoids the risk of the model memorising repeated minority
    samples, which can happen with oversampling.

    Parameters:
    specs : List[CSVDatasetSpec]
        List of (path, species) pairs for each species CSV file.
    seq_col : str
        Name of the column containing RNA sequences.
    target_col : str
        Name of the column containing raw structure scores.
    dropna : bool
        Whether to drop rows with missing sequence or score values.
    seed : int
        Random seed for shuffling and subsampling.
    """

    def __init__(
        self,
        specs: List[CSVDatasetSpec],
        seq_col: str = "Seq",
        target_col: str = "Score",
        dropna: bool = True,
        seed: int = 0,
    ):
        # Load and concatenate all species CSVs
        dfs = []
        for s in specs:
            df = pd.read_csv(s.path, low_memory=False)
            df["Species"] = s.species
            dfs.append(df)
        big = pd.concat(dfs, ignore_index=True)

        # Verify required columns
        required = [seq_col, target_col, "Species"]
        missing = [c for c in required if c not in big.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(big.columns)}")

        if dropna:
            big = big.dropna(subset=[seq_col, target_col]).reset_index(drop=True)

        # Per-method reactivity normalisation
        big = normalise_scores_by_method(
            big, score_col=target_col,
            method_col="Method", reagent_col="Reagent",
            top_pct=EFFECTIVE_MAX_TOP_PCT,
        )

        # CHANGED: Binary labelling — discard intermediate scores
        # Structured = exactly 0 (zero reactivity means fully base-paired
        # or inaccessible to the probing reagent)
        # Unstructured = above threshold (high reactivity = flexible/unpaired)
        structured_mask   = big["Score_normalised"] == 0
        unstructured_mask = big["Score_normalised"] > UNSTRUCTURED_THRESHOLD
        intermediate_mask = ~structured_mask & ~unstructured_mask

        n_structured   = structured_mask.sum()
        n_unstructured = unstructured_mask.sum()
        n_intermediate = intermediate_mask.sum()
        n_total = len(big)

        print(f"\n{'='*60}")
        print(f"BINARY CLASSIFICATION: DISCARD INTERMEDIATE, SUBSAMPLE MAJORITY")
        print(f"{'='*60}")
        print(f"Before filtering:")
        print(f"  Structured (score == 0):         {n_structured:,} ({100*n_structured/n_total:.1f}%)")
        print(f"  Intermediate (0 < score <= 0.7): {n_intermediate:,} ({100*n_intermediate/n_total:.1f}%)")
        print(f"  Unstructured (score > 0.7):      {n_unstructured:,} ({100*n_unstructured/n_total:.1f}%)")

        # Keep only the two extreme classes
        keep_mask = structured_mask | unstructured_mask
        big = big[keep_mask].reset_index(drop=True)

        # Assign binary labels: 0 = structured, 1 = unstructured
        big["_class_label"] = (big["Score_normalised"] > UNSTRUCTURED_THRESHOLD).astype(int)

        # CHANGED: Subsample the majority class (structured) down to match
        # the minority class (unstructured). This creates a perfectly balanced
        # dataset where every sample is unique.
        structured_idx   = big.index[big["_class_label"] == 0].to_numpy()
        unstructured_idx = big.index[big["_class_label"] == 1].to_numpy()
        n_minority = len(unstructured_idx)

        rng = np.random.RandomState(seed)
        if len(structured_idx) > n_minority:
            # Randomly select n_minority samples from the structured class
            sampled_structured = rng.choice(structured_idx, size=n_minority, replace=False)
            keep_indices = np.concatenate([sampled_structured, unstructured_idx])
        else:
            # Rare case: unstructured is actually larger (unlikely, but safe)
            sampled_unstructured = rng.choice(unstructured_idx, size=len(structured_idx), replace=False)
            keep_indices = np.concatenate([structured_idx, sampled_unstructured])
            n_minority = len(structured_idx)

        keep_indices.sort()  # preserve original ordering before shuffle
        big = big.iloc[keep_indices].reset_index(drop=True)

        n_final_struct   = (big["_class_label"] == 0).sum()
        n_final_unstruct = (big["_class_label"] == 1).sum()
        print(f"\nAfter subsampling majority class:")
        print(f"  Structured:   {n_final_struct:,}")
        print(f"  Unstructured: {n_final_unstruct:,}")
        print(f"  Total:        {len(big):,} (balanced 50/50)")
        print(f"  Discarded intermediate: {n_intermediate:,}")
        print(f"  Discarded excess structured: {n_structured - n_final_struct:,}")
        print(f"{'='*60}\n")

        # Shuffle
        big = big.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.df = big
        self.seq_col = seq_col
        self.target_col = target_col

        # Metadata columns to return (for one-hot encoding + Temp)
        self.meta_cols = [
            "Species", "Method", "Reagent", "Temp", "Condition", "Specificity",
        ]
        self.meta_cols = [c for c in self.meta_cols if c in self.df.columns]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, Dict]:
        """
        Returns (sequence_string, binary_label, metadata_dict).

        Binary label: 0 = structured, 1 = unstructured.
        Uses torch.float32 because BCEWithLogitsLoss expects float targets.
        """
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        label = float(row["_class_label"])
        # BCEWithLogitsLoss expects float targets, not long
        y = torch.tensor(label, dtype=torch.float32)

        meta: Dict = {}
        for c in self.meta_cols:
            meta[c] = row[c]
        return seq, y, meta


# RiNALMo embedder

class RinalmoBatchEmbedder(nn.Module):
    """
    Wraps RiNALMo and converts raw RNA strings to per-token embeddings.
    Model is frozen by default, only the CNN head will be trained.
    Output shape: [B, L, D]  (B=batch, L=seq length, D=embedding dim)
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

        config  = model_config(model_size)
        model   = RiNALMo(config)
        alphabet = Alphabet(**config["alphabet"])

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()

        if freeze:
            for p in model.parameters():
                p.requires_grad = False

        self.model    = model
        self.alphabet = alphabet

    @torch.no_grad()
    def forward(self, seqs: List[str]) -> torch.Tensor:
        toks   = self.alphabet.batch_tokenize(seqs)
        tokens = torch.tensor(toks, dtype=torch.int64, device=self.device)
        if self.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = self.model(tokens)
        else:
            out = self.model(tokens)
        return out["representation"]  # [B, L, D]


# Collation and batch embedding

def cpu_collate(batch):
    """
    Collate function for DataLoader workers (CPU-only).

    Returns (seqs, labels, meta_vectors) where labels are float32 scalars
    for BCEWithLogitsLoss.
    """
    seqs, ys, metas = zip(*batch)
    y = torch.stack(ys, dim=0)  # [B] float32 binary labels
    # Build metadata feature vectors (one-hot categoricals + normalised Temp)
    meta_list = [build_meta_vector(m, META_CATEGORIES) for m in metas]
    meta = torch.stack(meta_list, dim=0)  # [B, meta_dim]
    return list(seqs), y, meta


def embed_batch(seqs: List[str], y: torch.Tensor, meta: torch.Tensor,
                embedder: RinalmoBatchEmbedder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs RiNALMo in the main process (where CUDA is initialised).
    Transposes embeddings from [B, L, D] to [B, D, L] for Conv1d.
    """
    emb = embedder(seqs)                        # [B, L, D] on GPU
    X   = emb.transpose(1, 2).contiguous()      # [B, D, L]
    y   = y.to(X.device, non_blocking=True)
    meta = meta.to(X.device, non_blocking=True)
    return X, y, meta


# CHANGED: Binary CNN classifier — single output logit
class BinaryCNNClassifier(nn.Module):
    """
    3-layer 1D CNN for binary classification of RNA structure.

    Same convolutional architecture as the 3-class version, but the
    final linear layer outputs a single logit instead of 3. This logit
    is passed through sigmoid (via BCEWithLogitsLoss) to get a
    probability of the positive class (unstructured).

    After global average pooling, the metadata feature vector (one-hot
    categoricals + normalised Temp) is concatenated with the 64-dim CNN
    features before the output layer.

    Architecture:
      [B, D, L]
        -> Conv1d(D->256, k=7) + ReLU
        -> Conv1d(256->128, k=5) + ReLU
        -> Conv1d(128->64,  k=3) + ReLU
        -> AdaptiveAvgPool1d(1) -> Flatten -> [B, 64]
        -> Concat with metadata -> [B, 64 + meta_dim]
        -> Linear(64 + meta_dim -> 1)   # single logit

    Output: raw logit [B, 1]. BCEWithLogitsLoss applies sigmoid internally.

    Parameters:
    d_model : int
        Embedding dimension from RiNALMo (480 for micro, 1280 for giga).
    meta_dim : int
        Dimensionality of the metadata feature vector.
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
        # Single output logit for binary classification
        self.head = nn.Linear(64 + meta_dim, 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x : torch.Tensor
            RiNALMo embeddings [B, D, L] in Conv1d format.
        meta : torch.Tensor
            Metadata feature vector [B, meta_dim].

        Returns:
        torch.Tensor
            Raw logit [B, 1]. Squeeze to [B] before passing to loss.
        """
        h = self.pool(self.feat(x))       # [B, 64]
        h = torch.cat([h, meta], dim=1)   # [B, 64 + meta_dim]
        return self.head(h)               # [B, 1]


# Metrics for binary classification

def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary accuracy: fraction of samples where (sigmoid(logit) > 0.5)
    matches the true label. Random baseline is 0.5 for a balanced dataset.
    """
    preds = (logits.squeeze(-1) > 0).float()  # threshold at logit=0 (equiv to prob=0.5)
    return float((preds == labels).float().mean().cpu())


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, F1 score, and AUROC for the
    binary classification task.

    Positive class = 1 (unstructured).

    Precision = TP / (TP + FP): of all samples predicted as unstructured,
                how many truly are?
    Recall    = TP / (TP + FN): of all truly unstructured samples, how
                many were correctly predicted?
    F1        = harmonic mean of precision and recall.
    AUROC     = area under the ROC curve. Measures ranking quality: how
                well the model separates the two classes regardless of
                the threshold choice.

    Parameters:
    logits : torch.Tensor
        Raw logits [B] or [B, 1].
    labels : torch.Tensor
        Binary labels [B], values in {0.0, 1.0}.

    Returns:
    Dict with keys: accuracy, precision, recall, f1, auroc
    """
    logits_flat = logits.squeeze(-1).cpu()
    labels_flat = labels.cpu()
    preds = (logits_flat > 0).float()

    tp = ((preds == 1) & (labels_flat == 1)).sum().float()
    fp = ((preds == 1) & (labels_flat == 0)).sum().float()
    fn = ((preds == 0) & (labels_flat == 1)).sum().float()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc       = float((preds == labels_flat).float().mean())

    # AUROC using the trapezoidal rule on sorted predictions
    # This is a manual implementation to avoid importing sklearn
    try:
        probs = torch.sigmoid(logits_flat)
        # Sort by predicted probability descending
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels_flat[sorted_indices]
        n_pos = labels_flat.sum().item()
        n_neg = len(labels_flat) - n_pos
        if n_pos > 0 and n_neg > 0:
            # Accumulate TP rate and FP rate
            tp_acc = 0.0
            fp_acc = 0.0
            auroc = 0.0
            prev_fpr = 0.0
            for i in range(len(sorted_labels)):
                if sorted_labels[i] == 1:
                    tp_acc += 1
                else:
                    fp_acc += 1
                    # Each time we add a FP, the trapezoid area increases
                    tpr = tp_acc / n_pos
                    fpr = fp_acc / n_neg
                    auroc += tpr * (fpr - prev_fpr)
                    prev_fpr = fpr
            auroc = float(auroc)
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


# Train & eval

def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp) -> Tuple[float, Dict[str, float], float, float]:
    """
    Train the binary CNN classifier for one epoch.

    Returns:
    (avg_loss, metrics_dict, elapsed_seconds, sequences_per_second)
    where metrics_dict contains accuracy, precision, recall, f1, auroc.
    """
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits, all_labels = [], []
    n_seqs = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, meta_cpu in loader:
        X, y, meta = embed_batch(seqs, y_cpu, meta_cpu, embedder)

        opt.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(X, meta).squeeze(-1)  # [B]
                loss   = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X, meta).squeeze(-1)  # [B]
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()

        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
        n_seqs += bs

    elapsed = time.perf_counter() - t_start
    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    n = cat_labels.shape[0]
    avg_loss = total_loss / max(n, 1)
    metrics = binary_metrics(cat_logits, cat_labels)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, metrics, elapsed, seqs_per_sec


@torch.no_grad()
def eval_one_epoch(model, loader, embedder, device) -> Tuple[float, Dict[str, float], float, float]:
    """
    Evaluate the binary CNN classifier on validation or test set.

    Returns:
    (avg_loss, metrics_dict, elapsed_seconds, sequences_per_second)
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits, all_labels = [], []
    n_seqs = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, meta_cpu in loader:
        X, y, meta = embed_batch(seqs, y_cpu, meta_cpu, embedder)
        logits = model(X, meta).squeeze(-1)  # [B]
        loss   = loss_fn(logits, y)
        bs     = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits)
        all_labels.append(y)
        n_seqs += bs

    elapsed = time.perf_counter() - t_start
    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    n = cat_labels.shape[0]
    avg_loss = total_loss / max(n, 1)
    metrics = binary_metrics(cat_logits, cat_labels)
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, metrics, elapsed, seqs_per_sec


def epoch_sampler(indices: np.ndarray, frac: float, rng: np.random.Generator) -> SubsetRandomSampler:
    """
    Randomly selects `frac` fraction of `indices` without replacement.
    Called once per epoch so each epoch sees a fresh random subset.

    Since the dataset is already balanced (50/50 after subsampling), we
    don't need WeightedRandomSampler — a uniform SubsetRandomSampler
    preserves the balance.
    """
    n = max(1, int(len(indices) * frac))
    chosen = rng.choice(indices, size=n, replace=False)
    return SubsetRandomSampler(chosen.tolist())


"""
Main section
"""

def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_dim = get_meta_dim(META_CATEGORIES)
    print(f"device: {device}  |  model_size: {MODEL_SIZE}  |  batch: {BATCH_SIZE}  |  subset: {SUBSET_FRAC*100:.0f}%")
    print(f"metadata dim: {meta_dim} (one-hot categoricals + normalised Temp)")
    print(f"EXPERIMENT: binary classifier (structured vs unstructured)")
    print(f"  structured = score == 0  |  unstructured = score > {UNSTRUCTURED_THRESHOLD}")
    print(f"  intermediate scores discarded, majority class subsampled")
    print(f"  loss = BCEWithLogitsLoss")
    print(f"  metadata: Species, Method, Reagent, Temp, Condition, Specificity")

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

    # Load dataset (normalisation + binary filtering + subsampling happens inside)
    ds = BinaryScoresDataset(
        specs=specs, seq_col="Seq", target_col="Score",
        dropna=True, seed=SEED,
    )

    print(f"total sequences (balanced binary dataset): {len(ds):,}")

    # Three-way split: test -> val -> train
    n_test  = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val   = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=split_gen
    )

    # Positional indices for epoch subsampling
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

    # Test DataLoader
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

    # Initialise binary classifier
    model   = BinaryCNNClassifier(d_model=d_model, meta_dim=meta_dim).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    start_epoch   = 1
    best_val_loss = float("inf")

    if RESUME_FROM:
        print(f"resuming from checkpoint: {RESUME_FROM}")
        ckpt_loaded = torch.load(RESUME_FROM, map_location=device)
        ckpt_type = ckpt_loaded.get("model_type", None)
        if ckpt_type is not None and ckpt_type != "binary_v3":
            raise ValueError(
                f"Checkpoint model_type is '{ckpt_type}', expected 'binary_v3'. "
                f"Are you loading the wrong checkpoint?"
            )
        model.load_state_dict(ckpt_loaded["model"])
        opt.load_state_dict(ckpt_loaded["opt"])
        start_epoch   = ckpt_loaded["epoch"] + 1
        best_val_loss = ckpt_loaded.get("val_loss", float("inf"))

        # Create a FRESH cosine schedule for the remaining epochs.
        # Do NOT load the old scheduler state, because CosineAnnealingLR
        # reverses direction after T_max steps (it's periodic). If the
        # previous run completed its full T_max cycle, loading that state
        # would cause the LR to ramp UP instead of decaying down again.
        remaining_epochs = TOTAL_EPOCHS - ckpt_loaded["epoch"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=remaining_epochs, eta_min=1e-5
        )
        print(f"  resumed at epoch {start_epoch}  "
              f"(best val_loss so far: {best_val_loss:.4f})")
        print(f"  fresh cosine schedule: T_max={remaining_epochs} "
              f"(epochs {start_epoch}..{TOTAL_EPOCHS})")
        rng = np.random.default_rng(SEED + start_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=1e-5
        )
        rng = np.random.default_rng(SEED)

    end_epoch = (TOTAL_EPOCHS if RESUME_FROM else EPOCHS) + 1

    metrics_history = []

    # Training loop
    for ep in range(start_epoch, end_epoch):

        # Uniform subsampling (dataset is already balanced)
        sampler      = epoch_sampler(train_positions, SUBSET_FRAC, rng)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False,
            collate_fn=cpu_collate, drop_last=True,
        )

        tr_loss, tr_m, tr_secs, tr_sps = train_one_epoch(model, train_loader, embedder, opt, scaler, device, USE_AMP)
        va_loss, va_m, va_secs, va_sps = eval_one_epoch(model, val_loader, embedder, device)
        scheduler.step()

        epoch_secs = tr_secs + va_secs
        full_tr_est        = tr_secs / max(SUBSET_FRAC, 1e-6)
        full_epoch_est_hrs = (full_tr_est + va_secs) / 3600

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {ep:02d} | "
            f"train loss={tr_loss:.4f} acc={tr_m['accuracy']:.4f} f1={tr_m['f1']:.4f} | "
            f"val loss={va_loss:.4f} acc={va_m['accuracy']:.4f} f1={va_m['f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )
        print(
            f"         val metrics | "
            f"prec={va_m['precision']:.4f} rec={va_m['recall']:.4f} "
            f"auroc={va_m['auroc']:.4f}"
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
            "train_acc": tr_m["accuracy"],
            "train_precision": tr_m["precision"],
            "train_recall": tr_m["recall"],
            "train_f1": tr_m["f1"],
            "train_auroc": tr_m["auroc"],
            "val_loss": va_loss,
            "val_acc": va_m["accuracy"],
            "val_precision": va_m["precision"],
            "val_recall": va_m["recall"],
            "val_f1": va_m["f1"],
            "val_auroc": va_m["auroc"],
            "lr": current_lr,
            "train_secs": tr_secs,
            "val_secs": va_secs,
            "train_seqs_per_sec": tr_sps,
            "val_seqs_per_sec": va_sps,
        }
        epoch_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in epoch_metrics.items()}
        metrics_history.append(epoch_metrics)

        metrics_path = f"{CHECKPOINT_DIR}/metrics_binary.json"
        with open(metrics_path, "w") as f:
            json.dump({"epochs": metrics_history, "test": None}, f, indent=2)

        ckpt = {
            "model_type": "binary_v3",
            "epoch":    ep,
            "model":    model.state_dict(),
            "opt":      opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": va_loss,
            "val_acc":  va_m["accuracy"],
            "val_f1":   va_m["f1"],
        }
        torch.save(ckpt, f"{CHECKPOINT_DIR}/latest.pt")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(ckpt, f"{CHECKPOINT_DIR}/best.pt")
            print(f"  -> new best val_loss={best_val_loss:.4f} "
                  f"acc={va_m['accuracy']:.4f} f1={va_m['f1']:.4f}, saved best.pt")

    # Final evaluation on held-out test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    best_ckpt = torch.load(f"{CHECKPOINT_DIR}/best.pt", map_location=device)
    if best_ckpt.get("model_type", None) not in (None, "binary_v3"):
        raise ValueError("best.pt model_type mismatch — wrong checkpoint directory?")
    model.load_state_dict(best_ckpt["model"])
    te_loss, te_m, te_secs, te_sps = eval_one_epoch(
        model, test_loader, embedder, device
    )
    print(f"test loss={te_loss:.4f}")
    print(f"  accuracy:  {te_m['accuracy']:.4f}")
    print(f"  precision: {te_m['precision']:.4f}")
    print(f"  recall:    {te_m['recall']:.4f}")
    print(f"  F1:        {te_m['f1']:.4f}")
    print(f"  AUROC:     {te_m['auroc']:.4f}")
    print(f"  eval time: {te_secs/60:.1f}min ({te_sps:.0f} seq/s)")
    print("="*60)

    # Append test results to metrics JSON
    test_metrics = {
        "test_loss": te_loss,
        "test_acc": te_m["accuracy"],
        "test_precision": te_m["precision"],
        "test_recall": te_m["recall"],
        "test_f1": te_m["f1"],
        "test_auroc": te_m["auroc"],
    }
    test_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in test_metrics.items()}
    final_output = {"epochs": metrics_history, "test": test_metrics}
    metrics_path = f"{CHECKPOINT_DIR}/metrics_binary.json"
    with open(metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
