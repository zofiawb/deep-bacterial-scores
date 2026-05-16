#!/usr/bin/env python3
"""

Output columns:
    Seq               - 101-nt RNA sequence window
    Coord             - genomic coordinate of the window centre
    Species           - bacterial species
    Method            - probing method (DMS-seq, SHAPE-MaP, etc.)
    Reagent           - chemical reagent used
    Temp              - experimental temperature (°C)
    Condition         - in_vivo / in_vitro / ex_vivo
    Specificity       - transcriptome-wide / targeted
    Study_ID          - numerical study identifier
    Paper             - journal and year
    Score_raw         - original raw structure score from the CSV
    Score_normalised  - score after per-method reactivity normalisation
    True_label        - ground truth binary label (0=structured, 1=unstructured)
    Pred_logit        - raw model logit (before sigmoid)
    Pred_prob         - sigmoid(logit), P(unstructured), range [0,1]
    Pred_label        - binary prediction (0 if prob<0.5, 1 if prob>=0.5)

The Pred_prob column can be used as a continuous pseudo-reactivity for
RNAstructure's pseudo-free energy equation: ΔG = m·ln(Pred_prob) + b.
The Pred_label column gives binary paired(0)/unpaired(1) calls.

Author: Zofia Witkowski-Blake
Project: Predicting RNA structure scores using CNNs and RiNALMo embeddings
University of Melbourne, MSc Bioinformatics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet

MODEL_SIZE   = "micro"
WEIGHTS_PATH = "weights/rinalmo_micro_pretrained.pt"
BATCH_SIZE   = 256
NUM_WORKERS  = 4
USE_AMP      = True
SEED         = 0

TEST_FRAC    = 0.10
VAL_FRAC     = 0.10

UNSTRUCTURED_THRESHOLD = 0.7
EFFECTIVE_MAX_TOP_PCT  = 0.08

TEMP_MIN = 25.0
TEMP_MAX = 95.0

# Path to the best checkpoint from training run 2 (epochs 21-40)
# which achieved the best test AUROC of 0.831
CHECKPOINT_PATH = "checkpoints_binary_v3/best.pt"
OUTPUT_DIR      = "predictions"

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




def build_meta_vector(meta: Dict, categories: Dict[str, List[str]],
                      temp_min: float = TEMP_MIN,
                      temp_max: float = TEMP_MAX) -> torch.Tensor:
    parts = []
    for field, cats in categories.items():
        n = len(cats) + 1
        vec = torch.zeros(n, dtype=torch.float32)
        val = meta.get(field, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)) and val in cats:
            vec[cats.index(val)] = 1.0
        else:
            vec[-1] = 1.0
        parts.append(vec)

    temp_val = meta.get("Temp", None)
    if temp_val is not None and not (isinstance(temp_val, float) and np.isnan(temp_val)):
        try:
            t = float(temp_val)
            t_norm = (t - temp_min) / max(temp_max - temp_min, 1e-6)
            t_norm = max(0.0, min(1.0, t_norm))
        except (ValueError, TypeError):
            t_norm = 0.5
    else:
        t_norm = 0.5
    parts.append(torch.tensor([t_norm], dtype=torch.float32))
    return torch.cat(parts)


def get_meta_dim(categories: Dict[str, List[str]]) -> int:
    return sum(len(cats) + 1 for cats in categories.values()) + 1


def normalise_scores_by_method(df, score_col, method_col="Method",
                               reagent_col="Reagent",
                               top_pct=EFFECTIVE_MAX_TOP_PCT):
    df = df.copy()
    n_initial = len(df)

    rl_mask = df[method_col].str.strip().str.lower() == "rl-seq"
    df = df[~rl_mask].reset_index(drop=True)

    control_mask = df[method_col].isna() & df[reagent_col].isna()
    df = df[~control_mask].reset_index(drop=True)

    n_negative = (df[score_col] < 0).sum()
    df[score_col] = df[score_col].clip(lower=0)

    normalised_dfs = []
    for method in sorted(df[method_col].dropna().unique()):
        mdf = df[df[method_col] == method].copy()
        scores = mdf[score_col].values

        q3 = np.percentile(scores, 75)
        q1 = np.percentile(scores, 25)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        mdf = mdf[scores <= upper_fence].reset_index(drop=True)
        scores_clean = mdf[score_col].values

        n_top = max(1, int(len(scores_clean) * top_pct))
        top_values = np.partition(scores_clean, -n_top)[-n_top:]
        effective_max = np.mean(top_values)

        if effective_max > 0:
            mdf["Score_normalised"] = mdf[score_col] / effective_max
        else:
            mdf["Score_normalised"] = 0.0
        normalised_dfs.append(mdf)

    df = pd.concat(normalised_dfs, ignore_index=True)
    print(f"Normalisation: {n_initial:,} -> {len(df):,} samples")
    return df



@dataclass(frozen=True)
class CSVDatasetSpec:
    path: str
    species: str


class BinaryInferenceDataset(Dataset):
    """
    Loads and prepares data identically to BinaryScoresDataset from
    training, but preserves ALL original columns for export. The same
    seed, same normalisation, same filtering, and same subsampling logic
    ensures the test split contains exactly the same samples as during
    training.
    """

    # Columns to preserve in the output CSV
    EXPORT_COLS = [
        "Seq", "Coord", "Species", "Method", "Reagent", "Temp",
        "Condition", "Specificity", "Study ID", "Paper",
        "Score", "Score_normalised", "_class_label",
    ]

    def __init__(self, specs, seq_col="Seq", target_col="Score",
                 dropna=True, seed=0):
        dfs = []
        for s in specs:
            df = pd.read_csv(s.path, low_memory=False)
            df["Species"] = s.species
            dfs.append(df)
        big = pd.concat(dfs, ignore_index=True)

        if dropna:
            big = big.dropna(subset=[seq_col, target_col]).reset_index(drop=True)

        # Store raw score before normalisation overwrites
        big["Score_raw"] = big[target_col].copy()

        big = normalise_scores_by_method(big, score_col=target_col)

        # Binary filtering
        structured_mask   = big["Score_normalised"] == 0
        unstructured_mask = big["Score_normalised"] > UNSTRUCTURED_THRESHOLD

        keep_mask = structured_mask | unstructured_mask
        big = big[keep_mask].reset_index(drop=True)
        big["_class_label"] = (big["Score_normalised"] > UNSTRUCTURED_THRESHOLD).astype(int)

        # Subsample majority (same logic and seed as training)
        structured_idx   = big.index[big["_class_label"] == 0].to_numpy()
        unstructured_idx = big.index[big["_class_label"] == 1].to_numpy()
        n_minority = min(len(structured_idx), len(unstructured_idx))

        rng = np.random.RandomState(seed)
        if len(structured_idx) > n_minority:
            sampled = rng.choice(structured_idx, size=n_minority, replace=False)
            keep_indices = np.concatenate([sampled, unstructured_idx])
        else:
            sampled = rng.choice(unstructured_idx, size=n_minority, replace=False)
            keep_indices = np.concatenate([structured_idx, sampled])

        keep_indices.sort()
        big = big.iloc[keep_indices].reset_index(drop=True)
        big = big.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        print(f"Inference dataset: {len(big):,} samples "
              f"({(big['_class_label']==0).sum():,} structured, "
              f"{(big['_class_label']==1).sum():,} unstructured)")

        self.df = big
        self.seq_col = seq_col

        self.meta_cols = [
            "Species", "Method", "Reagent", "Temp", "Condition", "Specificity",
        ]
        self.meta_cols = [c for c in self.meta_cols if c in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        y = torch.tensor(float(row["_class_label"]), dtype=torch.float32)
        meta = {c: row[c] for c in self.meta_cols}
        return seq, y, meta, idx  # return idx so we can map back to df rows



class RinalmoBatchEmbedder(nn.Module):
    def __init__(self, weights_path, device, model_size="micro",
                 use_amp=True, freeze=True):
        super().__init__()
        self.device = device
        self.use_amp = use_amp
        config = model_config(model_size)
        model = RiNALMo(config)
        alphabet = Alphabet(**config["alphabet"])
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        self.model = model
        self.alphabet = alphabet

    @torch.no_grad()
    def forward(self, seqs):
        toks = self.alphabet.batch_tokenize(seqs)
        tokens = torch.tensor(toks, dtype=torch.int64, device=self.device)
        if self.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = self.model(tokens)
        else:
            out = self.model(tokens)
        return out["representation"]


class BinaryCNNClassifier(nn.Module):
    def __init__(self, d_model, meta_dim):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(d_model, 256, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(64 + meta_dim, 1)

    def forward(self, x, meta):
        h = self.pool(self.feat(x))
        h = torch.cat([h, meta], dim=1)
        return self.head(h)



def cpu_collate(batch):
    seqs, ys, metas, idxs = zip(*batch)
    y = torch.stack(ys, dim=0)
    meta_list = [build_meta_vector(m, META_CATEGORIES) for m in metas]
    meta = torch.stack(meta_list, dim=0)
    return list(seqs), y, meta, torch.tensor(idxs, dtype=torch.long)


def embed_batch(seqs, y, meta, embedder):
    emb = embedder(seqs)
    X = emb.transpose(1, 2).contiguous()
    y = y.to(X.device, non_blocking=True)
    meta = meta.to(X.device, non_blocking=True)
    return X, y, meta



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_dim = get_meta_dim(META_CATEGORIES)
    print(f"device: {device}  |  model: {MODEL_SIZE}  |  meta_dim: {meta_dim}")
    print(f"checkpoint: {CHECKPOINT_PATH}")

    # Load dataset with identical pipeline to training
    specs = [
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_cereus_processed.csv", "b_cereus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/s_enterica_processed.csv", "s_enterica"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/synechococcus_processed.csv", "synechococcus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_subtilis_processed.csv", "b_subtilis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/p_putida_processed.csv", "p_putida"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/y_pseudotuberculosis_processed.csv", "y_pseudotuberculosis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/e_coli_processed.csv", "e_coli"),
    ]

    ds = BinaryInferenceDataset(specs=specs, dropna=True, seed=SEED)

    # Reproduce the EXACT same splits as training
    # (same seed, same sizes, same Generator)
    n_test = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=split_gen)

    print(f"Splits: {n_train:,} train / {n_val:,} val / {n_test:,} test")
    print(f"Running inference on TEST SET ({n_test:,} samples)")

    # DataLoader for the test set (no shuffling, no subsampling)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=cpu_collate, drop_last=False)

    # Load embedder
    embedder = RinalmoBatchEmbedder(
        weights_path=WEIGHTS_PATH, device=device,
        model_size=MODEL_SIZE, use_amp=USE_AMP, freeze=True)

    # Probe d_model
    sample_seqs = [ds[i][0] for i in range(4)]
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)
    d_model = sample_emb.shape[2]
    del sample_emb

    # Load model from best checkpoint
    model = BinaryCNNClassifier(d_model=d_model, meta_dim=meta_dim).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')}")

    # Run inference
    all_logits = []
    all_labels = []
    all_indices = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for seqs, y_cpu, meta_cpu, idx_cpu in test_loader:
            X, y, meta = embed_batch(seqs, y_cpu, meta_cpu, embedder)
            logits = model(X, meta).squeeze(-1)  # [B]

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            all_indices.append(idx_cpu)

    elapsed = time.perf_counter() - t_start
    print(f"Inference complete: {elapsed:.1f}s "
          f"({n_test / max(elapsed, 1e-6):.0f} seq/s)")

    # Concatenate results
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    indices = torch.cat(all_indices).numpy()

    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    pred_labels = (probs >= 0.5).astype(int)

    # Build output DataFrame from the original dataset rows
    # indices map back to positions in the full dataset (ds.df)
    out_df = ds.df.iloc[indices].copy().reset_index(drop=True)

    # Rename columns for clarity
    out_df = out_df.rename(columns={
        "Score": "Score_raw",
        "_class_label": "True_label",
    })

    # Add prediction columns
    out_df["Pred_logit"] = logits
    out_df["Pred_prob"]  = probs
    out_df["Pred_label"] = pred_labels

    # Rename Study ID to avoid spaces in column name
    if "Study ID" in out_df.columns:
        out_df = out_df.rename(columns={"Study ID": "Study_ID"})

    # Select and order output columns
    output_cols = [
        "Seq", "Coord", "Species", "Method", "Reagent", "Temp",
        "Condition", "Specificity", "Study_ID", "Paper",
        "Score_raw", "Score_normalised", "True_label",
        "Pred_logit", "Pred_prob", "Pred_label",
    ]
    # Keep only columns that exist (some may be missing from CSVs)
    output_cols = [c for c in output_cols if c in out_df.columns]
    out_df = out_df[output_cols]

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved: {csv_path}")
    print(f"  rows: {len(out_df):,}")
    print(f"  columns: {list(out_df.columns)}")

    # Compute and save summary metrics
    tp = ((pred_labels == 1) & (labels == 1)).sum()
    fp = ((pred_labels == 1) & (labels == 0)).sum()
    fn = ((pred_labels == 0) & (labels == 1)).sum()
    tn = ((pred_labels == 0) & (labels == 0)).sum()

    accuracy  = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Pearson correlation between predicted probability and true label
    pearson_r = np.corrcoef(probs, labels)[0, 1]

    # Pearson correlation between predicted probability and normalised score
    if "Score_normalised" in out_df.columns:
        pearson_r_score = np.corrcoef(probs, out_df["Score_normalised"].values)[0, 1]
    else:
        pearson_r_score = None

    summary = {
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": ckpt.get("epoch"),
        "n_test": int(n_test),
        "n_structured": int((labels == 0).sum()),
        "n_unstructured": int((labels == 1).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pearson_r_vs_label": float(pearson_r),
        "pearson_r_vs_normalised_score": float(pearson_r_score) if pearson_r_score is not None else None,
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
        },
        "inference_time_secs": elapsed,
        "pred_prob_stats": {
            "mean": float(probs.mean()),
            "std": float(probs.std()),
            "min": float(probs.min()),
            "max": float(probs.max()),
            "median": float(np.median(probs)),
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SET INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Pearson r (prob vs label):            {pearson_r:.4f}")
    if pearson_r_score is not None:
        print(f"  Pearson r (prob vs normalised score): {pearson_r_score:.4f}")
    print(f"  Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n  Pred_prob distribution:")
    print(f"    mean={probs.mean():.4f}  std={probs.std():.4f}")
    print(f"    min={probs.min():.4f}  median={np.median(probs):.4f}  max={probs.max():.4f}")

    # Class breakdown
    print(f"\n  Per-class Pred_prob means:")
    struct_probs = probs[labels == 0]
    unstruct_probs = probs[labels == 1]
    print(f"    Structured (true=0):   mean={struct_probs.mean():.4f}  std={struct_probs.std():.4f}")
    print(f"    Unstructured (true=1): mean={unstruct_probs.mean():.4f}  std={unstruct_probs.std():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
