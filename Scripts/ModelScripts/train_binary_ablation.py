#!/usr/bin/env python3
"""
train_binary_ablation.py

Metadata ablation study for the binary RNA structure classifier.

This is a parameterised version of train_cnn_rinalmo_binary_v3.py that
accepts a --ablate flag to exclude one metadata field at a time.

Author: Zofia Witkowski-Blake
Project: Predicting RNA structure scores using CNNs and RiNALMo embeddings
University of Melbourne, MSc Bioinformatics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet


MODEL_SIZE   = "micro"
WEIGHTS_PATH = "weights/rinalmo_micro_pretrained.pt"
BATCH_SIZE   = 256
SUBSET_FRAC  = 0.1
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-2

TEST_FRAC    = 0.10
VAL_FRAC     = 0.10

SEED         = 0
NUM_WORKERS  = 4
USE_AMP      = True

UNSTRUCTURED_THRESHOLD = 0.7
EFFECTIVE_MAX_TOP_PCT  = 0.08

TEMP_MIN = 25.0
TEMP_MAX = 95.0

# All metadata fields and their categories
ALL_META_CATEGORIES = {
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

VALID_ABLATE_FIELDS = list(ALL_META_CATEGORIES.keys()) + ["Temp", "none"]


def build_meta_vector(meta: Dict, categories: Dict[str, List[str]],
                      include_temp: bool = True,
                      temp_min: float = TEMP_MIN,
                      temp_max: float = TEMP_MAX) -> torch.Tensor:
    """
    Builds the metadata feature vector, optionally excluding Temp.

    Parameters:
    meta : Dict
        Row metadata from __getitem__.
    categories : Dict[str, List[str]]
        Categorical fields to include (already filtered by ablation).
    include_temp : bool
        Whether to append the normalised Temp scalar.
    """
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

    if include_temp:
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

    if not parts:
        # Edge case: all metadata ablated (shouldn't happen, but safe)
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def get_meta_dim(categories: Dict[str, List[str]], include_temp: bool = True) -> int:
    """Total dimensionality of the metadata vector."""
    dim = sum(len(cats) + 1 for cats in categories.values())
    if include_temp:
        dim += 1
    return dim


def normalise_scores_by_method(df: pd.DataFrame, score_col: str,
                               method_col: str = "Method",
                               reagent_col: str = "Reagent",
                               top_pct: float = EFFECTIVE_MAX_TOP_PCT) -> pd.DataFrame:
    """Per-method reactivity normalisation. See binary_v3 for full docs."""
    df = df.copy()
    n_initial = len(df)

    print(f"\n{'='*60}")
    print(f"PER-METHOD REACTIVITY NORMALISATION")
    print(f"{'='*60}")

    rl_mask = df[method_col].str.strip().str.lower() == "rl-seq"
    n_rl = rl_mask.sum()
    df = df[~rl_mask].reset_index(drop=True)
    print(f"Step 1: Removed {n_rl:,} RL-seq scores ({100*n_rl/max(n_initial,1):.2f}%)")

    control_mask = df[method_col].isna() & df[reagent_col].isna()
    n_controls = control_mask.sum()
    df = df[~control_mask].reset_index(drop=True)
    print(f"Step 2: Removed {n_controls:,} control samples")

    n_negative = (df[score_col] < 0).sum()
    df[score_col] = df[score_col].clip(lower=0)
    print(f"Step 3: Clipped {n_negative:,} negative scores to 0")

    normalised_dfs = []
    methods_in_data = df[method_col].dropna().unique()
    print(f"\nProcessing {len(methods_in_data)} methods:")

    for method in sorted(methods_in_data):
        method_mask = df[method_col] == method
        mdf = df[method_mask].copy()
        n_method_initial = len(mdf)
        scores = mdf[score_col].values

        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        outlier_mask = scores > upper_fence
        n_outliers = outlier_mask.sum()
        pct_outliers = 100 * n_outliers / max(len(scores), 1)

        mdf = mdf[~outlier_mask].reset_index(drop=True)
        scores_clean = mdf[score_col].values

        n_top = max(1, int(len(scores_clean) * top_pct))
        top_values = np.partition(scores_clean, -n_top)[-n_top:]
        effective_max = np.mean(top_values)

        if effective_max > 0:
            mdf["Score_normalised"] = mdf[score_col] / effective_max
        else:
            mdf["Score_normalised"] = 0.0

        normalised_dfs.append(mdf)
        print(f"  {method}: {n_method_initial:,} -> {len(mdf):,} "
              f"({n_outliers:,} outliers, eff_max={effective_max:.4f})")

    df = pd.concat(normalised_dfs, ignore_index=True)
    n_final = len(df)
    print(f"\nTotal: {n_initial:,} -> {n_final:,} "
          f"({n_initial - n_final:,} removed, {100*(n_initial-n_final)/max(n_initial,1):.1f}%)")
    print(f"{'='*60}\n")
    return df



@dataclass(frozen=True)
class CSVDatasetSpec:
    path: str
    species: str


class BinaryScoresDataset(Dataset):
    """Binary dataset: structured (score==0) vs unstructured (score>0.7)."""

    def __init__(self, specs, seq_col="Seq", target_col="Score",
                 dropna=True, seed=0):
        dfs = []
        for s in specs:
            df = pd.read_csv(s.path, low_memory=False)
            df["Species"] = s.species
            dfs.append(df)
        big = pd.concat(dfs, ignore_index=True)

        required = [seq_col, target_col, "Species"]
        missing = [c for c in required if c not in big.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if dropna:
            big = big.dropna(subset=[seq_col, target_col]).reset_index(drop=True)

        big = normalise_scores_by_method(big, score_col=target_col)

        structured_mask   = big["Score_normalised"] == 0
        unstructured_mask = big["Score_normalised"] > UNSTRUCTURED_THRESHOLD

        n_structured   = structured_mask.sum()
        n_unstructured = unstructured_mask.sum()
        n_intermediate = (~structured_mask & ~unstructured_mask).sum()

        print(f"Structured: {n_structured:,} | Intermediate (discarded): "
              f"{n_intermediate:,} | Unstructured: {n_unstructured:,}")

        keep_mask = structured_mask | unstructured_mask
        big = big[keep_mask].reset_index(drop=True)
        big["_class_label"] = (big["Score_normalised"] > UNSTRUCTURED_THRESHOLD).astype(int)

        # Subsample majority to match minority
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

        n_s = (big["_class_label"] == 0).sum()
        n_u = (big["_class_label"] == 1).sum()
        print(f"Balanced: {n_s:,} structured + {n_u:,} unstructured = {len(big):,} total")

        self.df = big
        self.seq_col = seq_col
        self.meta_cols = [c for c in
            ["Species", "Method", "Reagent", "Temp", "Condition", "Specificity"]
            if c in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        y = torch.tensor(float(row["_class_label"]), dtype=torch.float32)
        meta = {c: row[c] for c in self.meta_cols}
        return seq, y, meta



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



def binary_metrics(logits, labels):
    logits_flat = logits.squeeze(-1).cpu()
    labels_flat = labels.cpu()
    preds = (logits_flat > 0).float()

    tp = ((preds == 1) & (labels_flat == 1)).sum().float()
    fp = ((preds == 1) & (labels_flat == 0)).sum().float()
    fn = ((preds == 0) & (labels_flat == 1)).sum().float()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = float((preds == labels_flat).float().mean())

    try:
        probs = torch.sigmoid(logits_flat)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels_flat[sorted_indices]
        n_pos = labels_flat.sum().item()
        n_neg = len(labels_flat) - n_pos
        if n_pos > 0 and n_neg > 0:
            tp_acc, fp_acc, auroc, prev_fpr = 0.0, 0.0, 0.0, 0.0
            for i in range(len(sorted_labels)):
                if sorted_labels[i] == 1:
                    tp_acc += 1
                else:
                    fp_acc += 1
                    tpr = tp_acc / n_pos
                    fpr = fp_acc / n_neg
                    auroc += tpr * (fpr - prev_fpr)
                    prev_fpr = fpr
            auroc = float(auroc)
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")

    return {"accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "auroc": auroc}



def make_collate(active_categories, include_temp):
    """
    Returns a collate function that builds metadata vectors using
    only the active (non-ablated) categories and temp flag.
    """
    def cpu_collate(batch):
        seqs, ys, metas = zip(*batch)
        y = torch.stack(ys, dim=0)
        meta_list = [build_meta_vector(m, active_categories,
                                       include_temp=include_temp)
                     for m in metas]
        meta = torch.stack(meta_list, dim=0)
        return list(seqs), y, meta
    return cpu_collate


def embed_batch(seqs, y, meta, embedder):
    emb = embedder(seqs)
    X = emb.transpose(1, 2).contiguous()
    y = y.to(X.device, non_blocking=True)
    meta = meta.to(X.device, non_blocking=True)
    return X, y, meta


def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, meta_cpu in loader:
        X, y, meta = embed_batch(seqs, y_cpu, meta_cpu, embedder)
        opt.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(X, meta).squeeze(-1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X, meta).squeeze(-1)
            loss = loss_fn(logits, y)
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
    return total_loss / max(n, 1), binary_metrics(cat_logits, cat_labels), elapsed, n_seqs / max(elapsed, 1e-6)


@torch.no_grad()
def eval_one_epoch(model, loader, embedder, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs = 0
    t_start = time.perf_counter()

    for seqs, y_cpu, meta_cpu in loader:
        X, y, meta = embed_batch(seqs, y_cpu, meta_cpu, embedder)
        logits = model(X, meta).squeeze(-1)
        loss = loss_fn(logits, y)
        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits)
        all_labels.append(y)
        n_seqs += bs

    elapsed = time.perf_counter() - t_start
    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    n = cat_labels.shape[0]
    return total_loss / max(n, 1), binary_metrics(cat_logits, cat_labels), elapsed, n_seqs / max(elapsed, 1e-6)


def epoch_sampler(indices, frac, rng):
    n = max(1, int(len(indices) * frac))
    chosen = rng.choice(indices, size=n, replace=False)
    return SubsetRandomSampler(chosen.tolist())



def main():
    import os

    parser = argparse.ArgumentParser(
        description="Binary classifier metadata ablation study")
    parser.add_argument(
        "--ablate", type=str, required=True,
        choices=VALID_ABLATE_FIELDS,
        help="Metadata field to REMOVE. Use 'none' for the full-metadata baseline.")
    args = parser.parse_args()

    ablate_field = args.ablate

    # Build active metadata categories (everything except the ablated field)
    if ablate_field == "none":
        active_categories = dict(ALL_META_CATEGORIES)
        include_temp = True
        experiment_name = "none (full metadata baseline)"
    elif ablate_field == "Temp":
        active_categories = dict(ALL_META_CATEGORIES)
        include_temp = False
        experiment_name = "Temp (temperature removed)"
    else:
        active_categories = {k: v for k, v in ALL_META_CATEGORIES.items()
                             if k != ablate_field}
        include_temp = True
        experiment_name = f"{ablate_field} (removed from metadata)"

    checkpoint_dir = f"checkpoints_ablate_{ablate_field}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    meta_dim = get_meta_dim(active_categories, include_temp)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"ABLATION STUDY: --ablate {ablate_field}")
    print(f"  experiment: {experiment_name}")
    print(f"  metadata dim: {meta_dim}")
    print(f"  active categoricals: {list(active_categories.keys())}")
    print(f"  include Temp: {include_temp}")
    print(f"  checkpoints: {checkpoint_dir}/")
    print(f"  device: {device}  |  model: {MODEL_SIZE}  |  batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    # Dataset
    specs = [
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_cereus_processed.csv", "b_cereus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/s_enterica_processed.csv", "s_enterica"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/synechococcus_processed.csv", "synechococcus"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/b_subtilis_processed.csv", "b_subtilis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/p_putida_processed.csv", "p_putida"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/y_pseudotuberculosis_processed.csv", "y_pseudotuberculosis"),
        CSVDatasetSpec("/data/scratch/projects/punim1021/zwitkowskibl/processed/e_coli_processed.csv", "e_coli"),
    ]

    ds = BinaryScoresDataset(specs=specs, dropna=True, seed=SEED)
    print(f"Dataset: {len(ds):,} samples")

    # Splits (identical seed -> identical splits across all ablation runs)
    n_test = int(len(ds) * TEST_FRAC)
    n_remaining = len(ds) - n_test
    n_val = int(n_remaining * VAL_FRAC)
    n_train = n_remaining - n_val

    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=split_gen)

    train_positions = np.arange(n_train)
    print(f"Split: {n_train:,} train / {n_val:,} val / {n_test:,} test "
          f"(~{int(n_train*SUBSET_FRAC):,}/epoch)")

    # Collate function with the active metadata configuration
    collate_fn = make_collate(active_categories, include_temp)

    # Embedder
    embedder = RinalmoBatchEmbedder(
        weights_path=WEIGHTS_PATH, device=device,
        model_size=MODEL_SIZE, use_amp=USE_AMP, freeze=True)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=collate_fn, drop_last=False)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=collate_fn, drop_last=False)

    # Probe d_model
    sample_seqs = [ds[i][0] for i in range(4)]
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)
    d_model = sample_emb.shape[2]
    print(f"d_model: {d_model}")
    del sample_emb

    # Model
    model = BinaryCNNClassifier(d_model=d_model, meta_dim=meta_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    rng = np.random.default_rng(SEED)
    best_val_loss = float("inf")
    metrics_history = []

    # Training loop
    for ep in range(1, EPOCHS + 1):
        sampler = epoch_sampler(train_positions, SUBSET_FRAC, rng)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False,
            collate_fn=collate_fn, drop_last=True)

        tr_loss, tr_m, tr_secs, tr_sps = train_one_epoch(
            model, train_loader, embedder, opt, scaler, device, USE_AMP)
        va_loss, va_m, va_secs, va_sps = eval_one_epoch(
            model, val_loader, embedder, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {ep:02d} | "
            f"train loss={tr_loss:.4f} acc={tr_m['accuracy']:.4f} f1={tr_m['f1']:.4f} | "
            f"val loss={va_loss:.4f} acc={va_m['accuracy']:.4f} f1={va_m['f1']:.4f} "
            f"auroc={va_m['auroc']:.4f} | lr={current_lr:.2e} | "
            f"{tr_secs/60:.1f}+{va_secs/60:.1f}min")

        epoch_metrics = {
            "epoch": ep,
            "train_loss": tr_loss,   "train_acc": tr_m["accuracy"],
            "train_precision": tr_m["precision"], "train_recall": tr_m["recall"],
            "train_f1": tr_m["f1"],  "train_auroc": tr_m["auroc"],
            "val_loss": va_loss,     "val_acc": va_m["accuracy"],
            "val_precision": va_m["precision"], "val_recall": va_m["recall"],
            "val_f1": va_m["f1"],    "val_auroc": va_m["auroc"],
            "lr": current_lr,
            "train_secs": tr_secs,   "val_secs": va_secs,
        }
        epoch_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in epoch_metrics.items()}
        metrics_history.append(epoch_metrics)

        metrics_path = f"{checkpoint_dir}/metrics_binary.json"
        with open(metrics_path, "w") as f:
            json.dump({"ablated": ablate_field, "meta_dim": meta_dim,
                        "epochs": metrics_history, "test": None}, f, indent=2)

        ckpt = {
            "model_type": f"binary_ablate_{ablate_field}",
            "ablated": ablate_field,
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": va_loss,
        }
        torch.save(ckpt, f"{checkpoint_dir}/latest.pt")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(ckpt, f"{checkpoint_dir}/best.pt")
            print(f"  -> new best val_loss={best_val_loss:.4f}")

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION (ablate={ablate_field})")
    print(f"{'='*60}")
    best_ckpt = torch.load(f"{checkpoint_dir}/best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    te_loss, te_m, te_secs, te_sps = eval_one_epoch(
        model, test_loader, embedder, device)
    print(f"test loss={te_loss:.4f}  acc={te_m['accuracy']:.4f}  "
          f"prec={te_m['precision']:.4f}  rec={te_m['recall']:.4f}  "
          f"f1={te_m['f1']:.4f}  auroc={te_m['auroc']:.4f}")
    print(f"eval time: {te_secs/60:.1f}min ({te_sps:.0f} seq/s)")
    print(f"{'='*60}")

    test_metrics = {
        "test_loss": te_loss, "test_acc": te_m["accuracy"],
        "test_precision": te_m["precision"], "test_recall": te_m["recall"],
        "test_f1": te_m["f1"], "test_auroc": te_m["auroc"],
    }
    test_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in test_metrics.items()}
    final_output = {"ablated": ablate_field, "meta_dim": meta_dim,
                    "epochs": metrics_history, "test": test_metrics}
    with open(metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Metrics: {metrics_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
