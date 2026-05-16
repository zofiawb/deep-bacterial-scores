from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import time
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
embeddings to classify per-window RNA structure scores into three structural
classes: unstructured, intermediate, and structured.

Pipeline overview:
    1. Load processed CSV files (one per bacterial species) containing
       RNA sequences and their experimentally-derived structure scores
    2. Bin continuous structure scores into 3 discrete classes
    3. Embed each RNA sequence using RiNALMo (a pre-trained RNA language
       model) as a frozen feature extractor 
    4. Train a lightweight CNN classifier on the embeddings
    5. Save checkpoints every epoch for resumability on HPC (SPARTAN)

- RiNALMo 'micro' (35M params, D=480) is used instead of 'giga'
      (600M params, D=1280) because with ~13M sequences, on-the-fly
      embedding with giga takes >1hr per epoch. Micro is ~10x faster.
    - GPU embedding is done in the main process, NOT in DataLoader
      workers, because CUDA cannot be re-initialised in forked
      subprocesses (a PyTorch/Linux limitation).
    - Epoch-level subsampling (default 10%) enables fast iteration while covering the full dataset across multiple epochs.
    - Per-class accuracy is tracked alongside overall accuracy to detect class imbalance issues (e.g. model only predicting the majority class).

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
VAL_FRAC     = 0.10              # fraction of data held out for validation
SEED         = 0                 # random seed for reproducibility
NUM_WORKERS  = 4                 # CPU workers for DataLoader prefetching
USE_AMP      = True              # automatic mixed precision (float16) on GPU
CHECKPOINT_DIR = "checkpoints"   # directory to save per-epoch model weights

# Scores are mapped to 3 classes:
#   score <= BIN_MIN  ->  0  (unstructured / flexible nucleotide)
#   BIN_MIN < score < BIN_MAX -> 1  (intermediate)
#   score >= BIN_MAX  ->  2  (structured / rigid nucleotide)
# CrossEntropyLoss requires integer class labels 0, 1, 2.
BIN_MIN = -0.70   # corresponds to 'min' in bin_SHAPE_data
BIN_MAX =  0.33   # corresponds to 'max' in bin_SHAPE_data
NUM_CLASSES = 3

# Resume-from-checkpoint settings. RESUME_FROM = checkpoint path to resume an interrupted run
RESUME_FROM  = None              # can be checkpoints/...
TOTAL_EPOCHS = 40                # total epochs INCLUDING already-completed ones



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

    The continuous structure scores are binned into 3 classes at load time

    PyTorch's CrossEntropyLoss requires 0-indexed integer labels, which is
    why the original R labels {-1, 0, 1} are remapped to {0, 1, 2}.

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
        Returns one sample as (sequence_string, class_label, metadata_dict).
        The continuous score is binned into a class label here.
        """
        row   = self.df.iloc[idx]
        seq   = str(row[self.seq_col])
        score = float(row[self.target_col])

        # Bin score
        if score <= BIN_MIN:
            label = 0 
        elif score >= BIN_MAX:
            label = 2
        else:
            label = 1
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
    Assembles raw sequences and class labels from individual samples.

    NO GPU operations. GPU embedding happens afterwards in the main process via embed_batch().

    Parameters:
    batch : list of (str, torch.Tensor, dict)
        Individual samples from BacterialScoresDataset.__getitem__

    Returns:
    (List[str], torch.Tensor)
        Sequences as strings (not yet embedded), labels as CPU tensor.

    """
    seqs, ys, _ = zip(*batch)
    y = torch.stack(ys, dim=0)  # [B] int64 class labels on CPU
    return list(seqs), y        # seqs: List[str], y: CPU tensor


def embed_batch(seqs: List[str], y: torch.Tensor,
                embedder: RinalmoBatchEmbedder) -> Tuple[torch.Tensor, torch.Tensor]:
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
    embedder : RinalmoBatchEmbedder
        The frozen RiNALMo model.

    Returns:
    (torch.Tensor, torch.Tensor)
        X: embedded sequences [B, D, L] on GPU
        y: class labels [B] on GPU

    """
    emb = embedder(seqs)                        # [B, L, D] on GPU
    X   = emb.transpose(1, 2).contiguous()      # [B, D, L] — Conv1d format
    y   = y.to(X.device, non_blocking=True)     # move labels to GPU
    return X, y


#CNN classifier
class SimpleCNNClassifier(nn.Module):
    """
    3-layer 1D CNN that reads per-position embeddings and predicts
    one of 3 structural classes (unstructured / intermediate / structured).

    Architecture:
      [B, D, L]
        -> Conv1d(D->256, k=7) + ReLU   # detects 7-nt local patterns
        -> Conv1d(256->128, k=5) + ReLU  # combines nearby patterns
        -> Conv1d(128->64,  k=3) + ReLU  # refines
        -> AdaptiveAvgPool1d(1)           # collapses variable sequence length -> fixed size. CHANGE THIS
        -> Flatten -> Linear(64->3)       # 3 output logits, one per class

    Output: raw logits [B, 3]. CrossEntropyLoss applies softmax internally,
    so no softmax here during training.

    Parameters:
    d_model : int
        Embedding dimension from RiNALMo (480 for micro, 1280 for giga).
    num_classes : int
        Number of output classes (default 3).

    """

    def __init__(self, d_model: int, num_classes: int = 3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(d_model, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          #  [B, 64, 1]
            nn.Flatten(),                      #  [B, 64]
            nn.Linear(64, num_classes),        #  [B, 3]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feat(x))  # -> [B, 3] logits


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

    Class names: 0=unstructured, 1=intermediate, 2=structured

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

def train_one_epoch(model, loader, embedder, opt, scaler, device, use_amp) -> Tuple[float, float, float, float]:
    """
    Train the CNN classifier for one epoch.
    For each batch:
        1. cpu_collate returns raw sequences + labels (from worker processes)
        2. embed_batch runs RiNALMo on GPU (main process) to get embeddings
        3. Forward pass through CNN to logits
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
    (avg_loss, accuracy, elapsed_seconds, sequences_per_second)
    """

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    for seqs, y_cpu in loader:
        # GPU embedding
        X, y = embed_batch(seqs, y_cpu, embedder)

        opt.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(X)           # [B, 3]
                loss   = loss_fn(logits, y) # y is [B] int64 class labels
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()

        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
        n_seqs += bs

    elapsed      = time.perf_counter() - t_start
    n            = sum(t.shape[0] for t in all_labels)
    avg_loss     = total_loss / max(n, 1)
    acc          = accuracy(torch.cat(all_logits), torch.cat(all_labels))
    seqs_per_sec = n_seqs / max(elapsed, 1e-6)
    return avg_loss, acc, elapsed, seqs_per_sec


@torch.no_grad() # No gradient calculation as this is a validation function
def eval_one_epoch(model, loader, embedder, device) -> Tuple[float, float, List[float], float, float]:
    """
    Evaluates the CNN classifier on the validation set.
    Returns:
    (avg_loss, overall_accuracy, per_class_accuracies, elapsed_seconds, sequences_per_second)
    """
    import time
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    n_seqs  = 0
    t_start = time.perf_counter()

    for seqs, y_cpu in loader:
        X, y   = embed_batch(seqs, y_cpu, embedder)
        logits = model(X)
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
    print(f"device: {device}  |  model_size: {MODEL_SIZE}  |  batch: {BATCH_SIZE}  |  subset: {SUBSET_FRAC*100:.0f}%")

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
    ds = BacterialScoresDataset(
        specs=specs, seq_col="Seq", target_col="Score",
        keep_meta=False, dropna=True, shuffle=True, seed=SEED,
    )
    
    print(f"total sequences: {len(ds):,}")

    # Fixed seed for separating out the val set - keeps it cleanly seperate
    n_val   = int(len(ds) * VAL_FRAC)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    # IMPORTANT: use positional indices 0..n_train-1, NOT train_ds.indices.
    # train_ds.indices contains indices into the underlying 13M-row dataset,
    # but SubsetRandomSampler passes its values as positions into train_ds itself,
    # which only has positions 0..n_train-1. Using raw dataset indices causes IndexError.
    train_positions = np.arange(n_train)
    print(f"train: {n_train:,}  val: {n_val:,}  "
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

    # Chech embedding dimension
    sample_seqs = [ds[i][0] for i in range(4)]  # 4 raw sequences
    with torch.no_grad():
        sample_emb = embedder(sample_seqs)       # [4, L, D]
    d_model = sample_emb.shape[2]
    L       = sample_emb.shape[1]
    print(f"d_model: {d_model}  L (sample): {L}")
    del sample_emb

    # Initialise model, optimiser and scheduler
    model   = SimpleCNNClassifier(d_model=d_model, num_classes=NUM_CLASSES).to(device)
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


    # Training loop
    for ep in range(start_epoch, end_epoch):

        # Fresh random subset of training data for this epoch
        sampler      = epoch_sampler(train_positions, SUBSET_FRAC, rng)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False,
            collate_fn=cpu_collate, drop_last=True,
        )

        tr_loss, tr_acc, tr_secs, tr_sps = train_one_epoch(model, train_loader, embedder, opt, scaler, device, USE_AMP)
        va_loss, va_acc, va_cls_accs, va_secs, va_sps = eval_one_epoch(model, val_loader, embedder, device)
        scheduler.step()

        epoch_secs = tr_secs + va_secs
        # Extrapolate to full-dataset epoch: scale train time by 1/SUBSET_FRAC,
        # val time stays the same (val set is always fully evaluated).
        full_tr_est        = tr_secs / max(SUBSET_FRAC, 1e-6)
        full_epoch_est_hrs = (full_tr_est + va_secs) / 3600

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {ep:02d} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        # Per-class accuracy: detect if model is ignoring minority classes
        cls_str = " | ".join(
            f"{'nan' if np.isnan(a) else f'{a:.3f}'}"
            for a in va_cls_accs
        )
        print(
            f"         per-class val acc "
            f"[unstruct | inter | struct]: {cls_str}"
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

        # Each epoch overwrites 'latest.pt' so disk usage stays small,and the best model is saved separately.
        ckpt = {
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

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
