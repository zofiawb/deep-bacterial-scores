import yaml
import csv
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import re
import subprocess
import ast
from datetime import datetime
import traceback




"""
This script builds per-species datasets. From the RASP filenames, it pulls out 
"Seq" - a 101 nt window of the refseq around the given score
"Species" - the species of bacteria 
"Method" - DMS, SHAPE-Map, SHAPE-Seq etc
"Reagent" - DMS or SHAPE reagent 
"Temp" - if an explicit experimental condition
"Condition" - in vivo, in vitro, ex vivo
"Specificity" - targeted or transcriptome wide
"Score" - the structure score
"Coord" - the genomic coordinate
"Study ID" - a numerical identifier for the study, used to split the database
"Paper" - the journal name and year
"""


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
#Scripts/ProcessingScripts -> Scripts -> project root

PATHS_FILE = PROJECT_ROOT / "Setup" / "paths.yaml"

with PATHS_FILE.open() as f:
    paths = yaml.safe_load(f)

DATA_ROOT = Path(paths["data_root"])
RAW = DATA_ROOT / "raw"
COMBINED = DATA_ROOT / "combined_datasets"
PROCESSED = DATA_ROOT / "processed"

#Checking that the path to the data exists
assert DATA_ROOT.exists(), f"DATA_ROOT does not exist: {DATA_ROOT}"
assert RAW.exists(), f"RAW does not exist: {RAW}"
assert COMBINED.exists(), f"COMBINED does not exist: {COMBINED}"
assert PROCESSED.exists(), f"PROCESSED does not exist: {PROCESSED}"

#Log file of progress
LOG_FILE = PROCESSED / "processing_log.txt"

def log_message(msg: str):
    timestamp = datetime.now().isoformat(timespec="seconds")
    with LOG_FILE.open("a") as f:
        f.write(f"[{timestamp}] {msg}\n")
        f.flush()


#Helper functions for parsing from filenames

#Canonical species labels
SPECIES_PATTERNS = {

    "s_enterica": [("senterica",)],
    "e_coli": [("ecoli",), ("e", "coli")], 
    "b_cereus": [("bcereus",)],
    "b_subtilis": [("bsubtilis",)],
    "p_putida": [("pputida",)],
    "synechococcus": [("synechococcus",)],
    "y_pseudotuberculosis": [("y", "pseudotuberculosis"), ("y_pseudotuberculosis",)],  
}

YEAR_RE = re.compile(r"^(19|20)\d{2}$") #to check if a year is correct - matches the start and then check two digits after

def split_parts(colname: str):
    #splits colname into tokens
    return str(colname).strip().split("_")

def find_year_index(parts):
    #Finds where the year starts in the colname
    for i, p in enumerate(parts):
        if YEAR_RE.match(p):
            return i
    return None

def find_species_span(parts):
    """
    This function find where the species is listed in the filename
    Returns (canonical_species, start_idx, end_idx_exclusive) or (pd.NA, None, None)
    """
    lowered = [p.lower() for p in parts]

    # scan left-to-right; pick the earliest match; if ties, prefer longer species token seq
    best = None  # (start, -len, canonical, end)
    for canonical, seqs in SPECIES_PATTERNS.items():
        for seq in seqs:
            # seq may be a tuple of tokens; match against lowered tokens
            L = len(seq)
            for i in range(0, len(lowered) - L + 1):
                window = tuple(lowered[i:i+L])
                if window == seq:
                    cand = (i, -L, canonical, i+L)
                    if best is None or cand < best:
                        best = cand

    if best is None:
        return pd.NA, None, None
    _, _, canonical, end = best
    start = best[0]
    return canonical, start, end

def parse_method(colname: str):
    parts = split_parts(colname)
    species, s0, _ = find_species_span(parts)
    if s0 is None:
        # fallback: first token only
        return parts[0] if parts else pd.NA
    method = "_".join(parts[:s0])
    return method if method else pd.NA

def parse_species(colname: str):
    parts = split_parts(colname)
    species, _, _ = find_species_span(parts)
    return species

def parse_journal_and_year(colname: str):
    parts = split_parts(colname)
    yidx = find_year_index(parts)
    species, s0, s1 = find_species_span(parts)

    if yidx is None or s0 is None:
        return pd.NA, pd.NA

    journal = "_".join(parts[s1:yidx])  # may contain underscores, triple underscores, etc.
    year = parts[yidx]
    return f"{journal} ({year})"

def parse_temp(colname: str):
    """
    Matches: 42C, 42c, 42degree, 42 degree, etc.
    Avoids matching things like K150 (since it requires C/degree or word boundary context).
    """
    s = str(colname).lower()
    m = re.search(r"\b(25|30|37|42|80|95)\s*(c|°c|degree|degrees)\b", s)
    if m:
        return int(m.group(1))
    # allow bare tokens like "_42C_" where C attaches
    m = re.search(r"\b(25|30|37|42|80|95)c\b", s)
    return int(m.group(1)) if m else pd.NA

def parse_condition(colname: str):
    s = str(colname).lower()
    if "in_vitro_transcribed" in s:
        return "in_vitro"
    if "in_vitro" in s or "invitro" in s:
        return "in_vitro"
    if "in_vivo" in s or "invivo" in s:
        return "in_vivo"
    if "ex_vivo" in s or "exvivo" in s:
        return "ex_vivo"
    if "incell" in s:
        return "in_vivo"
    if "cellfree" in s:
        return "in_vitro"
    return pd.NA

def parse_specificity(colname: str):
    s = str(colname).lower()
    if "transcriptome-wide" in s or "transcriptomewide" in s:
        return "transcriptome-wide"
    if "targeted" in s:
        return "targeted"
    return pd.NA

def parse_reagent(colname: str):
    
    s = str(colname).lower()
    for key in ["dms", "bzcn", "1m7", "1m4", "2a3", "nai", "nic", "b5", "i5", "6a3", "lead(ii)", "hydroxyl_radical"]:
        if key in s:
            return key.upper()
    return pd.NA


#Get fasta window

def faidx_window(pos1_1based: int, window: int, fasta_path: Path, ref_id: str) -> str:
    start = max(1, pos1_1based - window) #bounding by seq len 
    end = pos1_1based + window
    region = f"{ref_id}:{start}-{end}"
    cmd = ["samtools", "faidx", str(fasta_path), region]

    out = subprocess.check_output(cmd, cwd=str(COMBINED), text=True) #executing in correct wd
    #samtools faidx returns FASTA format: header + wrapped sequence lines
    lines = out.splitlines()
    seq = "".join(lines[1:]).strip().upper()
    return seq


#Builder function

def build_dataset_streaming(df, fasta_path, ref_id, window, out_csv: Path, flush_every=10000, study_IDs_dict = dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    #Input df: each cell is (pos1, pos2, score), a string representation of that tuple.
    #Column names encode conditions
    #Output df: one row per score, with parsed metadata.

    fieldnames = [
        "Seq","Species","Method","Reagent","Temp","Condition","Specificity",
        "Score", "Coord", "Study ID", "Paper"
    ]

    # create/overwrite file with header
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    buffer = []
    total_non_na = sum(df[col].notna().sum() for col in df.columns)
    species_label = fasta_path.stem  # "s_enterica" from "s_enterica.fasta"

    with tqdm(total=total_non_na, desc=fasta_path.stem, unit="cell") as pbar:
        for col in df.columns:
            #every column is a different length and is FILLED with Nas. No need for these
            cleaned_col = df[col].dropna()

            temp = parse_temp(col)
            condition = parse_condition(col)
            reagent = parse_reagent(col)
            method = parse_method(col)
            specificity = parse_specificity(col)
            study_name = parse_journal_and_year(col)

            # assigning an identifier
            identifier = (study_name, species_label) #Some studies have multiple species
            if identifier in study_IDs_dict.keys():
                study_id  = study_IDs_dict[identifier]
            else:
                study_id = len(study_IDs_dict) + 1
                study_IDs_dict[identifier] = study_id

            # iterate cells in the column
            for cell in cleaned_col.values:
                try:
                    pos1, pos2, score =  ast.literal_eval(cell)
                except Exception:
                    pos1, pos2, score =  pd.NA, pd.NA, pd.NA

                if pd.isna(pos1):
                    print(f"[WARN] unparsable cell in column '{col}': {repr(cell)}")
                    pbar.update(1)
                    continue

                seq = faidx_window(int(pos1), window, fasta_path, ref_id)

                buffer.append({"Seq": seq,
                    "Species": species_label,
                    "Method": method,
                    "Reagent": reagent,
                    "Temp": temp,
                    "Condition": condition,
                    "Specificity": specificity,
                    "Score": score,
                    "Coord": pos1,
                    "Study ID": study_id,
                    "Paper": study_name})

                if len(buffer) >= flush_every:
                    pd.DataFrame.from_records(buffer).to_csv(out_csv, mode="a", header=False, index=False)
                    buffer.clear()

                pbar.update(1)

    if buffer:
        pd.DataFrame.from_records(buffer).to_csv(out_csv, mode="a", header=False, index=False)

#Reading in dfs

FASTA_LIST = [
    "s_enterica.fasta", "b_cereus.fasta", "b_subtilis.fasta",
    "synechococcus.fasta", "p_putida.fasta", "y_pseudotuberculosis.fasta", "e_coli.fasta"
]
REF_ID_LIST = ["NC_003197.2", "AE017194.1", "NC_000964.3", "BX548020.1", "NC_0002947.4", 
                "CP009792.1", "U00096.2"]
# NOTE: Could not find a NCBI download for Y_pseudotuberculosis refseq NC_010456. Therefore trying a different ref.

"""INPUT_DF_LIST = ["raw_s_enterica.csv", "raw_b_cereus.csv", "raw_b_subtilis.csv", "raw_synechococcus.fasta", 
                 "raw_p_putida.csv", "raw_y_pseudotuberculosis.csv", "raw_e_coli.csv"]"""

WINDOW = 50

study_IDs = {} #creating a numerical ID to keep different studies separate later on

for fasta, ref_id in zip(FASTA_LIST, REF_ID_LIST):
    fasta_path = COMBINED / fasta
    species = fasta_path.stem
    raw_path = COMBINED / f"raw_{species}.csv"
    out_path = PROCESSED / f"{species}_processed.csv"

    try:
        log_message(f"START {species}")

        df_raw = pd.read_csv(raw_path)

        build_dataset_streaming(
            df=df_raw,
            fasta_path=fasta_path,
            ref_id=ref_id,
            window=WINDOW, 
            out_csv=out_path,
            flush_every=10000,
            study_IDs_dict=study_IDs,
            )

        log_message(
            f"DONE {species}, study IDs dict is now {study_IDs}"
        )

    except Exception as e:
        log_message(f"FAIL {species} error={repr(e)}")
        log_message(traceback.format_exc())
        continue

