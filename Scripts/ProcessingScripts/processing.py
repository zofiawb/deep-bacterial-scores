import yaml
from pathlib import Path
import pandas as pd

with open("Setup/paths.yaml") as f:
    paths = yaml.safe_load(f)

DATA_ROOT = Path(paths["data_root"])
RAW = DATA_ROOT / "raw"
COMBINED = DATA_ROOT / "combined_datasets"
PROCESSED = DATA_ROOT / "processed"


