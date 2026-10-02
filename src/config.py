"""
config.py
Central place for file paths and settings used across the FORESIGHT project.
"""

from pathlib import Path

# The project root is the folder that contains "src". We work it out from this
# file's own location, so the project works no matter where it is saved.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Data folders ----
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- Important files ----
RAW_EXCEL_PATH = RAW_DIR / "online_retail_II.xlsx"
COMBINED_RAW_PATH = INTERIM_DIR / "raw_combined.pkl"