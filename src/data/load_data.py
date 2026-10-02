

import sys
import time

import pandas as pd

from src.config import RAW_EXCEL_PATH, COMBINED_RAW_PATH, INTERIM_DIR


def load_raw_excel():
    """Read every sheet of the Excel file and stack them into one DataFrame."""
    if not RAW_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_EXCEL_PATH}\n"
            "Copy online_retail_II.xlsx into the data/raw folder."
        )

    print("Reading Excel file. This is slow (1-5 minutes). Please wait...")

    # Invoice and StockCode contain a mix of numbers and text
    # (for example 'C489449' or '85123A'), so we read them as text.
    sheets = pd.read_excel(
        RAW_EXCEL_PATH,
        sheet_name=None,  # None means: read ALL sheets
        dtype={"Invoice": str, "StockCode": str},
    )

    # Remember which sheet each row came from, then stack the sheets.
    tables = []
    for sheet_name, table in sheets.items():
        table["source_sheet"] = sheet_name
        print(f"  Sheet '{sheet_name}': {len(table):,} rows")
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def save_combined(df):
    """Save the combined table to the interim folder."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(COMBINED_RAW_PATH)
    print(f"Saved fast copy to: {COMBINED_RAW_PATH}")


def load_combined():
    """Load the fast copy. Other scripts will call this function."""
    if not COMBINED_RAW_PATH.exists():
        raise FileNotFoundError(
            "Fast copy not found. Run:  python -m src.data.load_data"
        )
    return pd.read_pickle(COMBINED_RAW_PATH)


if __name__ == "__main__":
    # Pass --force to rebuild the fast copy even if it already exists.
    force = "--force" in sys.argv

    if COMBINED_RAW_PATH.exists() and not force:
        print("Fast copy already exists. Use --force to rebuild it.")
        data = load_combined()
    else:
        start = time.time()
        data = load_raw_excel()
        save_combined(data)
        print(f"Done in {time.time() - start:.0f} seconds.")

    print()
    print("Shape (rows, columns):", data.shape)
    print("Columns:", list(data.columns))
    print()
    print(data.head(3).to_string())