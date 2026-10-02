"""
clean_data.py
Stage 2: clean the raw data and label every row with what it really is.

Nothing is silently thrown away. After removing true duplicates, every row gets
a 'line_type' label. Only rows labelled 'sale' count as demand.

Run with:   python -m src.preprocessing.clean_data

Outputs (in data/processed/):
  clean_lines.pkl    all de-duplicated rows, each with a line_type label
  sales_lines.pkl    only the rows labelled 'sale' (this is our demand data)
  product_names.csv  one clean product name per StockCode
"""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.load_data import load_combined

# The 8 original data columns (everything except the 'source_sheet' we added).
DATA_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
                "InvoiceDate", "Price", "Customer ID", "Country"]

# Codes that are fees, postage, manual entries, adjustments, vouchers or tests.
# They are NOT products customers bought, so they are not product demand.
# (Codes starting with DCGS are real products and are deliberately NOT listed.)
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "C3", "D", "S", "B", "BANK CHARGES",
    "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "TEST001", "TEST002", "GIFT",
}
NON_PRODUCT_PREFIXES = ("GIFT_0001",)

# An order that is reversed by an identical cancellation within this many hours
# is treated as an order mistake, not real demand.
REVERSAL_WINDOW_HOURS = 24


def report(step, rows_before, rows_after):
    """Print how many rows a cleaning step removed."""
    print(f"{step:<55} removed {rows_before - rows_after:>9,}   "
          f"rows left {rows_after:>10,}")


def remove_sheet_overlap(df):
    """
    The two sheets both contain 2010-12-01 to 2010-12-09. First PROVE the
    overlap is an exact copy, then drop it from the second sheet.
    """
    sheet_names = sorted(df["source_sheet"].unique())
    first = df[df["source_sheet"] == sheet_names[0]]
    second = df[df["source_sheet"] == sheet_names[1]]

    overlap_end = first["InvoiceDate"].max()
    overlap_start = second["InvoiceDate"].min().normalize()
    first_overlap = first[first["InvoiceDate"] >= overlap_start]
    second_overlap = second[second["InvoiceDate"] <= overlap_end]

    # Compare the two overlap blocks row by row (order does not matter).
    hash_first = np.sort(pd.util.hash_pandas_object(
        first_overlap[DATA_COLUMNS], index=False).to_numpy())
    hash_second = np.sort(pd.util.hash_pandas_object(
        second_overlap[DATA_COLUMNS], index=False).to_numpy())
    if not np.array_equal(hash_first, hash_second):
        raise ValueError("The overlap is NOT an exact copy. Investigate before cleaning.")
    print(f"Verified: the {len(second_overlap):,} overlap rows are identical in both sheets.")

    keep = ~((df["source_sheet"] == sheet_names[1]) & (df["InvoiceDate"] <= overlap_end))
    return df[keep].copy()


def standardise_text(df):
    """Fix inconsistent codes and descriptions."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    # Description is free text; some cells are numbers or empty.
    # Convert to text and remove spaces at the start and end. Keep empty as None.
    df["Description"] = df["Description"].map(
        lambda text: str(text).strip() if pd.notna(text) else None)
    return df


def label_line_types(df):
    """
    Give every row ONE label. The order of the rules below matters:
    the first rule that matches wins.
    """
    code = df["StockCode"]
    is_non_product = code.isin(NON_PRODUCT_CODES) | code.str.startswith(NON_PRODUCT_PREFIXES)
    is_cancel_invoice = df["Invoice"].astype(str).str.startswith("C")

    labels = np.select(
        [
            is_non_product,                                   # fees, postage, tests
            df["Price"] < 0,                                  # impossible price
            (df["Price"] == 0) & (df["Quantity"] < 0),        # stock written off
            (df["Price"] == 0) & (df["Quantity"] > 0),        # free / unpriced goods
            is_cancel_invoice,                                # customer cancellation or return
            df["Quantity"] < 0,                               # any other negative
        ],
        ["non_product", "invalid_price", "stock_adjustment",
         "zero_price_positive", "cancellation_return", "other_negative"],
        default="sale",
    )
    df = df.copy()
    df["line_type"] = labels
    return df


def flag_reversed_orders(df):
    """
    Relabel 'sale' rows as 'reversed_order' when the same customer cancelled
    exactly the same product and quantity within REVERSAL_WINDOW_HOURS.
    """
    columns = ["Customer ID", "StockCode", "Quantity", "InvoiceDate"]

    sales = df.loc[(df["line_type"] == "sale") & df["Customer ID"].notna(), columns].copy()
    sales["sale_row"] = sales.index
    sales = sales.sort_values("InvoiceDate")

    cancels = df.loc[(df["line_type"] == "cancellation_return")
                     & df["Customer ID"].notna(), columns].copy()
    cancels["Quantity"] = -cancels["Quantity"]          # make it comparable
    cancels = cancels.sort_values("InvoiceDate")

    # For each cancellation, find the latest earlier sale with the same
    # customer, product and quantity, no more than 24 hours before.
    matched = pd.merge_asof(
        cancels, sales, on="InvoiceDate",
        by=["Customer ID", "StockCode", "Quantity"],
        direction="backward",
        tolerance=pd.Timedelta(hours=REVERSAL_WINDOW_HOURS),
    )
    reversed_rows = matched["sale_row"].dropna().astype(int).unique()

    df = df.copy()
    df.loc[reversed_rows, "line_type"] = "reversed_order"
    return df


def build_product_names(sales):
    """One clean name per StockCode: its most common description on real sales."""
    names = (sales.dropna(subset=["Description"])
                  .groupby("StockCode")["Description"]
                  .agg(lambda text: text.value_counts().index[0]))
    return names.rename("ProductName").reset_index()


def clean():
    raw = load_combined()
    print(f"Raw rows loaded: {len(raw):,}\n")

    # Step 1: overlap between sheets
    before = len(raw)
    df = remove_sheet_overlap(raw)
    report("1. Remove sheet overlap (exact copy)", before, len(df))

    # Step 2: standardise text, then drop exact duplicate rows
    df = standardise_text(df)
    before = len(df)
    df = df.drop_duplicates(subset=DATA_COLUMNS, keep="first").reset_index(drop=True)
    report("2. Remove exact duplicate rows", before, len(df))

    # Step 3: label every row
    df = label_line_types(df)
    df = flag_reversed_orders(df)

    return df


def print_summary(df):
    """Show how many rows and units fall into each category."""
    print("\nWhat each row is (after cleaning):")
    summary = df.groupby("line_type").agg(
        rows=("Quantity", "size"),
        units=("Quantity", "sum"),
    ).sort_values("rows", ascending=False)
    summary["percent_rows"] = (100 * summary["rows"] / len(df)).round(2)
    print(summary.to_string())

    sales = df[df["line_type"] == "sale"]
    print(f"\nDemand data (line_type = 'sale'):")
    print(f"  Rows:           {len(sales):,}")
    print(f"  Distinct SKUs:  {sales['StockCode'].nunique():,}")
    print(f"  Total units:    {sales['Quantity'].sum():,}")
    print(f"  Date range:     {sales['InvoiceDate'].min()}  to  {sales['InvoiceDate'].max()}")
    print(f"  Largest single line: {sales['Quantity'].max():,} units")


if __name__ == "__main__":
    cleaned = clean()
    print_summary(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sales_only = cleaned[cleaned["line_type"] == "sale"].copy()
    cleaned.to_pickle(PROCESSED_DIR / "clean_lines.pkl")
    sales_only.to_pickle(PROCESSED_DIR / "sales_lines.pkl")
    build_product_names(sales_only).to_csv(PROCESSED_DIR / "product_names.csv", index=False)
    print(f"\nSaved 3 files to: {PROCESSED_DIR}")