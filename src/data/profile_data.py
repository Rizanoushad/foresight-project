

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.load_data import load_combined

DOCS_DIR = PROJECT_ROOT / "docs"

# A normal product code is 5 digits, optionally followed by 1-2 letters
# (for example 85123A). Anything else is probably not a real product.
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Za-z]{0,2}$"


def print_title(text):
    """Print a section heading so the output is easy to read."""
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def show_overview(df):
    """Shape, column names and data types."""
    print_title("1. OVERVIEW")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print()
    print(df.dtypes.to_string())


def show_missing_values(df):
    """How many values are missing in each column."""
    print_title("2. MISSING VALUES")
    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())


def show_duplicates(df):
    """Rows that are exactly identical (ignoring which sheet they came from)."""
    print_title("3. DUPLICATE ROWS")
    data_columns = [c for c in df.columns if c != "source_sheet"]
    is_duplicate = df.duplicated(subset=data_columns, keep="first")
    print(f"Exact duplicate rows: {is_duplicate.sum():,}")

    # Rows that are duplicated inside the same sheet only.
    same_sheet_duplicates = df.duplicated(
        subset=data_columns + ["source_sheet"], keep="first"
    ).sum()
    print(f"  of which inside the same sheet: {same_sheet_duplicates:,}")
    print(f"  of which repeated across sheets: "
          f"{is_duplicate.sum() - same_sheet_duplicates:,}")


def show_sheet_overlap(df):
    """The two sheets cover overlapping dates. Measure the overlap."""
    print_title("4. OVERLAP BETWEEN THE TWO SHEETS")
    sheet_names = sorted(df["source_sheet"].unique())
    first = df[df["source_sheet"] == sheet_names[0]]
    second = df[df["source_sheet"] == sheet_names[1]]

    print(first.groupby("source_sheet")["InvoiceDate"].agg(["min", "max"]).to_string())
    print(second.groupby("source_sheet")["InvoiceDate"].agg(["min", "max"]).to_string())

    overlap_start = second["InvoiceDate"].min().normalize()
    overlap_end = first["InvoiceDate"].max()
    first_overlap = first[first["InvoiceDate"] >= overlap_start]
    second_overlap = second[second["InvoiceDate"] <= overlap_end]
    shared_invoices = set(first_overlap["Invoice"]) & set(second_overlap["Invoice"])

    print()
    print(f"Overlap window: {overlap_start.date()} to {overlap_end.date()}")
    print(f"Rows in overlap, first sheet:  {len(first_overlap):,}")
    print(f"Rows in overlap, second sheet: {len(second_overlap):,}")
    print(f"Invoices present in both sheets: {len(shared_invoices):,}")


def show_unique_counts(df):
    """Number of different values in each column (cardinality)."""
    print_title("5. UNIQUE VALUES PER COLUMN")
    print(df.nunique().to_string())


def show_numeric_summary(df):
    """Basic statistics for Quantity and Price."""
    print_title("6. NUMERIC SUMMARY (Quantity, Price)")
    summary = df[["Quantity", "Price"]].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    )
    print(summary.round(2).to_string())


def show_invalid_values(df):
    """Zero or negative quantities and prices."""
    print_title("7. ZERO OR NEGATIVE VALUES")
    invoice_text = df["Invoice"].astype(str)
    is_cancellation = invoice_text.str.startswith("C")

    print(f"Quantity < 0:  {(df['Quantity'] < 0).sum():,}")
    print(f"Quantity == 0: {(df['Quantity'] == 0).sum():,}")
    print(f"Price < 0:     {(df['Price'] < 0).sum():,}")
    print(f"Price == 0:    {(df['Price'] == 0).sum():,}")
    print()
    print(f"Invoices starting with 'C' (cancellations): {is_cancellation.sum():,}")
    print(f"  cancellations with Quantity > 0: "
          f"{(is_cancellation & (df['Quantity'] > 0)).sum():,}")
    print(f"Negative Quantity but NOT a cancellation: "
          f"{(~is_cancellation & (df['Quantity'] < 0)).sum():,}")
    print()
    print("Rows with negative Price:")
    print(df.loc[df["Price"] < 0, ["Invoice", "StockCode", "Description",
                                   "Quantity", "Price"]].to_string())


def show_stock_code_problems(df):
    """Codes that are not normal products, and code/description inconsistencies."""
    print_title("8. STOCK CODE AND DESCRIPTION QUALITY")
    codes = df["StockCode"].astype(str)

    is_normal = codes.str.match(PRODUCT_CODE_PATTERN)
    odd_codes = codes[~is_normal]
    print(f"Rows with a non-standard StockCode: {len(odd_codes):,} "
          f"({odd_codes.nunique()} distinct codes)")
    print("Most common non-standard codes:")
    print(odd_codes.value_counts().head(15).to_string())

    # Codes that differ only by upper/lower case.
    codes_by_upper = codes.groupby(codes.str.upper()).nunique()
    print()
    print(f"StockCodes that differ only by letter case: {(codes_by_upper > 1).sum():,}")

    # Codes with more than one description.
    descriptions_per_code = df.groupby("StockCode")["Description"].nunique()
    print(f"StockCodes with more than 1 description: "
          f"{(descriptions_per_code > 1).sum():,} of {len(descriptions_per_code):,}")

    # Descriptions with extra spaces at the start or end.
    text = df["Description"].dropna().astype(str)
    print(f"Descriptions with extra leading/trailing spaces: "
          f"{(text != text.str.strip()).sum():,}")


def show_date_checks(df):
    """Date range, trading days and closed days."""
    print_title("9. DATES")
    dates = df["InvoiceDate"].dt.normalize()
    all_days = pd.date_range(dates.min(), dates.max())
    trading_days = pd.DatetimeIndex(dates.unique())
    missing_days = all_days.difference(trading_days)

    print(f"First date: {df['InvoiceDate'].min()}")
    print(f"Last date:  {df['InvoiceDate'].max()}")
    print(f"Calendar days in range: {len(all_days):,}")
    print(f"Days with at least one sale: {len(trading_days):,}")
    print(f"Days with no sales at all: {len(missing_days):,}")
    print()
    print("Rows by day of week (Monday=0 ... Sunday=6):")
    print(df["InvoiceDate"].dt.dayofweek.value_counts().sort_index().to_string())


# Plain-English meaning of each column. Written after inspecting the data.
COLUMN_MEANINGS = {
    "Invoice": "Invoice number. A leading 'C' marks a cancellation; a leading "
               "'A' marks an accounting adjustment.",
    "StockCode": "Product code (the SKU). Normally 5 digits plus optional "
                 "letters. Some codes are fees or manual entries, not products.",
    "Description": "Product name as free text. Not always consistent for the "
                   "same StockCode.",
    "Quantity": "Units on this invoice line. Negative for cancellations and "
                "stock write-offs.",
    "InvoiceDate": "Date and time of the invoice.",
    "Price": "Unit price. The dataset documentation says pounds sterling; "
             "the file itself does not state the currency.",
    "Customer ID": "Customer identifier. Stored as a decimal number because "
                   "some values are missing.",
    "Country": "Country of the customer.",
    "source_sheet": "Added by FORESIGHT: which Excel sheet the row came from.",
}


def write_data_dictionary(df):
    """Write docs/data_dictionary.md using the real numbers from the data."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DOCS_DIR / "data_dictionary.md"

    lines = [
        "# FORESIGHT - Data Dictionary",
        "",
        f"Rows: {len(df):,}  |  Columns: {df.shape[1]}",
        "",
        "| Column | Type | Missing | Unique | Example | Meaning |",
        "|---|---|---|---|---|---|",
    ]
    for column in df.columns:
        example = df[column].dropna().iloc[0]
        lines.append(
            f"| {column} | {df[column].dtype} | {df[column].isna().sum():,} "
            f"| {df[column].nunique():,} | {example} "
            f"| {COLUMN_MEANINGS.get(column, '')} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_title("10. DATA DICTIONARY")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    data = load_combined()
    show_overview(data)
    show_missing_values(data)
    show_duplicates(data)
    show_sheet_overlap(data)
    show_unique_counts(data)
    show_numeric_summary(data)
    show_invalid_values(data)
    show_stock_code_problems(data)
    show_date_checks(data)
    write_data_dictionary(data)