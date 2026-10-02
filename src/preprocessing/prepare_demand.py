"""
FORESIGHT - Demand Forecasting Preprocessing

Stage 3: Convert cleaned transaction-level sales data into a
machine-learning-ready SKU-level daily demand dataset.

Input:
    data/processed/sales_lines.pkl

Outputs:
    data/processed/daily_demand.csv
    data/processed/model_data.csv
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

Run:
    python -m src.preprocessing.prepare_demand
"""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = PROCESSED_DIR / "sales_lines.pkl"

DAILY_DEMAND_FILE = PROCESSED_DIR / "daily_demand.csv"
MODEL_DATA_FILE = PROCESSED_DIR / "model_data.csv"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

# Historical demand used by the model
LAG_DAYS = [1, 7, 14, 28]

# Rolling historical demand windows
ROLLING_WINDOWS = [7, 14, 28]

# We need at least 28 previous days of history
MIN_HISTORY_DAYS = 28

# Chronological split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_sales_data():
    """
    Load the cleaned sales dataset created by clean_data.py.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT_FILE}\n\n"
            "Run the cleaning stage first:\n"
            "python -m src.preprocessing.clean_data"
        )

    df = pd.read_pickle(INPUT_FILE)

    print("=" * 70)
    print("FORESIGHT - PREPROCESSING")
    print("=" * 70)

    print(f"\nInput file : {INPUT_FILE}")
    print(f"Rows loaded: {len(df):,}")

    return df


# ============================================================
# 2. VALIDATE DATA
# ============================================================

def validate_data(df):
    """
    Validate the cleaned sales dataset.
    """

    required_columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "line_type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # sales_lines.pkl should already contain only sales.
    non_sales = (df["line_type"] != "sale").sum()

    if non_sales > 0:
        print(
            f"\nWARNING: Found {non_sales:,} non-sale rows."
        )

        df = df[
            df["line_type"] == "sale"
        ].copy()

    # Convert date
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    invalid_dates = df["InvoiceDate"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates:,} invalid InvoiceDate values."
        )

    # Remove missing SKU
    missing_sku = df["StockCode"].isna().sum()

    if missing_sku > 0:
        print(
            f"Removing {missing_sku:,} rows with missing StockCode."
        )

        df = df.dropna(
            subset=["StockCode"]
        )

    # Standardize SKU
    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert quantity to numeric
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    invalid_quantity = df["Quantity"].isna().sum()

    if invalid_quantity > 0:
        print(
            f"Removing {invalid_quantity:,} rows with invalid Quantity."
        )

        df = df.dropna(
            subset=["Quantity"]
        )

    print("\nData validation: PASSED")

    return df


# ============================================================
# 3. DATE PREPARATION
# ============================================================

def prepare_dates(df):
    """
    Convert transaction timestamps into calendar dates.
    """

    df = df.copy()

    df["Date"] = (
        df["InvoiceDate"]
        .dt.normalize()
    )

    print("\nDate information:")
    print(
        f"Start date : {df['Date'].min().date()}"
    )
    print(
        f"End date   : {df['Date'].max().date()}"
    )
    print(
        f"Unique days: {df['Date'].nunique():,}"
    )
    print(
        f"Unique SKUs: {df['StockCode'].nunique():,}"
    )

    return df


# ============================================================
# 4. CREATE DAILY SKU DEMAND
# ============================================================

def create_daily_demand(df):
    """
    Aggregate transaction-level sales into daily demand per SKU.

    Example:

        Date        SKU      Quantity
        2010-01-01  A        5
        2010-01-01  A        10
        2010-01-01  A        3

    becomes:

        Date        SKU      Demand
        2010-01-01  A        18
    """

    daily = (
        df.groupby(
            ["Date", "StockCode"],
            as_index=False
        )
        .agg(
            Demand=("Quantity", "sum"),
            Transactions=("Invoice", "nunique"),
            AveragePrice=("Price", "mean")
        )
    )

    daily = daily.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    print("\nDaily SKU demand created:")
    print(
        f"Rows: {len(daily):,}"
    )
    print(
        f"SKUs: {daily['StockCode'].nunique():,}"
    )

    print("\nSample:")
    print(
        daily.head(10).to_string(index=False)
    )

    return daily


# ============================================================
# 5. CREATE COMPLETE SKU-DATE CALENDAR
# ============================================================

def create_complete_calendar(daily):
    """
    Create a daily timeline for every SKU between its first
    and last observed sale.

    If there was no recorded sale for a SKU on a date inside
    its active period, demand is treated as zero.

    We do NOT create dates before a SKU's first appearance
    or after its last observed date.
    """

    print("\nCreating complete SKU-date calendar...")

    all_frames = []

    for sku, group in daily.groupby(
        "StockCode",
        sort=False
    ):

        start_date = group["Date"].min()
        end_date = group["Date"].max()

        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D"
        )

        sku_calendar = pd.DataFrame({
            "Date": date_range,
            "StockCode": sku
        })

        sku_calendar = sku_calendar.merge(
            group,
            on=["Date", "StockCode"],
            how="left"
        )

        all_frames.append(
            sku_calendar
        )

    complete = pd.concat(
        all_frames,
        ignore_index=True
    )

    # No recorded transaction during an active SKU period
    # is treated as zero observed demand.
    complete["Demand"] = (
        complete["Demand"]
        .fillna(0)
    )

    complete["Transactions"] = (
        complete["Transactions"]
        .fillna(0)
        .astype(int)
    )

    complete = complete.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    print(
        f"Complete calendar rows: {len(complete):,}"
    )

    return complete


# ============================================================
# 6. CREATE TIME FEATURES
# ============================================================

def create_time_features(df):
    """
    Create calendar features used by the forecasting model.
    """

    df = df.copy()

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Quarter"] = df["Date"].dt.quarter

    df["DayOfMonth"] = df["Date"].dt.day

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["WeekOfYear"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Saturday and Sunday = 1
    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # Cyclical representation of month
    df["MonthSin"] = (
        np.sin(
            2 * np.pi * df["Month"] / 12
        )
    )

    df["MonthCos"] = (
        np.cos(
            2 * np.pi * df["Month"] / 12
        )
    )

    return df


# ============================================================
# 7. CREATE LAG FEATURES
# ============================================================

def create_lag_features(df):
    """
    Create historical demand features.

    Lag 1  = yesterday's demand
    Lag 7  = demand seven days ago
    Lag 14 = demand fourteen days ago
    Lag 28 = demand twenty-eight days ago

    Only previous demand is used.
    """

    df = df.copy()

    df = df.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    grouped = (
        df.groupby("StockCode")["Demand"]
    )

    for lag in LAG_DAYS:

        df[f"Lag_{lag}"] = (
            grouped.shift(lag)
        )

    return df


# ============================================================
# 8. CREATE ROLLING FEATURES
# ============================================================

def create_rolling_features(df):
    """
    Create rolling historical demand statistics.

    The demand is shifted by one day before calculating
    rolling statistics.

    This prevents today's demand from being used to
    predict today's demand.
    """

    df = df.copy()

    df = df.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    # Yesterday's demand and earlier
    previous_demand = (
        df.groupby("StockCode")["Demand"]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:

        df[f"RollingMean_{window}"] = (
            previous_demand
            .groupby(df["StockCode"])
            .rolling(window)
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"RollingStd_{window}"] = (
            previous_demand
            .groupby(df["StockCode"])
            .rolling(window)
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )

    return df


# ============================================================
# 9. PREPARE MODEL DATA
# ============================================================

def prepare_model_data(df):
    """
    Remove rows where forecasting features cannot yet be
    calculated because there is insufficient historical data.
    """

    df = df.copy()

    feature_columns = []

    # Lag features
    for lag in LAG_DAYS:
        feature_columns.append(
            f"Lag_{lag}"
        )

    # Rolling features
    for window in ROLLING_WINDOWS:

        feature_columns.append(
            f"RollingMean_{window}"
        )

        feature_columns.append(
            f"RollingStd_{window}"
        )

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    removed = before - len(df)

    print("\nFeature history preparation:")
    print(
        f"Rows before : {before:,}"
    )
    print(
        f"Rows removed: {removed:,}"
    )
    print(
        f"Rows left   : {len(df):,}"
    )

    return df


# ============================================================
# 10. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_time_series(df):
    """
    Split the data chronologically.

    IMPORTANT:
    We do not randomly shuffle time-series data.

    Earlier dates  -> Training
    Middle dates   -> Validation
    Latest dates   -> Testing
    """

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    unique_dates = sorted(
        df["Date"].unique()
    )

    total_dates = len(unique_dates)

    train_end_index = int(
        total_dates * TRAIN_RATIO
    )

    validation_end_index = int(
        total_dates *
        (
            TRAIN_RATIO +
            VALIDATION_RATIO
        )
    )

    train_end_date = unique_dates[
        train_end_index - 1
    ]

    validation_end_date = unique_dates[
        validation_end_index - 1
    ]

    train = df[
        df["Date"] <= train_end_date
    ].copy()

    validation = df[
        (df["Date"] > train_end_date)
        &
        (df["Date"] <= validation_end_date)
    ].copy()

    test = df[
        df["Date"] > validation_end_date
    ].copy()

    print("\n" + "=" * 70)
    print("TIME-BASED DATA SPLIT")
    print("=" * 70)

    print(
        f"\nTraining:"
        f"\n  Dates: {train['Date'].min().date()} "
        f"→ {train['Date'].max().date()}"
        f"\n  Rows : {len(train):,}"
    )

    print(
        f"\nValidation:"
        f"\n  Dates: {validation['Date'].min().date()} "
        f"→ {validation['Date'].max().date()}"
        f"\n  Rows : {len(validation):,}"
    )

    print(
        f"\nTesting:"
        f"\n  Dates: {test['Date'].min().date()} "
        f"→ {test['Date'].max().date()}"
        f"\n  Rows : {len(test):,}"
    )

    return train, validation, test


# ============================================================
# 11. SAVE ALL OUTPUT FILES
# ============================================================

def save_data(
    daily_demand,
    model_data,
    train,
    validation,
    test
):
    """
    Save all preprocessing outputs.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily_demand.to_csv(
        DAILY_DEMAND_FILE,
        index=False
    )

    model_data.to_csv(
        MODEL_DATA_FILE,
        index=False
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"✓ {DAILY_DEMAND_FILE}"
    )

    print(
        f"✓ {MODEL_DATA_FILE}"
    )

    print(
        f"✓ {TRAIN_FILE}"
    )

    print(
        f"✓ {VALIDATION_FILE}"
    )

    print(
        f"✓ {TEST_FILE}"
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

def print_final_summary(df):
    """
    Print a summary of the final model-ready dataset.
    """

    print("\n" + "=" * 70)
    print("FINAL PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"\nRows       : {len(df):,}"
    )

    print(
        f"SKUs       : {df['StockCode'].nunique():,}"
    )

    print(
        f"Dates      : {df['Date'].nunique():,}"
    )

    print(
        f"Start date : {df['Date'].min().date()}"
    )

    print(
        f"End date   : {df['Date'].max().date()}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # 1. Load cleaned sales
    df = load_sales_data()

    # 2. Validate data
    df = validate_data(df)

    # 3. Prepare dates
    df = prepare_dates(df)

    # 4. Convert transactions into daily SKU demand
    daily_demand = create_daily_demand(df)

    # 5. Create complete SKU-date timeline
    daily_demand = create_complete_calendar(
        daily_demand
    )

    # 6. Create calendar features
    daily_demand = create_time_features(
        daily_demand
    )

    # 7. Create lag features
    daily_demand = create_lag_features(
        daily_demand
    )

    # 8. Create rolling features
    daily_demand = create_rolling_features(
        daily_demand
    )

    # 9. Remove rows without sufficient history
    model_data = prepare_model_data(
        daily_demand
    )

    # 10. Chronological split
    train, validation, test = split_time_series(
        model_data
    )

    # 11. Save all datasets
    save_data(
        daily_demand=daily_demand,
        model_data=model_data,
        train=train,
        validation=validation,
        test=test
    )

    # 12. Final summary
    print_final_summary(
        model_data
    )

    print(
        "\n✓ PREPROCESSING COMPLETED SUCCESSFULLY"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()