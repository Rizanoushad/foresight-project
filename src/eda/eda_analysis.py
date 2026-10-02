import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# FORESIGHT - EXPLORATORY DATA ANALYSIS


INPUT_FILE = "data/processed/daily_demand.csv"
OUTPUT_DIR = "reports/eda"


def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)



# 1. LOAD DATA


def load_data():

    print("=" * 70)
    print("FORESIGHT - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows          : {len(df):,}")
    print(f"Columns       : {len(df.columns)}")
    print(f"Unique SKUs   : {df['StockCode'].nunique():,}")
    print(f"Start date    : {df['Date'].min().date()}")
    print(f"End date      : {df['Date'].max().date()}")

    return df



# 2. DATASET OVERVIEW


def dataset_overview(df):

    print("\n" + "=" * 70)
    print("1. DATASET OVERVIEW")
    print("=" * 70)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nBasic statistics:")
    print(df.describe())



# 3. DEMAND SUMMARY


def demand_summary(df):

    print("\n" + "=" * 70)
    print("2. DEMAND SUMMARY")
    print("=" * 70)

    total_demand = df["Demand"].sum()
    average_demand = df["Demand"].mean()
    median_demand = df["Demand"].median()
    max_demand = df["Demand"].max()

    print(f"\nTotal demand       : {total_demand:,.0f}")
    print(f"Average SKU-day    : {average_demand:,.2f}")
    print(f"Median SKU-day     : {median_demand:,.2f}")
    print(f"Maximum SKU-day    : {max_demand:,.0f}")

    zero_demand = (df["Demand"] == 0).sum()

    print(
        f"Zero-demand days   : {zero_demand:,} "
        f"({zero_demand / len(df) * 100:.2f}%)"
    )



# 4. DAILY DEMAND TREND


def daily_demand_trend(df):

    daily = (
        df.groupby("Date")["Demand"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily["Date"],
        daily["Demand"]
    )

    plt.title("Daily Total Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/daily_demand_trend.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: daily_demand_trend.png")



# 5. MONTHLY DEMAND TREND


def monthly_demand_trend(df):

    monthly = (
        df.set_index("Date")
        .resample("ME")["Demand"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        monthly["Date"],
        monthly["Demand"],
        marker="o"
    )

    plt.title("Monthly Total Demand")
    plt.xlabel("Month")
    plt.ylabel("Demand")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/monthly_demand_trend.png",
        dpi=150
    )

    plt.close()

    print("Saved: monthly_demand_trend.png")



# 6. TOP SKUs BY DEMAND


def top_skus_by_demand(df):

    top_skus = (
        df.groupby("StockCode")["Demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n" + "=" * 70)
    print("3. TOP 10 SKUs BY TOTAL DEMAND")
    print("=" * 70)

    print(top_skus)

    plt.figure(figsize=(12, 6))

    top_skus.sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 SKUs by Total Demand")
    plt.xlabel("Total Demand")
    plt.ylabel("SKU")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top_10_skus_demand.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: top_10_skus_demand.png")



# 7. TOP SKUs BY TRANSACTIONS


def top_skus_by_transactions(df):

    top_skus = (
        df.groupby("StockCode")["Transactions"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n" + "=" * 70)
    print("4. TOP 10 SKUs BY TRANSACTIONS")
    print("=" * 70)

    print(top_skus)

    plt.figure(figsize=(12, 6))

    top_skus.sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 SKUs by Transactions")
    plt.xlabel("Number of Transactions")
    plt.ylabel("SKU")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top_10_skus_transactions.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: top_10_skus_transactions.png")



# 8. DEMAND DISTRIBUTION


def demand_distribution(df):

    plt.figure(figsize=(12, 6))

    # Clip extreme values only for visualization
    upper_limit = df["Demand"].quantile(0.99)

    demand_for_plot = df[
        df["Demand"] <= upper_limit
    ]["Demand"]

    plt.hist(
        demand_for_plot,
        bins=50
    )

    plt.title("Demand Distribution (Up to 99th Percentile)")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/demand_distribution.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: demand_distribution.png")



# 9. WEEKDAY VS WEEKEND


def weekday_analysis(df):

    temp = df.copy()

    temp["DayOfWeek"] = temp["Date"].dt.day_name()
    temp["IsWeekend"] = temp["Date"].dt.dayofweek >= 5

    weekday_demand = (
        temp.groupby("DayOfWeek")["Demand"]
        .mean()
        .reindex([
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ])
    )

    print("\n" + "=" * 70)
    print("5. AVERAGE DEMAND BY DAY")
    print("=" * 70)

    print(weekday_demand)

    plt.figure(figsize=(12, 6))

    weekday_demand.plot(
        kind="bar"
    )

    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Demand")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/weekday_demand.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: weekday_demand.png")



# 10. MONTHLY SEASONALITY


def monthly_seasonality(df):

    temp = df.copy()

    temp["Month"] = temp["Date"].dt.month

    monthly = (
        temp.groupby("Month")["Demand"]
        .mean()
    )

    print("\n" + "=" * 70)
    print("6. MONTHLY SEASONALITY")
    print("=" * 70)

    print(monthly)

    plt.figure(figsize=(12, 6))

    monthly.plot(
        kind="bar"
    )

    plt.title("Average Demand by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Demand")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/monthly_seasonality.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: monthly_seasonality.png")



# 11. SKU DEMAND VARIABILITY


def sku_variability(df):

    sku_stats = (
        df.groupby("StockCode")["Demand"]
        .agg(
            MeanDemand="mean",
            StdDemand="std",
            TotalDemand="sum"
        )
    )

    sku_stats["CV"] = (
        sku_stats["StdDemand"] /
        sku_stats["MeanDemand"].replace(0, np.nan)
    )

    print("\n" + "=" * 70)
    print("7. SKU DEMAND VARIABILITY")
    print("=" * 70)

    print("\nMost variable SKUs:")

    print(
        sku_stats["CV"]
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nAverage coefficient of variation:")

    print(
        sku_stats["CV"]
        .replace([np.inf, -np.inf], np.nan)
        .mean()
    )



# 12. CORRELATION ANALYSIS


def correlation_analysis(df):

    numeric_columns = [
        "Demand",
        "Transactions",
        "AveragePrice"
    ]

    correlation = df[numeric_columns].corr()

    print("\n" + "=" * 70)
    print("8. CORRELATION ANALYSIS")
    print("=" * 70)

    print(correlation)

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_matrix.png",
        dpi=150
    )

    plt.close()

    print("\nSaved: correlation_matrix.png")



# 13. BUSINESS INSIGHTS


def generate_insights(df):

    print("\n" + "=" * 70)
    print("9. KEY EDA INSIGHTS")
    print("=" * 70)

    total_demand = df["Demand"].sum()

    sku_demand = (
        df.groupby("StockCode")["Demand"]
        .sum()
        .sort_values(ascending=False)
    )

    top_sku = sku_demand.index[0]

    top_sku_share = (
        sku_demand.iloc[0] /
        total_demand *
        100
    )

    zero_demand_pct = (
        (df["Demand"] == 0).mean() * 100
    )

    print(f"""
1. The dataset contains {df['StockCode'].nunique():,} unique SKUs.

2. Total recorded demand is approximately
   {total_demand:,.0f} units.

3. SKU {top_sku} has the highest total demand.

4. The highest-demand SKU contributes approximately
   {top_sku_share:.2f}% of total demand.

5. Approximately {zero_demand_pct:.2f}% of SKU-day
   observations have zero demand.

6. Demand varies across time, indicating that
   temporal features and forecasting models are relevant.

7. SKU-level demand is highly heterogeneous, meaning
   different products have different demand patterns.

8. These patterns justify using lag and rolling-demand
   features in the forecasting stage.
""")



# MAIN


def main():

    create_output_directory()

    df = load_data()

    dataset_overview(df)

    demand_summary(df)

    daily_demand_trend(df)

    monthly_demand_trend(df)

    top_skus_by_demand(df)

    top_skus_by_transactions(df)

    demand_distribution(df)

    weekday_analysis(df)

    monthly_seasonality(df)

    sku_variability(df)

    correlation_analysis(df)

    generate_insights(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()