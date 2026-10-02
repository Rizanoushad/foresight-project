import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/daily_demand.csv"
OUTPUT_DIR = "reports/visualizations"


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. OVERALL DAILY DEMAND
# ============================================================

def plot_daily_demand(df):

    daily = (
        df.groupby("Date")["Demand"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily["Date"],
        daily["Demand"],
        linewidth=1
    )

    plt.title("FORESIGHT - Daily Demand Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Demand")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/01_daily_demand_trend.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 2. MONTHLY DEMAND
# ============================================================

def plot_monthly_demand(df):

    monthly = (
        df.set_index("Date")
        .resample("ME")["Demand"]
        .sum()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.title("Monthly Demand Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Demand")

    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/02_monthly_demand.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 3. TOP 10 PRODUCTS
# ============================================================

def plot_top_products(df):

    top_products = (
        df.groupby("StockCode")["Demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_products.index.astype(str),
        top_products.values
    )

    plt.title("Top 10 Products by Total Demand")
    plt.xlabel("Total Demand")
    plt.ylabel("SKU")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/03_top_products.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 4. DEMAND BY DAY OF WEEK
# ============================================================

def plot_weekday_demand(df):

    temp = df.copy()

    temp["DayName"] = temp["Date"].dt.day_name()

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday = (
        temp.groupby("DayName")["Demand"]
        .mean()
        .reindex(order)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        weekday.index,
        weekday.values
    )

    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Demand")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/04_weekday_demand.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 5. MONTHLY SEASONALITY
# ============================================================

def plot_monthly_seasonality(df):

    temp = df.copy()

    temp["Month"] = temp["Date"].dt.month

    monthly = (
        temp.groupby("Month")["Demand"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.title("Monthly Demand Seasonality")
    plt.xlabel("Month")
    plt.ylabel("Average Demand")

    plt.xticks(range(1, 13))

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/05_monthly_seasonality.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 6. DEMAND DISTRIBUTION
# ============================================================

def plot_demand_distribution(df):

    upper = df["Demand"].quantile(0.99)

    values = df.loc[
        df["Demand"] <= upper,
        "Demand"
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=50
    )

    plt.title("Demand Distribution")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/06_demand_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 7. DEMAND VS TRANSACTIONS
# ============================================================

def plot_demand_transactions(df):

    # Aggregate by date to make the relationship easier to read
    daily = (
        df.groupby("Date")
        .agg(
            Demand=("Demand", "sum"),
            Transactions=("Transactions", "sum")
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        daily["Transactions"],
        daily["Demand"],
        alpha=0.5
    )

    plt.title("Demand vs Transactions")
    plt.xlabel("Transactions")
    plt.ylabel("Demand")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/07_demand_vs_transactions.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 8. TOP SKU DEMAND TREND
# ============================================================

def plot_top_sku_trend(df):

    top_sku = (
        df.groupby("StockCode")["Demand"]
        .sum()
        .idxmax()
    )

    sku_data = (
        df[df["StockCode"] == top_sku]
        .groupby("Date")["Demand"]
        .sum()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        sku_data.index,
        sku_data.values
    )

    plt.title(
        f"Demand Trend for Top SKU: {top_sku}"
    )

    plt.xlabel("Date")
    plt.ylabel("Demand")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/08_top_sku_trend.png",
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT - VISUALIZATION")
    print("=" * 70)

    setup()

    print("\nLoading data...")

    df = load_data()

    print(f"Rows: {len(df):,}")
    print(f"SKUs: {df['StockCode'].nunique():,}")

    print("\nCreating visualizations...")

    plot_daily_demand(df)
    print("✓ Daily demand trend")

    plot_monthly_demand(df)
    print("✓ Monthly demand trend")

    plot_top_products(df)
    print("✓ Top products")

    plot_weekday_demand(df)
    print("✓ Weekday demand")

    plot_monthly_seasonality(df)
    print("✓ Monthly seasonality")

    plot_demand_distribution(df)
    print("✓ Demand distribution")

    plot_demand_transactions(df)
    print("✓ Demand vs transactions")

    plot_top_sku_trend(df)
    print("✓ Top SKU demand trend")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)

    print(f"\nCharts saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()