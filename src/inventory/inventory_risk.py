import os
import numpy as np
import pandas as pd





PREDICTION_FILE = "data/processed/test_predictions.csv"
OUTPUT_DIR = "data/processed"

OUTPUT_FILE = (
    f"{OUTPUT_DIR}/inventory_risk_analysis.csv"
)



# SCENARIO SETTINGS


# These are scenario assumptions, NOT actual inventory data.
FORECAST_HORIZON_DAYS = 7

# Assume inventory available for this many days
INVENTORY_COVERAGE_DAYS = 14

# Safety buffer applied to forecast demand
SAFETY_BUFFER = 0.20

# Risk thresholds
HIGH_RISK_THRESHOLD = 1.0
MEDIUM_RISK_THRESHOLD = 0.5



def load_predictions():

    print("\nLoading demand predictions...")

    df = pd.read_csv(
        PREDICTION_FILE,
        low_memory=False
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    print(
        f"Prediction rows: {len(df):,}"
    )

    return df




def prepare_sku_analysis(df):

    print("\nPreparing SKU-level inventory analysis...")

    
    latest = (
        df.sort_values("Date")
        .groupby("StockCode")
        .tail(1)
        .copy()
    )

    

    recent_demand = (
        df.sort_values("Date")
        .groupby("StockCode")
        .tail(28)
        .groupby("StockCode")["Demand"]
        .mean()
        .rename("RecentAverageDemand")
    )

    latest = latest.merge(
        recent_demand,
        on="StockCode",
        how="left"
    )

    return latest




def calculate_demand_metrics(df):

    print("\nCalculating demand metrics...")

    

    df["ForecastDemand"] = (
        df["PredictedDemand"]
        .clip(lower=0)
    )

   
    df["ForecastHorizonDemand"] = (
        df["ForecastDemand"]
        * FORECAST_HORIZON_DAYS
    )

    
    df["SafetyStock"] = (
        df["ForecastHorizonDemand"]
        * SAFETY_BUFFER
    )

    
    df["RecommendedInventory"] = (
        df["ForecastHorizonDemand"]
        + df["SafetyStock"]
    )

    

    df["AssumedInventory"] = (
        df["RecentAverageDemand"]
        * INVENTORY_COVERAGE_DAYS
    )

    return df



def calculate_risk(df):

    print("\nCalculating inventory risk...")

    
    df["InventoryGap"] = (
        df["AssumedInventory"]
        - df["RecommendedInventory"]
    )

    
    df["CoverageRatio"] = np.where(
        df["RecommendedInventory"] > 0,
        df["AssumedInventory"]
        / df["RecommendedInventory"],
        np.inf
    )

    

    df["StockoutRiskScore"] = np.where(
        df["RecommendedInventory"] > 0,
        np.maximum(
            0,
            1 -
            (
                df["AssumedInventory"]
                /
                df["RecommendedInventory"]
            )
        ),
        0
    )

   
    df["OverstockRiskScore"] = np.maximum(
        0,
        (
            df["AssumedInventory"]
            /
            df["RecommendedInventory"].replace(
                0,
                np.nan
            )
        ) - 1
    ).fillna(0)

    
    conditions = [
        df["StockoutRiskScore"]
        >= HIGH_RISK_THRESHOLD,

        df["StockoutRiskScore"]
        >= MEDIUM_RISK_THRESHOLD,

        df["OverstockRiskScore"]
        >= 1.0
    ]

    choices = [
        "High Stockout Risk",
        "Medium Stockout Risk",
        "Overstock Risk"
    ]

    df["RiskCategory"] = np.select(
        conditions,
        choices,
        default="Low Risk"
    )

    return df




def generate_recommendations(df):

    print("\nGenerating inventory recommendations...")

    conditions = [
        df["RiskCategory"] == "High Stockout Risk",

        df["RiskCategory"] == "Medium Stockout Risk",

        df["RiskCategory"] == "Overstock Risk"
    ]

    recommendations = [
        "Prioritize replenishment",

        "Monitor demand and consider replenishment",

        "Reduce replenishment and review excess stock"
    ]

    df["Recommendation"] = np.select(
        conditions,
        recommendations,
        default="Maintain current inventory strategy"
    )

    return df



def print_summary(df):

    print("\n" + "=" * 70)
    print("INVENTORY RISK SUMMARY")
    print("=" * 70)

    total_skus = len(df)

    high_risk = (
        df["RiskCategory"]
        == "High Stockout Risk"
    ).sum()

    medium_risk = (
        df["RiskCategory"]
        == "Medium Stockout Risk"
    ).sum()

    overstock = (
        df["RiskCategory"]
        == "Overstock Risk"
    ).sum()

    low_risk = (
        df["RiskCategory"]
        == "Low Risk"
    ).sum()

    print(
        f"\nTotal SKUs analyzed : {total_skus:,}"
    )

    print(
        f"High stockout risk  : {high_risk:,}"
    )

    print(
        f"Medium stockout risk: {medium_risk:,}"
    )

    print(
        f"Overstock risk      : {overstock:,}"
    )

    print(
        f"Low risk            : {low_risk:,}"
    )

    print("\nRisk distribution:")

    print(
        df["RiskCategory"]
        .value_counts()
        .to_string()
    )


# ============================================================
# TOP RISK PRODUCTS
# ============================================================

def show_top_risk_products(df):

    print("\n" + "=" * 70)
    print("TOP STOCKOUT-RISK SKUs")
    print("=" * 70)

    high_risk = (
        df[
            df["RiskCategory"]
            == "High Stockout Risk"
        ]
        .sort_values(
            "StockoutRiskScore",
            ascending=False
        )
        .head(10)
    )

    if len(high_risk) == 0:

        print(
            "\nNo high stockout-risk SKUs "
            "under the current scenario."
        )

        return

    print(
        high_risk[
            [
                "StockCode",
                "ForecastDemand",
                "RecentAverageDemand",
                "AssumedInventory",
                "RecommendedInventory",
                "StockoutRiskScore",
                "Recommendation"
            ]
        ].to_string(
            index=False
        )
    )



def save_results(df):

    output_columns = [
        "Date",
        "StockCode",
        "Demand",
        "PredictedDemand",
        "RecentAverageDemand",
        "ForecastDemand",
        "ForecastHorizonDemand",
        "SafetyStock",
        "RecommendedInventory",
        "AssumedInventory",
        "InventoryGap",
        "CoverageRatio",
        "StockoutRiskScore",
        "OverstockRiskScore",
        "RiskCategory",
        "Recommendation"
    ]

    df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nInventory analysis saved to:"
        f"\n{OUTPUT_FILE}"
    )



def main():

    print("=" * 70)
    print("FORESIGHT - INVENTORY RISK INTELLIGENCE")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    
    predictions = load_predictions()

    
    df = prepare_sku_analysis(
        predictions
    )

    

    df = calculate_demand_metrics(
        df
    )

   

    df = calculate_risk(
        df
    )

    
    df = generate_recommendations(
        df
    )

   

    print_summary(
        df
    )

    

    show_top_risk_products(
        df
    )



    save_results(
        df
    )

    print("\n" + "=" * 70)
    print("INVENTORY RISK ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()