import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path



# PAGE CONFIGURATION


st.set_page_config(
    page_title="FORESIGHT - Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)



# PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "model"



# LOAD DATA


@st.cache_data
def load_data():

    predictions = pd.read_csv(
        DATA_DIR / "test_predictions.csv",
        dtype={"StockCode": str}
    )

    inventory = pd.read_csv(
        DATA_DIR / "inventory_risk_analysis.csv",
        dtype={"StockCode": str}
    )

    daily_demand = pd.read_csv(
        DATA_DIR / "daily_demand.csv",
        dtype={"StockCode": str}
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    daily_demand["Date"] = pd.to_datetime(
        daily_demand["Date"]
    )

    predictions["StockCode"] = (
        predictions["StockCode"]
        .astype(str)
        .str.strip()
    )

    inventory["StockCode"] = (
        inventory["StockCode"]
        .astype(str)
        .str.strip()
    )

    daily_demand["StockCode"] = (
        daily_demand["StockCode"]
        .astype(str)
        .str.strip()
    )

    return predictions, inventory, daily_demand



# LOAD DATA WITH ERROR HANDLING


try:

    predictions, inventory, daily_demand = load_data()

except Exception as e:

    st.error("Unable to load project data.")

    st.code(str(e))

    st.info(
        "Make sure you have already run the preprocessing, "
        "model training and inventory risk scripts."
    )

    st.stop()



# SIDEBAR


st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown(
    """
    ### FORESIGHT

    AI-Powered Demand & Inventory Intelligence
    """
)

st.sidebar.divider()



# INVENTORY SCENARIO CONTROLS


st.sidebar.subheader("Inventory Scenario")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=30,
    value=7
)

inventory_coverage = st.sidebar.slider(
    "Inventory Coverage (days)",
    min_value=1,
    max_value=60,
    value=14
)

safety_buffer = st.sidebar.slider(
    "Safety Buffer (%)",
    min_value=0,
    max_value=50,
    value=20
)

safety_buffer_decimal = safety_buffer / 100



# PREPARE LATEST PREDICTIONS


latest_predictions = (
    predictions
    .sort_values("Date")
    .groupby("StockCode")
    .tail(1)
    .copy()
)



# RECENT DEMAND


recent_demand = (
    daily_demand
    .sort_values("Date")
    .groupby("StockCode")
    .tail(28)
    .groupby("StockCode")["Demand"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "Demand": "RecentAverageDemand"
        }
    )
)



# BUILD INVENTORY SCENARIO


scenario_df = latest_predictions.merge(
    recent_demand,
    on="StockCode",
    how="left"
)


scenario_df["RecentAverageDemand"] = (
    scenario_df["RecentAverageDemand"]
    .fillna(0)
)


scenario_df["PredictedDemand"] = (
    scenario_df["PredictedDemand"]
    .clip(lower=0)
)


scenario_df["ForecastHorizonDemand"] = (
    scenario_df["PredictedDemand"]
    * forecast_horizon
)


scenario_df["SafetyStock"] = (
    scenario_df["ForecastHorizonDemand"]
    * safety_buffer_decimal
)


scenario_df["RecommendedInventory"] = (
    scenario_df["ForecastHorizonDemand"]
    + scenario_df["SafetyStock"]
)


scenario_df["AssumedInventory"] = (
    scenario_df["RecentAverageDemand"]
    * inventory_coverage
)


scenario_df["CoverageRatio"] = (
    scenario_df["AssumedInventory"]
    /
    scenario_df["RecommendedInventory"]
    .replace(0, np.nan)
)


scenario_df["CoverageRatio"] = (
    scenario_df["CoverageRatio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


scenario_df["StockoutRiskScore"] = (
    1 - scenario_df["CoverageRatio"]
).clip(lower=0)


scenario_df["OverstockRiskScore"] = (
    scenario_df["CoverageRatio"] - 1
).clip(lower=0)



# RISK CLASSIFICATION


def classify_risk(row):

    if row["StockoutRiskScore"] >= 1:

        return "High Stockout Risk"

    elif row["StockoutRiskScore"] >= 0.5:

        return "Medium Stockout Risk"

    elif row["OverstockRiskScore"] >= 1:

        return "Overstock Risk"

    else:

        return "Low Risk"


scenario_df["RiskCategory"] = (
    scenario_df.apply(
        classify_risk,
        axis=1
    )
)



# RECOMMENDATIONS


def recommendation(risk):

    if risk == "High Stockout Risk":

        return (
            "Prioritize replenishment immediately."
        )

    elif risk == "Medium Stockout Risk":

        return (
            "Monitor demand and consider replenishment."
        )

    elif risk == "Overstock Risk":

        return (
            "Reduce replenishment and review excess inventory."
        )

    else:

        return (
            "Maintain current inventory strategy."
        )


scenario_df["Recommendation"] = (
    scenario_df["RiskCategory"]
    .apply(recommendation)
)



# HEADER


st.title("📦 FORESIGHT")

st.subheader(
    "AI-Powered Demand & Inventory Intelligence Platform"
)

st.markdown(
    """
    Forecast SKU-level demand, identify inventory risks,
    and generate actionable inventory recommendations.
    """
)

st.divider()



# KPI SECTION


total_skus = scenario_df[
    "StockCode"
].nunique()


total_predicted_demand = scenario_df[
    "ForecastHorizonDemand"
].sum()


medium_risk = (
    scenario_df["RiskCategory"]
    == "Medium Stockout Risk"
).sum()


overstock_risk = (
    scenario_df["RiskCategory"]
    == "Overstock Risk"
).sum()


high_risk = (
    scenario_df["RiskCategory"]
    == "High Stockout Risk"
).sum()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📦 Total SKUs",
        f"{total_skus:,}"
    )


with col2:

    st.metric(
        "📈 Forecast Demand",
        f"{total_predicted_demand:,.0f}"
    )


with col3:

    st.metric(
        "⚠️ Medium Stockout Risk",
        f"{medium_risk:,}"
    )


with col4:

    st.metric(
        "📦 Overstock Risk",
        f"{overstock_risk:,}"
    )



# RISK OVERVIEW


st.divider()

st.header("🚦 Inventory Risk Overview")


risk_counts = (
    scenario_df["RiskCategory"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "RiskCategory",
    "Count"
]


col1, col2 = st.columns(2)


with col1:

    st.subheader("Risk Distribution")

    risk_chart = (
        risk_counts
        .set_index("RiskCategory")
    )

    st.bar_chart(
        risk_chart,
        width="stretch"
    )


with col2:

    st.subheader("Risk Summary")

    st.dataframe(
        risk_counts,
        width="stretch",
        hide_index=True
    )

    st.info(
        """
        Inventory risk is scenario-based because the
        Online Retail II dataset does not contain actual
        inventory-on-hand, supplier lead time, reorder point,
        or safety-stock data.
        """
    )



# SKU ANALYSIS


st.divider()

st.header("🔎 SKU-Level Analysis")


sku_list = sorted(
    scenario_df["StockCode"]
    .unique()
)


selected_sku = st.selectbox(
    "Select a SKU",
    sku_list
)


sku_info = scenario_df[
    scenario_df["StockCode"] == selected_sku
].iloc[0]


# ============================================================
# SKU KPI SECTION
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Predicted Daily Demand",
        f"{sku_info['PredictedDemand']:.2f}"
    )


with col2:

    st.metric(
        "Forecast Demand",
        f"{sku_info['ForecastHorizonDemand']:.2f}"
    )


with col3:

    st.metric(
        "Safety Stock",
        f"{sku_info['SafetyStock']:.2f}"
    )


with col4:

    st.metric(
        "Recommended Inventory",
        f"{sku_info['RecommendedInventory']:.2f}"
    )


with col5:

    st.metric(
        "Risk",
        sku_info["RiskCategory"]
    )



# PRODUCT DESCRIPTION


product_file = DATA_DIR / "product_names.csv"


if product_file.exists():

    try:

        products = pd.read_csv(
            product_file,
            dtype={"StockCode": str}
        )

        products["StockCode"] = (
            products["StockCode"]
            .astype(str)
            .str.strip()
        )

        product_match = products[
            products["StockCode"] == selected_sku
        ]

        if not product_match.empty:

            description = product_match.iloc[0].get(
                "Description",
                "Product description unavailable"
            )

            st.write(
                f"**Product:** {description}"
            )

    except Exception:

        pass



# HISTORICAL DEMAND


st.subheader("📊 Historical Demand")


sku_history = daily_demand[
    daily_demand["StockCode"] == selected_sku
].copy()


sku_history = (
    sku_history
    .sort_values("Date")
    .tail(120)
)


if not sku_history.empty:

    chart_data = (
        sku_history[
            ["Date", "Demand"]
        ]
        .set_index("Date")
    )

    st.line_chart(
        chart_data,
        width="stretch"
    )

else:

    st.warning(
        "Historical demand data unavailable for this SKU."
    )



# ACTUAL VS PREDICTED DEMAND


st.subheader("🤖 Actual vs Predicted Demand")


sku_predictions = predictions[
    predictions["StockCode"] == selected_sku
].copy()


sku_predictions = (
    sku_predictions
    .sort_values("Date")
    .tail(120)
)


if not sku_predictions.empty:

    # IMPORTANT:
    # test_predictions.csv contains:
    # Date
    # StockCode
    # Demand
    # PredictedDemand
    # AbsoluteError

    prediction_chart = (
        sku_predictions[
            [
                "Date",
                "Demand",
                "PredictedDemand"
            ]
        ]
        .set_index("Date")
    )


    prediction_chart = prediction_chart.rename(
        columns={
            "Demand": "Actual Demand",
            "PredictedDemand": "Predicted Demand"
        }
    )


    st.line_chart(
        prediction_chart,
        width="stretch"
    )


else:

    st.warning(
        "Prediction data unavailable for this SKU."
    )


# ============================================================
# INVENTORY RECOMMENDATION
# ============================================================

st.subheader("💡 Business Recommendation")


risk = sku_info["RiskCategory"]


if risk == "High Stockout Risk":

    st.error(
        """
        🚨 High Stockout Risk

        Prioritize replenishment immediately.
        """
    )


elif risk == "Medium Stockout Risk":

    st.warning(
        """
        ⚠️ Medium Stockout Risk

        Monitor demand closely and consider replenishment.
        """
    )


elif risk == "Overstock Risk":

    st.warning(
        """
        📦 Overstock Risk

        Consider reducing replenishment and reviewing
        excess inventory.
        """
    )


else:

    st.success(
        """
        ✅ Low Risk

        Maintain the current inventory strategy.
        """
    )



# TOP PRODUCTS REQUIRING ATTENTION


st.divider()

st.header("⚠️ Products Requiring Attention")


risk_priority = scenario_df.copy()


risk_priority["PriorityScore"] = (
    risk_priority["StockoutRiskScore"]
    +
    risk_priority["OverstockRiskScore"]
)


top_risk = (
    risk_priority
    .sort_values(
        "PriorityScore",
        ascending=False
    )
    .head(15)
)


display_columns = [
    "StockCode",
    "PredictedDemand",
    "RecommendedInventory",
    "AssumedInventory",
    "CoverageRatio",
    "RiskCategory",
    "Recommendation"
]


st.dataframe(
    top_risk[display_columns],
    width="stretch",
    hide_index=True
)



# MODEL PERFORMANCE


st.divider()

st.header("🤖 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Test MAE",
        "7.67"
    )


with col2:

    st.metric(
        "Test RMSE",
        "33.23"
    )


with col3:

    st.metric(
        "Baseline MAE",
        "9.82"
    )


st.success(
    """
    The forecasting model improved MAE by approximately
    21.8% compared with the naive previous-day baseline.
    """
)



# MODEL FEATURE IMPORTANCE


st.subheader("📊 Feature Importance")


importance_file = (
    REPORT_DIR /
    "feature_importance.csv"
)


if importance_file.exists():

    try:

        importance = pd.read_csv(
            importance_file
        )

        if "Feature" in importance.columns:

            importance = (
                importance
                .set_index("Feature")
                .head(10)
            )

        st.bar_chart(
            importance,
            width="stretch"
        )

    except Exception as e:

        st.warning(
            f"Unable to load feature importance: {e}"
        )

else:

    st.info(
        "Feature importance file not found."
    )



# MODEL INTERPRETATION


st.divider()

st.header("🧠 Model Insights")


st.markdown(
    """
    The model relies heavily on recent demand history.

    **Key observations:**

    - Rolling 14-day demand is one of the strongest features.
    - Rolling 28-day demand captures longer-term demand patterns.
    - Rolling 7-day demand captures short-term behavior.
    - Lag features help identify recent demand trends.
    - Calendar features capture weekly and seasonal patterns.

    The model can struggle with sudden demand spikes because
    unexpected events are not represented in the current dataset.
    """
)


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.divider()

st.header("🔄 FORESIGHT Pipeline")


st.markdown(
    """
    **Online Retail II Dataset**

    ↓

    **Data Cleaning**

    ↓

    **Preprocessing & Feature Engineering**

    ↓

    **Exploratory Data Analysis**

    ↓

    **Demand Forecasting**

    ↓

    **Model Evaluation**

    ↓

    **Model Interpretation**

    ↓

    **Inventory Risk Intelligence**

    ↓

    **Inventory Recommendations**

    ↓

    **Interactive Streamlit Dashboard**
    """
)



# ABOUT THE PROJECT


st.divider()

st.header("ℹ️ About FORESIGHT")


st.markdown(
    """
    ### What is FORESIGHT?

    FORESIGHT is an AI-powered demand forecasting and
    inventory intelligence platform designed to help
    businesses make better stock planning decisions.

    ### Main Capabilities

    - SKU-level demand forecasting
    - Historical demand analysis
    - Demand trend visualization
    - Inventory risk detection
    - Stockout risk analysis
    - Overstock risk analysis
    - Recommended inventory calculation
    - Business recommendations
    - Machine-learning model interpretation

    ### Machine Learning Model

    The demand forecasting system uses a
    **HistGradientBoostingRegressor**.

    The model uses:

    - Lag features
    - Rolling demand statistics
    - Calendar features
    - Seasonal features

    ### Dataset Limitation

    The Online Retail II dataset does not contain:

    - Actual inventory-on-hand
    - Supplier lead time
    - Reorder point
    - Safety stock
    - Purchase orders

    Therefore, inventory risk analysis in this project
    is **scenario-based** and uses configurable assumptions.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FORESIGHT | AI-Powered Demand & Inventory Intelligence Platform"
)