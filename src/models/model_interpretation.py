import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error




PREDICTION_FILE = "data/processed/test_predictions.csv"
IMPORTANCE_FILE = "reports/model/feature_importance.csv"

OUTPUT_DIR = "reports/model"


def create_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


def load_data():

    predictions = pd.read_csv(
        PREDICTION_FILE,
        low_memory=False
    )

    importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    return predictions, importance



def plot_feature_importance(importance):

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    top_features = importance.tail(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features.iloc[:, 0],
        top_features["Importance"]
    )

    plt.xlabel(
        "Permutation Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "FORESIGHT - Top Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/feature_importance.png",
        dpi=150
    )

    plt.close()

    print(
        "✓ Feature importance chart saved"
    )



def plot_actual_vs_predicted(predictions):

    sample = predictions.sample(
        min(5000, len(predictions)),
        random_state=42
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        sample["Demand"],
        sample["PredictedDemand"],
        alpha=0.3
    )

    max_value = max(
        sample["Demand"].max(),
        sample["PredictedDemand"].max()
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Demand"
    )

    plt.ylabel(
        "Predicted Demand"
    )

    plt.title(
        "Actual vs Predicted Demand"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    print(
        "✓ Actual vs predicted chart saved"
    )


def residual_analysis(predictions):

    predictions["Residual"] = (
        predictions["Demand"] -
        predictions["PredictedDemand"]
    )

    predictions["AbsoluteError"] = (
        np.abs(
            predictions["Residual"]
        )
    )

    mae = mean_absolute_error(
        predictions["Demand"],
        predictions["PredictedDemand"]
    )

    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)

    print(
        f"MAE: {mae:.4f}"
    )

    print(
        f"Mean residual: "
        f"{predictions['Residual'].mean():.4f}"
    )

    print(
        f"Median residual: "
        f"{predictions['Residual'].median():.4f}"
    )

    print(
        f"90th percentile absolute error: "
        f"{predictions['AbsoluteError'].quantile(0.90):.4f}"
    )

    # --------------------------------------------------------
    # Residual distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        predictions["Residual"],
        bins=100
    )

    plt.xlabel(
        "Residual (Actual - Predicted)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Demand Forecast Residual Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/residual_distribution.png",
        dpi=150
    )

    plt.close()

    print(
        "✓ Residual distribution saved"
    )


# ============================================================
# LARGEST ERRORS
# ============================================================

def analyze_largest_errors(predictions):

    largest_errors = (
        predictions
        .sort_values(
            "AbsoluteError",
            ascending=False
        )
        .head(10)
    )

    print("\n" + "=" * 60)
    print("TOP 10 LARGEST FORECASTING ERRORS")
    print("=" * 60)

    print(
        largest_errors[
            [
                "Date",
                "StockCode",
                "Demand",
                "PredictedDemand",
                "AbsoluteError"
            ]
        ].to_string(
            index=False
        )
    )

    largest_errors.to_csv(
        f"{OUTPUT_DIR}/largest_errors.csv",
        index=False
    )

    print(
        "\n✓ Largest errors saved"
    )



def main():

    print("=" * 60)
    print("FORESIGHT - MODEL INTERPRETATION")
    print("=" * 60)

    create_directory()

    predictions, importance = load_data()

    print(
        f"\nPrediction rows: "
        f"{len(predictions):,}"
    )

    plot_feature_importance(
        importance
    )

    plot_actual_vs_predicted(
        predictions
    )

    residual_analysis(
        predictions
    )

    analyze_largest_errors(
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL INTERPRETATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()