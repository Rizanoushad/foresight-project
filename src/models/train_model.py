import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"
PREDICTION_DIR = "data/processed"
REPORT_DIR = "reports/model"

MODEL_FILE = f"{MODEL_DIR}/foresight_demand_model.pkl"
PREDICTION_FILE = f"{PREDICTION_DIR}/test_predictions.csv"
IMPORTANCE_FILE = f"{REPORT_DIR}/feature_importance.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Year",
    "Month",
    "Quarter",
    "DayOfMonth",
    "DayOfWeek",
    "WeekOfYear",
    "IsWeekend",
    "MonthSin",
    "MonthCos",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "RollingMean_7",
    "RollingStd_7",
    "RollingMean_14",
    "RollingStd_14",
    "RollingMean_28",
    "RollingStd_28",
]

TARGET = "Demand"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    """Create required output directories."""

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():
    """Load train, validation, and test datasets."""

    print("\nLoading datasets...")

    train = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    validation = pd.read_csv(
        VALIDATION_FILE,
        low_memory=False
    )

    test = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    print(f"Train rows      : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Test rows       : {len(test):,}")

    return train, validation, test


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation, test):
    """Prepare feature matrices and target variables."""

    print("\nPreparing features...")

    # Training data
    X_train = train[FEATURES].copy()
    y_train = train[TARGET].copy()

    # Validation data
    X_validation = validation[FEATURES].copy()
    y_validation = validation[TARGET].copy()

    # Test data
    X_test = test[FEATURES].copy()
    y_test = test[TARGET].copy()

    # Replace infinite values with NaN
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_validation = X_validation.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert features to float32
    X_train = X_train.astype("float32")
    X_validation = X_validation.astype("float32")
    X_test = X_test.astype("float32")

    # Convert target to float32
    y_train = y_train.astype("float32")
    y_validation = y_validation.astype("float32")
    y_test = y_test.astype("float32")

    print(f"Number of features: {len(FEATURES)}")

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """Train HistGradientBoostingRegressor."""

    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    print("\nModel: HistGradientBoostingRegressor")

    start_time = time.time()

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_time / 60:.2f} minutes"
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):
    """Evaluate model using MAE and RMSE."""

    predictions = model.predict(X)

    # Demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    print("\n" + "-" * 60)
    print(f"{dataset_name} RESULTS")
    print("-" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return predictions, mae, rmse


# ============================================================
# NAIVE BASELINE
# ============================================================

def evaluate_naive_baseline(test):
    """Evaluate previous-day demand as a baseline."""

    print("\n" + "=" * 70)
    print("NAIVE BASELINE")
    print("=" * 70)

    # Predict today's demand using previous day's demand
    actual = test[TARGET].values
    baseline = test["Lag_1"].values

    # Remove rows where Lag_1 is missing
    valid = ~np.isnan(baseline)

    actual = actual[valid]
    baseline = baseline[valid]

    # Demand cannot be negative
    baseline = np.maximum(
        baseline,
        0
    )

    mae = mean_absolute_error(
        actual,
        baseline
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            baseline
        )
    )

    print(
        f"Naive Baseline MAE  : {mae:.4f}"
    )

    print(
        f"Naive Baseline RMSE : {rmse:.4f}"
    )

    return mae, rmse


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test,
    predictions,
):
    """Save test predictions."""

    output = test[
        [
            "Date",
            "StockCode",
            "Demand",
        ]
    ].copy()

    output["PredictedDemand"] = predictions

    output["AbsoluteError"] = np.abs(
        output["Demand"] -
        output["PredictedDemand"]
    )

    output.to_csv(
        PREDICTION_FILE,
        index=False
    )

    print(
        f"\nPredictions saved to:"
        f"\n{PREDICTION_FILE}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """Save trained model."""

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    X_test,
    y_test,
):
    """Calculate permutation feature importance."""

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        "\nHistGradientBoostingRegressor does not provide "
        "built-in feature_importances_."
    )

    print(
        "Using permutation importance instead..."
    )

    sample_size = min(
        5000,
        len(X_test)
    )

    X_sample = X_test.iloc[
        :sample_size
    ]

    y_sample = y_test.iloc[
        :sample_size
    ]

    print(
        f"\nFeature importance sample: "
        f"{sample_size:,} rows"
    )

    start_time = time.time()

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=3,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    elapsed = time.time() - start_time

    importance = pd.Series(
        result.importances_mean,
        index=FEATURES,
    ).sort_values(
        ascending=False
    )

    print(
        f"\nFeature importance completed in "
        f"{elapsed:.2f} seconds"
    )

    print("\nTop features:")
    print("-" * 50)

    print(
        importance.head(10).to_string()
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        header=["Importance"]
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{IMPORTANCE_FILE}"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis(
    predictions,
    y_test,
):
    """Analyze prediction errors."""

    errors = np.abs(
        y_test.values -
        predictions
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Mean absolute error   : "
        f"{errors.mean():.4f}"
    )

    print(
        f"Median absolute error : "
        f"{np.median(errors):.4f}"
    )

    print(
        f"90th percentile error : "
        f"{np.percentile(errors, 90):.4f}"
    )

    print(
        f"Maximum error         : "
        f"{errors.max():.4f}"
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary(
    validation_mae,
    validation_rmse,
    test_mae,
    test_rmse,
    baseline_mae,
    baseline_rmse,
):
    """Print final model performance."""

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\n                    MAE        RMSE")
    print("-" * 45)

    print(
        f"Naive Baseline    {baseline_mae:8.4f}   "
        f"{baseline_rmse:8.4f}"
    )

    print(
        f"Validation        {validation_mae:8.4f}   "
        f"{validation_rmse:8.4f}"
    )

    print(
        f"Test              {test_mae:8.4f}   "
        f"{test_rmse:8.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete model training pipeline."""

    print("=" * 70)
    print("FORESIGHT - MACHINE LEARNING")
    print("=" * 70)

    # Create required directories
    create_directories()

    # Load datasets
    train, validation, test = load_datasets()

    # Prepare features
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_features(
        train,
        validation,
        test,
    )

    # Train model
    model = train_model(
        X_train,
        y_train,
    )

    # Validation evaluation
    (
        validation_predictions,
        validation_mae,
        validation_rmse,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    # Test evaluation
    (
        test_predictions,
        test_mae,
        test_rmse,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST",
    )

    # Baseline evaluation
    baseline_mae, baseline_rmse = (
        evaluate_naive_baseline(
            test
        )
    )

    # Save predictions
    save_predictions(
        test,
        test_predictions,
    )

    # Save model
    save_model(
        model
    )

    # Feature importance
    show_feature_importance(
        model,
        X_test,
        y_test,
    )

    # Error analysis
    error_analysis(
        test_predictions,
        y_test,
    )

    # Final summary
    print_model_summary(
        validation_mae,
        validation_rmse,
        test_mae,
        test_rmse,
        baseline_mae,
        baseline_rmse,
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nOutput files:")

    print(
        f"✓ Model:"
        f"\n  {MODEL_FILE}"
    )

    print(
        f"\n✓ Predictions:"
        f"\n  {PREDICTION_FILE}"
    )

    print(
        f"\n✓ Feature importance:"
        f"\n  {IMPORTANCE_FILE}"
    )

    print(
        "\nFORESIGHT ML stage completed successfully."
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()