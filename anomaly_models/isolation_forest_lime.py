"""
LIME explanations for the saved Isolation Forest model.

Usage:
    python -m anomaly_models.isolation_forest_lime --row-index 123
    python -m anomaly_models.isolation_forest_lime --anomaly-only
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from anomaly_models.common import (
    default_ml_ready_path,
    load_ml_ready,
    project_root,
    validate_features,
)


def _import_lime_explainer():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: lime. Install it with `pip install lime`."
        ) from exc
    return LimeTabularExplainer


def load_isolation_forest_artifacts(
    model_dir: str | None = None,
) -> Tuple[object, object, list[str]]:
    root = project_root()
    model_dir = model_dir or os.path.join(root, "models", "isolation_forest")

    model_path = os.path.join(model_dir, "isolation_forest.joblib")
    scaler_path = os.path.join(model_dir, "standard_scaler.joblib")
    features_path = os.path.join(model_dir, "feature_columns.joblib")

    missing = [
        path
        for path in [model_path, scaler_path, features_path]
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Isolation Forest artifacts. Run the training script first "
            "with `python train_anomaly.py --method isolation_forest --save`."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)
    return model, scaler, list(feature_columns)


def load_feature_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = load_ml_ready(csv_path)
    missing, non_numeric = validate_features(X, y)
    if missing:
        raise ValueError(
            f"Dataset contains {missing} missing values. Please clean the data first."
        )
    if non_numeric:
        raise ValueError(
            "LIME integration for Isolation Forest expects numeric features only. "
            f"Non-numeric columns: {non_numeric}"
        )
    return X, y


def _build_predict_proba(model, scaler, feature_columns: list[str], X_train: pd.DataFrame):
    train_scaled = scaler.transform(X_train[feature_columns])
    train_scores = model.decision_function(train_scaled)
    score_mean = float(np.mean(train_scores))
    score_std = float(np.std(train_scores)) or 1.0

    def predict_proba(raw_rows: np.ndarray) -> np.ndarray:
        rows = pd.DataFrame(raw_rows, columns=feature_columns)
        scaled = scaler.transform(rows)
        scores = model.decision_function(scaled)
        normalized = (scores - score_mean) / score_std
        normalized = np.clip(normalized, -20.0, 20.0)
        anomaly_prob = 1.0 / (1.0 + np.exp(normalized))
        normal_prob = 1.0 - anomaly_prob
        return np.column_stack([normal_prob, anomaly_prob])

    return predict_proba


def _select_row(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    scaler,
    feature_columns: list[str],
    row_index: int | None,
    anomaly_only: bool,
) -> int:
    if row_index is not None:
        if row_index < 0 or row_index >= len(X):
            raise IndexError(
                f"row-index {row_index} is out of range for dataset with {len(X)} rows."
            )
        return row_index

    if anomaly_only:
        scaled = scaler.transform(X[feature_columns])
        preds = model.predict(scaled)
        anomaly_indices = np.where(preds == -1)[0]
        if len(anomaly_indices) == 0:
            raise ValueError(
                "No anomaly rows were found in the current dataset. "
                "Try removing --anomaly-only or choose a specific --row-index."
            )
        return int(anomaly_indices[0])

    return 0


def explain_row(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    scaler,
    feature_columns: list[str],
    row_index: int,
    top_features: int,
    num_samples: int,
    html_output: str | None = None,
) -> str | None:
    LimeTabularExplainer = _import_lime_explainer()
    predict_proba = _build_predict_proba(model, scaler, feature_columns, X)

    explainer = LimeTabularExplainer(
        training_data=X[feature_columns].to_numpy(),
        feature_names=feature_columns,
        class_names=["normal", "anomaly"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    row = X.iloc[row_index][feature_columns]
    explanation = explainer.explain_instance(
        row.to_numpy(),
        predict_proba,
        num_features=top_features,
        num_samples=num_samples,
        top_labels=1,
    )

    scaled_row = scaler.transform(X.iloc[[row_index]][feature_columns])
    raw_pred = int(model.predict(scaled_row)[0])
    proba = predict_proba(row.to_numpy().reshape(1, -1))[0]
    anomaly_score = float(proba[1])
    label_name = "anomaly" if raw_pred == -1 else "normal"

    print("\n--- LIME explanation for Isolation Forest ---")
    print(f"Row index: {row_index}")
    print(f"Target label: {int(y.iloc[row_index])}")
    print(f"Model prediction: {raw_pred} ({label_name})")
    print(f"Estimated anomaly probability: {anomaly_score:.4f}")
    print("\nTop contributing features for anomaly class:")
    for feature, weight in explanation.as_list(label=1):
        print(f"- {feature}: {weight:+.4f}")

    if html_output:
        output_dir = os.path.dirname(html_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        explanation.save_to_file(html_output)
        print(f"\nHTML explanation saved to: {html_output}")
        return html_output

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LIME explanation for the saved Isolation Forest model."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=default_ml_ready_path(),
        help="Path to processedfiles/ml_ready.csv",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing isolation_forest.joblib, scaler and feature columns.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zero-based row index in the CSV to explain.",
    )
    parser.add_argument(
        "--anomaly-only",
        action="store_true",
        help="If no row index is provided, explain the first predicted anomaly.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="Number of features to show in the explanation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of perturbed samples used by LIME.",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Optional path to save the rendered HTML explanation.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print("Skedari nuk u gjet:", args.csv, file=sys.stderr)
        sys.exit(1)

    X, y = load_feature_data(args.csv)
    model, scaler, feature_columns = load_isolation_forest_artifacts(args.model_dir)

    selected_index = _select_row(
        X,
        y,
        model,
        scaler,
        feature_columns,
        args.row_index,
        args.anomaly_only,
    )

    html_output = args.html_output
    if html_output is None:
        html_output = os.path.join(
            project_root(),
            "analysis_outputs",
            "lime",
            f"isolation_forest_row_{selected_index}.html",
        )

    explain_row(
        X=X,
        y=y,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        row_index=selected_index,
        top_features=args.top_features,
        num_samples=args.num_samples,
        html_output=html_output,
    )


if __name__ == "__main__":
    main()
