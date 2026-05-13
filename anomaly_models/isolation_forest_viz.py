"""
Interactive Plotly visualizations for the saved Isolation Forest model.

Usage:
    python -m anomaly_models.isolation_forest_viz
    python -m anomaly_models.isolation_forest_viz --html-output analysis_outputs/viz/iforest.html
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from anomaly_models.common import (
    default_ml_ready_path,
    load_ml_ready,
    project_root,
    validate_features,
)


def _import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: plotly. Install it with `pip install plotly`."
        ) from exc
    return go, make_subplots


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
            "Missing Isolation Forest artifacts. Run "
            "`python train_anomaly.py --method isolation_forest --save` first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)
    return model, scaler, list(feature_columns)


def _prepare_frame(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    scaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    X_model = X[feature_columns].copy()
    scaled = scaler.transform(X_model)
    preds = model.predict(scaled)
    scores = model.decision_function(scaled)

    out = X_model.copy()
    out["target"] = y.values
    out["prediction"] = preds
    out["anomaly_flag"] = (preds == -1).astype(int)
    out["decision_score"] = scores
    return out


def build_dashboard(
    df: pd.DataFrame,
    html_output: str,
    title: str = "Isolation Forest Visualization",
) -> str:
    go, make_subplots = _import_plotly()

    pca = PCA(n_components=2, random_state=42)
    pca_xy = pca.fit_transform(df.drop(columns=["target", "prediction", "anomaly_flag", "decision_score"]))

    dashboard = df.copy()
    dashboard["pca_1"] = pca_xy[:, 0]
    dashboard["pca_2"] = pca_xy[:, 1]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Decision Score Distribution",
            "PCA Projection",
            "Prediction Counts",
            "Anomaly Score by Target",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    normal_scores = dashboard.loc[dashboard["prediction"] == 1, "decision_score"]
    anomaly_scores = dashboard.loc[dashboard["prediction"] == -1, "decision_score"]
    fig.add_trace(
        go.Histogram(
            x=normal_scores,
            name="Normal",
            nbinsx=50,
            opacity=0.75,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=anomaly_scores,
            name="Anomaly",
            nbinsx=50,
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    colors = np.where(dashboard["prediction"] == -1, "#dc3545", "#198754")
    fig.add_trace(
        go.Scattergl(
            x=dashboard["pca_1"],
            y=dashboard["pca_2"],
            mode="markers",
            marker=dict(color=colors, size=5, opacity=0.7),
            text=dashboard["target"].astype(str),
            name="Rows",
        ),
        row=1,
        col=2,
    )

    pred_counts = dashboard["prediction"].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=["Normal (1)", "Anomaly (-1)"],
            y=[int(pred_counts.get(1, 0)), int(pred_counts.get(-1, 0))],
            marker_color=["#198754", "#dc3545"],
            name="Counts",
        ),
        row=2,
        col=1,
    )

    target_groups = [
        dashboard.loc[dashboard["target"] == value, "decision_score"]
        for value in sorted(dashboard["target"].unique())
    ]
    target_labels = [str(value) for value in sorted(dashboard["target"].unique())]
    fig.add_trace(
        go.Box(
            y=target_groups[0] if target_groups else [],
            name=target_labels[0] if target_labels else "target",
            boxmean=True,
        ),
        row=2,
        col=2,
    )
    for label, values in zip(target_labels[1:], target_groups[1:]):
        fig.add_trace(
            go.Box(
                y=values,
                name=label,
                boxmean=True,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=title,
        height=900,
        width=1300,
        barmode="overlay",
        template="plotly_white",
        legend_title_text="Legend",
    )
    fig.update_xaxes(title_text="Decision score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="PCA 1", row=1, col=2)
    fig.update_yaxes(title_text="PCA 2", row=1, col=2)
    fig.update_xaxes(title_text="Prediction", row=2, col=1)
    fig.update_yaxes(title_text="Rows", row=2, col=1)
    fig.update_yaxes(title_text="Decision score", row=2, col=2)

    output_dir = os.path.dirname(html_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.write_html(html_output, include_plotlyjs="cdn")
    return html_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly dashboard for Isolation Forest."
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
        help="Directory containing the saved Isolation Forest artifacts.",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Where to save the HTML dashboard.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print("Skedari nuk u gjet:", args.csv, file=sys.stderr)
        sys.exit(1)

    X, y = load_ml_ready(args.csv)
    missing, non_numeric = validate_features(X, y)
    if missing:
        raise ValueError(f"Dataset contains {missing} missing values.")
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    model, scaler, feature_columns = load_isolation_forest_artifacts(args.model_dir)
    dashboard = _prepare_frame(X, y, model, scaler, feature_columns)

    html_output = args.html_output or os.path.join(
        project_root(),
        "analysis_outputs",
        "viz",
        "isolation_forest_dashboard.html",
    )

    path = build_dashboard(dashboard, html_output)
    print(f"Dashboard saved to: {path}")


if __name__ == "__main__":
    main()
