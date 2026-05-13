from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from anomaly_models.common import TARGET_COL, project_root


def _import_profile_report():
    try:
        from ydata_profiling import ProfileReport
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: ydata-profiling. Install it with "
            "`pip install ydata-profiling`."
        ) from exc
    return ProfileReport


@dataclass(frozen=True)
class Config:
    n_estimators: int = 100
    contamination: float = 0.05
    random_state: int = 42
    test_size: float = 0.2
    anomaly_label: int = 1
    report_file: str = "rainfall_report.html"


def generate_rainfall_report(
    df: pd.DataFrame,
    output_file: str = "rainfall_report.html",
) -> str:
    ProfileReport = _import_profile_report()
    report = ProfileReport(
        df,
        title="Rainfall Report",
        explorative=True,
        minimal=False,
        progress_bar=False,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        missing_diagrams={
            "matrix": True,
            "bar": True,
            "heatmap": True,
            "dendrogram": True,
        },
    )
    report.to_file(output_file)
    return output_file


def _to_binary_anomaly(y: pd.Series, anomaly_label: int = 1) -> pd.Series:
    return (y == anomaly_label).astype(int)


def train_and_log(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: Optional[pd.Series],
    config: Config,
    experiment_name: str = "rainfall",
) -> Dict[str, Any]:
    mlflow.set_experiment(experiment_name)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=-1,
        )

        model.fit(X_train_scaled)
        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred == -1).astype(int)

        accuracy = None
        f1 = None
        if y_test is not None:
            y_true_binary = _to_binary_anomaly(y_test, config.anomaly_label)
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary, average="weighted")
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "f1_score": f1,
                }
            )

        mlflow.log_params(
            {
                "model_type": "IsolationForest",
                "n_estimators": config.n_estimators,
                "contamination": config.contamination,
                "random_state": config.random_state,
                "test_size": config.test_size,
                "anomaly_label": config.anomaly_label,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_features": X_train.shape[1],
            }
        )

        mlflow.sklearn.log_model(model, "model")

        return {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "f1_score": f1,
            "run_id": mlflow.active_run().info.run_id,
        }


def run_rainfall_eda_and_tracking(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    report_file: str = "rainfall_report.html",
    config: Config = Config(),
    experiment_name: str = "rainfall",
) -> Dict[str, Any]:
    report_path = generate_rainfall_report(df, output_file=report_file)
    results = train_and_log(
        X_train,
        X_test,
        y_test,
        config=config,
        experiment_name=experiment_name,
    )
    results["report_path"] = report_path
    return results


def run_rainfall_from_csv(
    csv_path: str,
    target_column: Optional[str] = TARGET_COL,
    *,
    report_file: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    config: Config = Config(),
    experiment_name: str = "rainfall",
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False)
    if target_column and target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found in {csv_path!r}."
        )

    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        y = None
        X = df

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.nunique() > 1 else None,
        )
    else:
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
        )
        y_train = pd.Series(dtype="int64")
        y_test = None

    resolved_report_file = report_file or os.path.join(
        project_root(),
        "rainfall_report.html",
    )

    return run_rainfall_eda_and_tracking(
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        report_file=resolved_report_file,
        config=config,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate a rainfall report and log an Isolation Forest MLflow run."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=os.path.join("processedfiles", "ml_ready.csv"),
        help="Path to the prepared CSV file.",
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COL,
        help="Name of the target column in the CSV.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Output HTML report path. Defaults to ./rainfall_report.html",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--experiment-name", default="rainfall")
    args = parser.parse_args()

    config = Config(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
        test_size=args.test_size,
        report_file=args.report_file or "rainfall_report.html",
    )
    results = run_rainfall_from_csv(
        args.csv,
        args.target_column,
        report_file=args.report_file,
        test_size=args.test_size,
        random_state=args.random_state,
        config=config,
        experiment_name=args.experiment_name,
    )
    print(f"Report saved to: {results['report_path']}")
    print(f"MLflow run_id: {results['run_id']}")
    print(f"Accuracy: {results['accuracy']:.4f}" if results["accuracy"] is not None else "Accuracy: n/a")
    print(f"F1-score: {results['f1_score']:.4f}" if results["f1_score"] is not None else "F1-score: n/a")
