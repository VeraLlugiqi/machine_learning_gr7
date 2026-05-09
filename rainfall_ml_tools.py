from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


def _import_mlflow():
    try:
        import mlflow
        import mlflow.sklearn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: mlflow. Install it with `pip install mlflow`."
        ) from exc
    return mlflow


def _import_profile_report():
    try:
        from ydata_profiling import ProfileReport
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: ydata-profiling. Install it with "
            "`pip install ydata-profiling`."
        ) from exc
    return ProfileReport


def _import_random_forest_and_metrics():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: scikit-learn. Install it with "
            "`pip install scikit-learn`."
        ) from exc
    return RandomForestClassifier, accuracy_score, f1_score


def _import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: scikit-learn. Install it with "
            "`pip install scikit-learn`."
        ) from exc
    return train_test_split


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: Optional[int] = 12
    random_state: int = 42
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: Optional[str] = None


def generate_rainfall_profile_report(
    df: pd.DataFrame,
    output_file: str = "rainfall_report.html",
) -> str:
    """
    Generate a full ydata_profiling HTML report for exploratory data analysis.
    """
    ProfileReport = _import_profile_report()
    report = ProfileReport(
        df,
        title="Rainfall Dataset EDA Report",
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


def _resolve_average_for_f1(y_true: pd.Series) -> str:
    # Weighted keeps the score valid for binary and multi-class targets alike.
    return "weighted"


def train_and_log_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    config: RandomForestConfig = RandomForestConfig(),
    experiment_name: str = "rainfall",
    run_name: str = "random_forest",
    model_artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a RandomForest model, log params/metrics/model to MLflow, and return
    the fitted model plus evaluation results.
    """
    mlflow = _import_mlflow()
    RandomForestClassifier, accuracy_score, f1_score = _import_random_forest_and_metrics()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=_resolve_average_for_f1(y_test))

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": str(config.n_estimators),
                "max_depth": str(config.max_depth),
                "random_state": str(config.random_state),
                "min_samples_split": str(config.min_samples_split),
                "min_samples_leaf": str(config.min_samples_leaf),
                "class_weight": str(config.class_weight),
                "train_rows": str(len(X_train)),
                "test_rows": str(len(X_test)),
                "n_features": str(X_train.shape[1]),
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_score": f1,
            }
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_artifact_path,
        )

        return {
            "model": model,
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
    config: RandomForestConfig = RandomForestConfig(),
    experiment_name: str = "rainfall",
    run_name: str = "random_forest",
    model_artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that generates the profiling report and then runs MLflow
    tracking for a RandomForest baseline.
    """
    report_path = generate_rainfall_profile_report(df, output_file=report_file)
    results = train_and_log_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        config=config,
        experiment_name=experiment_name,
        run_name=run_name,
        model_artifact_path=model_artifact_path,
        tracking_uri=tracking_uri,
    )
    results["report_path"] = report_path
    return results


def run_rainfall_from_csv(
    csv_path: str,
    target_column: str,
    *,
    report_file: str = "rainfall_report.html",
    test_size: float = 0.2,
    random_state: int = 42,
    config: RandomForestConfig = RandomForestConfig(),
    experiment_name: str = "rainfall",
    run_name: str = "random_forest",
    model_artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    train_test_split = _import_train_test_split()
    df = pd.read_csv(csv_path, low_memory=False)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found in {csv_path!r}."
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )
    return run_rainfall_eda_and_tracking(
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        report_file=report_file,
        config=config,
        experiment_name=experiment_name,
        run_name=run_name,
        model_artifact_path=model_artifact_path,
        tracking_uri=tracking_uri,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate a rainfall EDA report and log a RandomForest MLflow run."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=os.path.join("processedfiles", "ml_ready.csv"),
        help="Path to the prepared CSV file.",
    )
    parser.add_argument(
        "--target-column",
        default="labels.authorization.k8s.io/decision__le",
        help="Name of the target column in the CSV.",
    )
    parser.add_argument(
        "--report-file",
        default="rainfall_report.html",
        help="Output HTML report path.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--experiment-name", default="rainfall")
    parser.add_argument("--run-name", default="random_forest")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    config = RandomForestConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    results = run_rainfall_from_csv(
        args.csv,
        args.target_column,
        report_file=args.report_file,
        test_size=args.test_size,
        random_state=args.random_state,
        config=config,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
    )
    print(f"Report saved to: {results['report_path']}")
    print(f"MLflow run_id: {results['run_id']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
