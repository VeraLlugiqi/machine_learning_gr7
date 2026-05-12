"""
Model improvement and retraining phase for anomaly detection.

This module keeps the baseline models separate from the optimized retraining
flow. It uses:
- Isolation Forest
- Local Outlier Factor with novelty detection
- One-Class SVM trained only on normal samples
- Elliptic Envelope

The public entrypoint is `run_model_improvement_phase(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class AnomalyModelResult:
    model_name: str
    variant: str
    params: Dict[str, Any]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[str]
    y_pred: np.ndarray
    model: Any
    scaler: StandardScaler

    def as_comparison_row(self) -> Dict[str, Any]:
        return {
            "Model Name": f"{self.model_name} ({self.variant})",
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
        }


def _validate_numeric_features(X: pd.DataFrame) -> None:
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            "All feature columns must be numeric for anomaly detection. "
            f"Non-numeric columns: {non_numeric}"
        )


def _prepare_train_test_scaler(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def _to_binary_anomaly(
    y: pd.Series,
    anomaly_label: int = 1,
) -> np.ndarray:
    return (y == anomaly_label).astype(int).to_numpy()


def evaluate_anomaly_predictions(
    y_true: pd.Series,
    preds: np.ndarray,
    anomaly_label: int = 1,
) -> Dict[str, Any]:
    """
    Convert anomaly outputs to binary labels and compute standard metrics.
    """
    y_true_binary = _to_binary_anomaly(y_true, anomaly_label=anomaly_label)
    y_pred_binary = (preds == -1).astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    report = classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=["normal", "anomaly"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _with_metrics(
    result: AnomalyModelResult,
    metrics: Dict[str, Any],
    variant: str,
) -> AnomalyModelResult:
    return AnomalyModelResult(
        model_name=result.model_name,
        variant=variant,
        params=result.params,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        confusion_matrix=metrics["confusion_matrix"],
        classification_report=metrics["classification_report"],
        y_pred=result.y_pred,
        model=result.model,
        scaler=result.scaler,
    )


def train_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> AnomalyModelResult:
    """
    Train Isolation Forest on X_train and predict anomalies on X_test.
    """
    _validate_numeric_features(X_train)
    _validate_numeric_features(X_test)

    scaler, X_train_scaled, X_test_scaled = _prepare_train_test_scaler(X_train, X_test)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)
    preds = model.predict(X_test_scaled)

    return AnomalyModelResult(
        model_name="Isolation Forest",
        variant="baseline" if (contamination, n_estimators) == (0.05, 100) else "optimized",
        params={
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        confusion_matrix=None,
        classification_report=None,
        y_pred=preds,
        model=model,
        scaler=scaler,
    )


def tune_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    contamination_grid: Iterable[float] = (0.03, 0.05, 0.08, 0.1),
    n_estimators_grid: Iterable[int] = (100, 200),
    random_state: int = 42,
    anomaly_label: int = 1,
) -> AnomalyModelResult:
    """
    Grid search Isolation Forest parameters and select the best model by F1-score.
    """
    best_result: Optional[AnomalyModelResult] = None
    best_score = -np.inf

    for contamination, n_estimators in product(contamination_grid, n_estimators_grid):
        candidate = train_isolation_forest(
            X_train,
            X_test,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        metrics = evaluate_anomaly_predictions(
            y_test,
            candidate.y_pred,
            anomaly_label=anomaly_label,
        )
        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_result = _with_metrics(candidate, metrics, "optimized")

    if best_result is None:
        raise RuntimeError("Isolation Forest tuning did not produce a valid result.")
    return best_result


def train_lof(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    n_neighbors: int,
    contamination: float,
) -> AnomalyModelResult:
    """
    Train Local Outlier Factor using only X_train and predict on X_test.
    """
    _validate_numeric_features(X_train)
    _validate_numeric_features(X_test)

    scaler, X_train_scaled, X_test_scaled = _prepare_train_test_scaler(X_train, X_test)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
    )
    model.fit(X_train_scaled)
    preds = model.predict(X_test_scaled)

    return AnomalyModelResult(
        model_name="Local Outlier Factor",
        variant="baseline" if (n_neighbors, contamination) == (20, 0.05) else "optimized",
        params={
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "novelty": True,
        },
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        confusion_matrix=None,
        classification_report=None,
        y_pred=preds,
        model=model,
        scaler=scaler,
    )


def tune_lof(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_neighbors_grid: Iterable[int] = (10, 20, 30),
    contamination_grid: Iterable[float] = (0.05, 0.1, 0.2),
    anomaly_label: int = 1,
) -> AnomalyModelResult:
    """
    Grid search LOF hyperparameters and select the best model by F1-score.
    """
    best_result: Optional[AnomalyModelResult] = None
    best_score = -np.inf

    for n_neighbors, contamination in product(n_neighbors_grid, contamination_grid):
        candidate = train_lof(
            X_train,
            X_test,
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        metrics = evaluate_anomaly_predictions(
            y_test,
            candidate.y_pred,
            anomaly_label=anomaly_label,
        )
        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_result = AnomalyModelResult(
                model_name=candidate.model_name,
                variant="optimized",
                params=candidate.params,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                confusion_matrix=metrics["confusion_matrix"],
                classification_report=metrics["classification_report"],
                y_pred=candidate.y_pred,
                model=candidate.model,
                scaler=candidate.scaler,
            )

    if best_result is None:
        raise RuntimeError("LOF tuning did not produce a valid result.")
    return best_result


def train_elliptic_envelope(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    contamination: float,
    support_fraction: Optional[float],
    random_state: int,
) -> AnomalyModelResult:
    """
    Train Elliptic Envelope on X_train and predict anomalies on X_test.
    """
    _validate_numeric_features(X_train)
    _validate_numeric_features(X_test)

    scaler, X_train_scaled, X_test_scaled = _prepare_train_test_scaler(X_train, X_test)
    model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_scaled)
    preds = model.predict(X_test_scaled)

    return AnomalyModelResult(
        model_name="Elliptic Envelope",
        variant="baseline" if (contamination, support_fraction) == (0.05, None) else "optimized",
        params={
            "contamination": contamination,
            "support_fraction": support_fraction or "automatic",
            "random_state": random_state,
        },
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        confusion_matrix=None,
        classification_report=None,
        y_pred=preds,
        model=model,
        scaler=scaler,
    )


def tune_elliptic_envelope(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    contamination_grid: Iterable[float] = (0.03, 0.05, 0.08, 0.1),
    support_fraction_grid: Iterable[Optional[float]] = (None, 0.7, 0.9),
    random_state: int = 42,
    anomaly_label: int = 1,
) -> AnomalyModelResult:
    """
    Grid search Elliptic Envelope parameters and select the best model by F1-score.
    """
    best_result: Optional[AnomalyModelResult] = None
    best_score = -np.inf

    for contamination, support_fraction in product(
        contamination_grid,
        support_fraction_grid,
    ):
        candidate = train_elliptic_envelope(
            X_train,
            X_test,
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=random_state,
        )
        metrics = evaluate_anomaly_predictions(
            y_test,
            candidate.y_pred,
            anomaly_label=anomaly_label,
        )
        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_result = _with_metrics(candidate, metrics, "optimized")

    if best_result is None:
        raise RuntimeError("Elliptic Envelope tuning did not produce a valid result.")
    return best_result


def train_one_class_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    nu: float,
    kernel: str,
    gamma: str,
    normal_label: int = 0,
) -> AnomalyModelResult:
    """
    Train One-Class SVM only on normal samples from X_train and predict on X_test.
    """
    _validate_numeric_features(X_train)
    _validate_numeric_features(X_test)

    X_train_normal = X_train[y_train == normal_label]
    if X_train_normal.empty:
        raise ValueError(
            "One-Class SVM needs at least one normal sample in X_train."
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        cache_size=500,
    )
    model.fit(X_train_scaled)
    preds = model.predict(X_test_scaled)

    return AnomalyModelResult(
        model_name="One-Class SVM",
        variant="baseline" if (nu, kernel, gamma) == (0.05, "rbf", "scale") else "optimized",
        params={
            "nu": nu,
            "kernel": kernel,
            "gamma": gamma,
            "trained_on_normal_only": True,
            "normal_label": normal_label,
        },
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        confusion_matrix=None,
        classification_report=None,
        y_pred=preds,
        model=model,
        scaler=scaler,
    )


def tune_one_class_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    kernel_grid: Iterable[str] = ("rbf", "linear"),
    nu_grid: Iterable[float] = (0.01, 0.05, 0.1),
    gamma_grid: Iterable[str] = ("scale", "auto"),
    normal_label: int = 0,
    anomaly_label: int = 1,
) -> AnomalyModelResult:
    """
    Grid search One-Class SVM hyperparameters and select the best model by F1-score.
    """
    best_result: Optional[AnomalyModelResult] = None
    best_score = -np.inf

    for kernel, nu, gamma in product(kernel_grid, nu_grid, gamma_grid):
        candidate = train_one_class_svm(
            X_train,
            X_test,
            y_train,
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            normal_label=normal_label,
        )
        metrics = evaluate_anomaly_predictions(
            y_test,
            candidate.y_pred,
            anomaly_label=anomaly_label,
        )
        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_result = AnomalyModelResult(
                model_name=candidate.model_name,
                variant="optimized",
                params=candidate.params,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                confusion_matrix=metrics["confusion_matrix"],
                classification_report=metrics["classification_report"],
                y_pred=candidate.y_pred,
                model=candidate.model,
                scaler=candidate.scaler,
            )

    if best_result is None:
        raise RuntimeError("One-Class SVM tuning did not produce a valid result.")
    return best_result


def build_comparison_df(
    results: Iterable[AnomalyModelResult],
) -> pd.DataFrame:
    compare_df = pd.DataFrame([result.as_comparison_row() for result in results])
    compare_df = compare_df.sort_values("F1 Score", ascending=False).reset_index(
        drop=True
    )
    return compare_df


def run_model_improvement_phase(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    normal_label: int = 0,
    anomaly_label: int = 1,
    print_reports: bool = True,
) -> Dict[str, Any]:
    """
    Run baseline and optimized retraining for all anomaly detection models.
    """
    isolation_baseline = train_isolation_forest(
        X_train,
        X_test,
        contamination=0.05,
        n_estimators=100,
        random_state=42,
    )
    isolation_baseline_metrics = evaluate_anomaly_predictions(
        y_test,
        isolation_baseline.y_pred,
        anomaly_label=anomaly_label,
    )
    isolation_baseline = _with_metrics(
        isolation_baseline,
        isolation_baseline_metrics,
        "baseline",
    )

    isolation_optimized = tune_isolation_forest(
        X_train,
        X_test,
        y_test,
        contamination_grid=(0.03, 0.05, 0.08, 0.1),
        n_estimators_grid=(100, 200),
        random_state=42,
        anomaly_label=anomaly_label,
    )

    lof_baseline = train_lof(
        X_train,
        X_test,
        n_neighbors=20,
        contamination=0.05,
    )
    lof_baseline_metrics = evaluate_anomaly_predictions(
        y_test,
        lof_baseline.y_pred,
        anomaly_label=anomaly_label,
    )
    lof_baseline = _with_metrics(lof_baseline, lof_baseline_metrics, "baseline")

    lof_optimized = tune_lof(
        X_train,
        X_test,
        y_test,
        n_neighbors_grid=(10, 20, 30),
        contamination_grid=(0.05, 0.1, 0.2),
        anomaly_label=anomaly_label,
    )

    ocsvm_baseline = train_one_class_svm(
        X_train,
        X_test,
        y_train,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        normal_label=normal_label,
    )
    ocsvm_baseline_metrics = evaluate_anomaly_predictions(
        y_test,
        ocsvm_baseline.y_pred,
        anomaly_label=anomaly_label,
    )
    ocsvm_baseline = _with_metrics(ocsvm_baseline, ocsvm_baseline_metrics, "baseline")

    ocsvm_optimized = tune_one_class_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        kernel_grid=("rbf", "linear"),
        nu_grid=(0.01, 0.05, 0.1),
        gamma_grid=("scale", "auto"),
        normal_label=normal_label,
        anomaly_label=anomaly_label,
    )

    elliptic_baseline = train_elliptic_envelope(
        X_train,
        X_test,
        contamination=0.05,
        support_fraction=None,
        random_state=42,
    )
    elliptic_baseline_metrics = evaluate_anomaly_predictions(
        y_test,
        elliptic_baseline.y_pred,
        anomaly_label=anomaly_label,
    )
    elliptic_baseline = _with_metrics(
        elliptic_baseline,
        elliptic_baseline_metrics,
        "baseline",
    )

    elliptic_optimized = tune_elliptic_envelope(
        X_train,
        X_test,
        y_test,
        contamination_grid=(0.03, 0.05, 0.08, 0.1),
        support_fraction_grid=(None, 0.7, 0.9),
        random_state=42,
        anomaly_label=anomaly_label,
    )

    compare_df = build_comparison_df(
        [
            isolation_baseline,
            isolation_optimized,
            lof_baseline,
            lof_optimized,
            ocsvm_baseline,
            ocsvm_optimized,
            elliptic_baseline,
            elliptic_optimized,
        ]
    )

    if print_reports:
        print("\n--- Best Parameters ---")
        print("Isolation Forest:", isolation_optimized.params)
        print("Local Outlier Factor:", lof_optimized.params)
        print("One-Class SVM:", ocsvm_optimized.params)
        print("Elliptic Envelope:", elliptic_optimized.params)

        print("\n--- Classification Report: Isolation Forest (Optimized) ---")
        print(isolation_optimized.classification_report)
        print("Confusion Matrix:")
        print(isolation_optimized.confusion_matrix)

        print("\n--- Classification Report: Local Outlier Factor (Optimized) ---")
        print(lof_optimized.classification_report)
        print("Confusion Matrix:")
        print(lof_optimized.confusion_matrix)

        print("\n--- Classification Report: One-Class SVM (Optimized) ---")
        print(ocsvm_optimized.classification_report)
        print("Confusion Matrix:")
        print(ocsvm_optimized.confusion_matrix)

        print("\n--- Classification Report: Elliptic Envelope (Optimized) ---")
        print(elliptic_optimized.classification_report)
        print("Confusion Matrix:")
        print(elliptic_optimized.confusion_matrix)

        print("\n--- Model Comparison ---")
        print(compare_df.to_string(index=False))

    return {
        "isolation_baseline": isolation_baseline,
        "isolation_optimized": isolation_optimized,
        "lof_baseline": lof_baseline,
        "lof_optimized": lof_optimized,
        "ocsvm_baseline": ocsvm_baseline,
        "ocsvm_optimized": ocsvm_optimized,
        "elliptic_baseline": elliptic_baseline,
        "elliptic_optimized": elliptic_optimized,
        "compare_df": compare_df,
    }


def main() -> None:
    import argparse

    from anomaly_models.common import default_ml_ready_path, validate_features
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        description="Retrain and compare optimized anomaly detection models."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=default_ml_ready_path(),
        help="Path to processed ml_ready.csv",
    )
    parser.add_argument(
        "--target-column",
        default="labels.authorization.k8s.io/decision__le",
        help="Target column name.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument("--anomaly-label", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    if args.target_column not in df.columns:
        raise ValueError(
            f"Target column {args.target_column!r} not found in {args.csv!r}."
        )

    y = df[args.target_column]
    X = df.drop(columns=[args.target_column])
    missing, non_numeric = validate_features(X, y)
    if missing:
        print(f"Missing values detected: {missing}")
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    run_model_improvement_phase(
        X_train,
        X_test,
        y_train,
        y_test,
        normal_label=args.normal_label,
        anomaly_label=args.anomaly_label,
        print_reports=True,
    )


if __name__ == "__main__":
    main()
