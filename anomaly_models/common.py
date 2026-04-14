"""
Përgatitje e përbashkët për të gjitha modelet e Fazës 2 (ngarkim nga ml_ready.csv
dhe eksport i rezultateve).
"""
import os
from typing import List, Tuple

import pandas as pd

TARGET_COL = "labels.authorization.k8s.io/decision__le"


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_ml_ready_path() -> str:
    return os.path.join(project_root(), "processedfiles", "ml_ready.csv")


def processed_dir() -> str:
    return os.path.join(project_root(), "processedfiles")


def load_ml_ready(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Mungon kolona e target-it: {TARGET_COL!r}. "
            "Ekzekuto fillimisht data.py për të gjeneruar ml_ready.csv."
        )
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def validate_features(X: pd.DataFrame, y: pd.Series) -> Tuple[int, List[str]]:
    missing = int(X.isnull().sum().sum() + y.isnull().sum())
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return missing, non_numeric


def export_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    preds,
    method_name: str,
) -> Tuple[str, str]:
    """
    Ruaj rezultatet e parashikimeve në `processedfiles/` me emra të veçantë
    për secilin algoritëm.
    """
    out_dir = processed_dir()
    os.makedirs(out_dir, exist_ok=True)

    results = X.copy()
    results["anomaly"] = preds
    results["target"] = y.values

    safe_name = method_name.lower().replace(" ", "_")
    results_path = os.path.join(out_dir, f"{safe_name}_results.csv")
    anomalies_path = os.path.join(out_dir, f"{safe_name}_anomalies_only.csv")

    results.to_csv(results_path, index=False)
    results[results["anomaly"] == -1].to_csv(anomalies_path, index=False)
    return results_path, anomalies_path
