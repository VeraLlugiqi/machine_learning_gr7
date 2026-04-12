"""
Përgatitje e përbashkët për të gjitha modelet e Fazës 2 (ngarkim nga ml_ready.csv).
"""
from typing import List, Tuple

import pandas as pd

TARGET_COL = "labels.authorization.k8s.io/decision__le"


def project_root() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_ml_ready_path() -> str:
    import os

    return os.path.join(project_root(), "processedfiles", "ml_ready.csv")


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
