"""
Trajnim dhe finalizim: vetëm Isolation Forest.

Ruajtja: models/isolation_forest/
"""
import argparse
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_models.common import (
    default_ml_ready_path,
    export_predictions,
    load_ml_ready,
    project_root,
    validate_features,
)


def train_and_predict(
    X: pd.DataFrame,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> Tuple[IsolationForest, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    preds = model.predict(X_scaled)
    return model, scaler, preds


def run_sweep(
    X: pd.DataFrame, y: pd.Series, n_estimators: int, random_state: int
) -> None:
    contaminations = [0.01, 0.03, 0.05, 0.1]
    print("\n--- Sweep contamination (Isolation Forest) ---")
    for c in contaminations:
        _, _, preds = train_and_predict(X, c, n_estimators, random_state)
        n_anom = int(np.sum(preds == -1))
        print(f"contamination={c}: anomaly count = {n_anom}")
        ct = pd.crosstab(y, preds, rownames=["y (decision__le)"], colnames=["pred"])
        print(ct)
        print()


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    feature_names: list,
) -> str:
    root = project_root()
    out_dir = os.path.join(root, "models", "isolation_forest")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "standard_scaler.joblib"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_columns.joblib"))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Faza 2 — Isolation Forest: trajnim mbi processedfiles/ml_ready.csv"
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=default_ml_ready_path(),
        help="Rruga te ml_ready.csv",
    )
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--export-results", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print("Skedari nuk u gjet:", args.csv, file=sys.stderr)
        sys.exit(1)

    X, y = load_ml_ready(args.csv)
    missing, non_numeric = validate_features(X, y)
    print("--- Kontroll dataset ---")
    print("Shape X:", X.shape, "| features:", list(X.columns))
    print("Missing values (gjithsej):", missing)
    if non_numeric:
        print("KUJDES: kolona jo-numerike në X:", non_numeric, file=sys.stderr)
        sys.exit(1)
    print("Të gjitha kolonat e X janë numerike: OK")

    if args.sweep:
        run_sweep(X, y, args.n_estimators, args.random_state)
        return

    model, scaler, preds = train_and_predict(
        X, args.contamination, args.n_estimators, args.random_state
    )
    ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])

    print("\n--- Isolation Forest ---")
    print("contamination:", args.contamination)
    print("n_estimators:", args.n_estimators)
    print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
    print("Normal count (pred == 1):", int(np.sum(preds == 1)))
    print("\n--- Crosstab: target vs pred ---")
    print(ct)

    if args.export_results:
        results_path, anomalies_path = export_predictions(
            X, y, preds, "isolation_forest"
        )
        print("\nRezultatet u ruajtën në:", results_path)
        print("Vetëm anomalitë u ruajtën në:", anomalies_path)

    if args.save:
        out_dir = save_artifacts(model, scaler, list(X.columns))
        print("\nRuajtur në:", out_dir)


if __name__ == "__main__":
    main()
