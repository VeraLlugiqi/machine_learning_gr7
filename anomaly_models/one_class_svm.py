"""
Trajnim dhe finalizim: One-Class SVM.

Ruajtja e modelit: models/one_class_svm/
Ruajtja e rezultateve: processedfiles/one_class_svm_results.csv
"""
import argparse
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_models.common import (
    default_ml_ready_path,
    export_predictions,
    load_ml_ready,
    project_root,
    validate_features,
)


def train_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    nu: float,
    kernel: str,
    gamma: str,
    train_only_normal: bool,
    normal_label: int,
) -> Tuple[OneClassSVM, StandardScaler, np.ndarray]:
    if train_only_normal:
        X_train = X[y == normal_label]
    else:
        X_train = X

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)
    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        cache_size=500,
    )
    model.fit(X_train_scaled)
    preds = model.predict(X_scaled)
    return model, scaler, preds


def run_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    kernel: str,
    gamma: str,
    train_only_normal: bool,
    normal_label: int,
) -> None:
    nus = [0.01, 0.03, 0.05, 0.1]
    print("\n--- Sweep nu (One-Class SVM) ---")
    for nu in nus:
        _, _, preds = train_and_predict(
            X, y, nu, kernel, gamma, train_only_normal, normal_label
        )
        n_anom = int(np.sum(preds == -1))
        print(f"nu={nu}: anomaly count = {n_anom}")
        ct = pd.crosstab(y, preds, rownames=["y (decision__le)"], colnames=["pred"])
        print(ct)
        print()


def save_artifacts(
    model: OneClassSVM,
    scaler: StandardScaler,
    feature_names: list,
) -> str:
    out_dir = os.path.join(project_root(), "models", "one_class_svm")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "one_class_svm.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "standard_scaler.joblib"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_columns.joblib"))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Faza 2 — One-Class SVM: trajnim mbi processedfiles/ml_ready.csv"
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=default_ml_ready_path(),
        help="Rruga te ml_ready.csv",
    )
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--kernel", type=str, default="rbf")
    p.add_argument("--gamma", type=str, default="scale")
    p.add_argument(
        "--train-on-normal-only",
        action="store_true",
        default=True,
        help="Trajnon modelin vetëm mbi rastet normale (allow=0).",
    )
    p.add_argument(
        "--use-all-data",
        action="store_false",
        dest="train_on_normal_only",
        help="Trajnon modelin mbi të gjitha rreshtat e X.",
    )
    p.add_argument("--normal-label", type=int, default=0)
    p.add_argument("--anomaly-label", type=int, default=1)
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
        run_sweep(
            X,
            y,
            args.kernel,
            args.gamma,
            args.train_on_normal_only,
            args.normal_label,
        )
        return

    model, scaler, preds = train_and_predict(
        X,
        y,
        args.nu,
        args.kernel,
        args.gamma,
        args.train_on_normal_only,
        args.normal_label,
    )
    ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])

    print("\n--- One-Class SVM ---")
    print("nu:", args.nu)
    print("kernel:", args.kernel)
    print("gamma:", args.gamma)
    print("train_on_normal_only:", args.train_on_normal_only)
    print("normal_label:", args.normal_label, "| anomaly_label:", args.anomaly_label)
    if args.train_on_normal_only:
        print("Train rows used:", int((y == args.normal_label).sum()))
    print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
    print("Normal count (pred == 1):", int(np.sum(preds == 1)))
    print("\n--- Crosstab: target vs pred ---")
    print(ct)
    print("\n--- Metrics for forbid (label=1) ---")
    print(classification_report(y, (preds == -1).astype(int), digits=4))
    print("Confusion matrix [true rows: 0/1, pred cols: normal/anomaly]")
    print(confusion_matrix(y, (preds == -1).astype(int)))

    if args.export_results:
        results_path, anomalies_path = export_predictions(X, y, preds, "one_class_svm")
        print("\nRezultatet u ruajtën në:", results_path)
        print("Vetëm anomalitë u ruajtën në:", anomalies_path)

    if args.save:
        out_dir = save_artifacts(model, scaler, list(X.columns))
        print("\nRuajtur në:", out_dir)


if __name__ == "__main__":
    main()
