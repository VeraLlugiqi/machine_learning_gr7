"""
Trajnim i Isolation Forest për anomaly detection (jo-supervised).
Target-i përdoret vetëm për vlerësim / krahasim, jo për fit().
"""
import argparse
import os
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

TARGET_COL = "labels.authorization.k8s.io/decision__le"


def load_ml_ready(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Mungon kolona e target-it: {TARGET_COL!r}. "
            "Ekzekuto fillimisht data.py për të gjeneruar ml_ready.csv."
        )
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y, df


def validate_features(X: pd.DataFrame, y: pd.Series) -> Tuple[int, List[str]]:
    missing = int(X.isnull().sum().sum() + y.isnull().sum())
    non_numeric = [
        c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
    ]
    return missing, non_numeric


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


def build_analysis_tables(X: pd.DataFrame, y: pd.Series, preds: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_results = X.copy()
    df_results["anomaly"] = preds
    df_results["target"] = y.values
    return df_results, df_results[df_results["anomaly"] == -1].copy()


def print_normal_vs_anomaly_stats(df_results: pd.DataFrame) -> None:
    normal = df_results[df_results["anomaly"] == 1]
    anomaly = df_results[df_results["anomaly"] == -1]
    num_cols = [
        c for c in df_results.columns
        if c not in {"anomaly", "target"} and pd.api.types.is_numeric_dtype(df_results[c])
    ]
    print("\n--- Mean comparison (normal vs anomaly) ---")
    comp = pd.DataFrame({
        "normal_mean": normal[num_cols].mean(numeric_only=True),
        "anomaly_mean": anomaly[num_cols].mean(numeric_only=True),
    })
    comp["delta_anom_minus_normal"] = comp["anomaly_mean"] - comp["normal_mean"]
    print(comp.sort_values("delta_anom_minus_normal", key=np.abs, ascending=False).head(12))


def save_delay_plot(df_results: pd.DataFrame, out_dir: str) -> str:
    # Optional dependency for visualization.
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "delay_distribution.png")
    plt.figure(figsize=(10, 5))
    plt.hist(df_results["timestamp_delay_s"], bins=50)
    plt.title("timestamp_delay_s distribution")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(root, "processedfiles", "ml_ready.csv")

    p = argparse.ArgumentParser(
        description="Isolation Forest anomaly detection mbi ml_ready.csv"
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=default_csv,
        help="Rruga te ml_ready.csv (default: processedfiles/ml_ready.csv)",
    )
    p.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Pritja e përqindjes së anomalive (default: 0.05)",
    )
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Provo disa vlera contamination dhe printo krahasim",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Ruaj modelin dhe scaler në models/",
    )
    p.add_argument(
        "--export-analysis",
        action="store_true",
        help="Ruaj rezultate/anomali/statistika në processedfiles/",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Ruaj grafik të delay distribution (kërkon matplotlib).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print("Skedari nuk u gjet:", args.csv, file=sys.stderr)
        sys.exit(1)

    X, y, _ = load_ml_ready(args.csv)
    missing, non_numeric = validate_features(X, y)

    print("--- Kontroll dataset ---")
    print("Shape X:", X.shape, "| features:", list(X.columns))
    print("Missing values (gjithsej):", missing)
    if non_numeric:
        print("KUJDES: kolona jo-numerike në X:", non_numeric, file=sys.stderr)
        sys.exit(1)
    print("Të gjitha kolonat e X janë numerike: OK")

    if args.sweep:
        contaminations = [0.01, 0.03, 0.05, 0.1]
        print("\n--- Sweep contamination ---")
        for c in contaminations:
            _, _, preds = train_and_predict(
                X, c, args.n_estimators, args.random_state
            )
            n_anom = int(np.sum(preds == -1))
            print(f"contamination={c}: anomaly count = {n_anom}")
            ct = pd.crosstab(y, preds, rownames=["y (decision__le)"], colnames=["pred"])
            print(ct)
            print()
        return

    model, scaler, preds = train_and_predict(
        X, args.contamination, args.n_estimators, args.random_state
    )

    print("\n--- Trajnim (Isolation Forest) ---")
    print("contamination:", args.contamination, "| n_estimators:", args.n_estimators)
    print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
    print("Normal count (pred == 1):", int(np.sum(preds == 1)))

    print("\n--- Crosstab: target vs pred ---")
    print("(y = label encoding i decision; pred: 1=normal, -1=anomaly)\n")
    ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])
    print(ct)
    df_results, anomalies = build_analysis_tables(X, y, preds)
    print("\n--- Sample anomalies ---")
    print(anomalies.head(5))
    print_normal_vs_anomaly_stats(df_results)

    analysis_dir = os.path.join(root, "processedfiles")
    if args.export_analysis:
        os.makedirs(analysis_dir, exist_ok=True)
        df_results.to_csv(os.path.join(analysis_dir, "anomaly_results.csv"), index=False)
        anomalies.to_csv(os.path.join(analysis_dir, "anomalies_only.csv"), index=False)
        print("\nRuajtur analiza në:", analysis_dir)
        print("- anomaly_results.csv")
        print("- anomalies_only.csv")

    if args.plot:
        try:
            plot_path = save_delay_plot(df_results, analysis_dir)
            print("\nRuajtur grafiku:", plot_path)
        except Exception as exc:
            print("\nNuk u krijua grafiku (matplotlib mungon ose dështoi):", exc)

    if args.save:
        out_dir = os.path.join(root, "models")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, os.path.join(out_dir, "isolation_forest.joblib"))
        joblib.dump(scaler, os.path.join(out_dir, "standard_scaler.joblib"))
        joblib.dump(list(X.columns), os.path.join(out_dir, "feature_columns.joblib"))
        print("\nRuajtur në:", out_dir)


if __name__ == "__main__":
    main()
