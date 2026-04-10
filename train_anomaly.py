"""
Trajnim i Isolation Forest për anomaly detection (jo-supervised).
Target-i përdoret vetëm për vlerësim / krahasim, jo për fit().
"""
import argparse
import json
import os
import sys
from datetime import datetime
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


def save_comparison_plots(df_results: pd.DataFrame, out_dir: str) -> List[str]:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    normal = df_results[df_results["anomaly"] == 1]
    anomaly = df_results[df_results["anomaly"] == -1]
    paths: List[str] = []

    # 1) Overlay histogram for delay
    plt.figure(figsize=(10, 5))
    plt.hist(normal["timestamp_delay_s"], bins=50, alpha=0.5, label="normal")
    plt.hist(anomaly["timestamp_delay_s"], bins=50, alpha=0.5, label="anomaly")
    plt.title("timestamp_delay_s: normal vs anomaly")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "plot_delay_normal_vs_anomaly.png")
    plt.savefig(p1, dpi=140)
    plt.close()
    paths.append(p1)

    # 2) Day of week counts
    day_counts = pd.DataFrame({
        "normal": normal["dayofweek"].value_counts().sort_index(),
        "anomaly": anomaly["dayofweek"].value_counts().sort_index(),
    }).fillna(0)
    day_counts.plot(kind="bar", figsize=(10, 5), title="dayofweek counts: normal vs anomaly")
    plt.xlabel("dayofweek")
    plt.ylabel("count")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "plot_dayofweek_counts.png")
    plt.savefig(p2, dpi=140)
    plt.close()
    paths.append(p2)

    # 3) Status code counts (top classes)
    status_counts = pd.DataFrame({
        "normal": normal["protoPayload.status.code"].value_counts(),
        "anomaly": anomaly["protoPayload.status.code"].value_counts(),
    }).fillna(0).sort_values("anomaly", ascending=False).head(10)
    status_counts.plot(kind="bar", figsize=(10, 5), title="status.code top counts: normal vs anomaly")
    plt.xlabel("protoPayload.status.code")
    plt.ylabel("count")
    plt.tight_layout()
    p3 = os.path.join(out_dir, "plot_status_code_top10.png")
    plt.savefig(p3, dpi=140)
    plt.close()
    paths.append(p3)

    return paths


def export_detailed_analysis(
    root: str,
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    contamination: float,
    n_estimators: int,
    random_state: int,
    generate_plots: bool,
) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "analysis_outputs", f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    df_results, anomalies = build_analysis_tables(X, y, preds)
    normal = df_results[df_results["anomaly"] == 1].copy()

    # Main exported tables
    df_results.to_csv(os.path.join(out_dir, "results_all_rows.csv"), index=False)
    anomalies.to_csv(os.path.join(out_dir, "results_anomalies_only.csv"), index=False)
    normal.to_csv(os.path.join(out_dir, "results_normal_only.csv"), index=False)

    # Crosstab
    ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])
    ct.to_csv(os.path.join(out_dir, "crosstab_target_vs_pred.csv"))

    # Means and deltas
    num_cols = [
        c for c in df_results.columns
        if c not in {"anomaly", "target"} and pd.api.types.is_numeric_dtype(df_results[c])
    ]
    mean_comp = pd.DataFrame({
        "normal_mean": normal[num_cols].mean(numeric_only=True),
        "anomaly_mean": anomalies[num_cols].mean(numeric_only=True),
    })
    mean_comp["delta_anom_minus_normal"] = mean_comp["anomaly_mean"] - mean_comp["normal_mean"]
    mean_comp = mean_comp.sort_values("delta_anom_minus_normal", key=np.abs, ascending=False)
    mean_comp.to_csv(os.path.join(out_dir, "mean_comparison_normal_vs_anomaly.csv"))

    # Value counts for key features
    pd.DataFrame({
        "normal_count": normal["dayofweek"].value_counts().sort_index(),
        "anomaly_count": anomalies["dayofweek"].value_counts().sort_index(),
    }).fillna(0).to_csv(os.path.join(out_dir, "counts_dayofweek.csv"))

    pd.DataFrame({
        "normal_count": normal["hour"].value_counts().sort_index(),
        "anomaly_count": anomalies["hour"].value_counts().sort_index(),
    }).fillna(0).to_csv(os.path.join(out_dir, "counts_hour.csv"))

    pd.DataFrame({
        "normal_count": normal["protoPayload.status.code"].value_counts().sort_index(),
        "anomaly_count": anomalies["protoPayload.status.code"].value_counts().sort_index(),
    }).fillna(0).to_csv(os.path.join(out_dir, "counts_status_code.csv"))

    # Minimal run metadata
    config = {
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "rows_total": int(len(df_results)),
        "rows_anomaly": int((preds == -1).sum()),
        "rows_normal": int((preds == 1).sum()),
        "features": list(X.columns),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if generate_plots:
        plots_dir = os.path.join(out_dir, "plots")
        save_comparison_plots(df_results, plots_dir)

    return out_dir


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
    p.add_argument(
        "--deep-analysis",
        action="store_true",
        help="Ruaj analizë të plotë dhe grafe në analysis_outputs/run_*/",
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

    if args.deep_analysis:
        try:
            out_dir = export_detailed_analysis(
                root=root,
                X=X,
                y=y,
                preds=preds,
                contamination=args.contamination,
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                generate_plots=True,
            )
            print("\nRuajtur analiza e plotë në:", out_dir)
        except Exception as exc:
            print("\nNuk u ruajt analiza e plotë:", exc, file=sys.stderr)

    if args.save:
        out_dir = os.path.join(root, "models")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, os.path.join(out_dir, "isolation_forest.joblib"))
        joblib.dump(scaler, os.path.join(out_dir, "standard_scaler.joblib"))
        joblib.dump(list(X.columns), os.path.join(out_dir, "feature_columns.joblib"))
        print("\nRuajtur në:", out_dir)


if __name__ == "__main__":
    main()
