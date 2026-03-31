import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_DT_HINT = re.compile(r"date|time|timestamp", re.I)
PRINCIPAL_EMAIL_COL = "protoPayload.authenticationInfo.principalEmail"


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_empty_rows_cols(self) -> "DataCleaner":
        self.df.dropna(how="all", inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)
        return self

    def fix_datetime_columns(self) -> "DataCleaner":
        for col in self.df.columns:
            if not _DT_HINT.search(col):
                continue
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce", utc=True)
        return self

    def normalize_string_columns(self) -> "DataCleaner":
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            self.df[col] = self.df[col].replace("", np.nan)
        return self

    def clean_boolean_columns(self, fill_value: bool = False) -> "DataCleaner":
        for col in self.df.select_dtypes(include=["bool"]).columns:
            self.df[col] = self.df[col].fillna(fill_value)
        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> "DataCleaner":
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self

    
    def feature_selection(self):
        cols_to_drop = []

        manual_drop = [
            "insertId",
            "operation.id",
            "operation.producer",
            "operation.first",
            "operation.last"
        ]

        for col in self.df.columns:
            if col == "labels.authorization.k8s.io/decision":
                continue

            if self.df[col].nunique(dropna=False) <= 1:
                cols_to_drop.append(col)

            elif col in manual_drop:
                cols_to_drop.append(col)

        cols_to_drop = list(set(cols_to_drop))

        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} low-value columns: {cols_to_drop}")
            self.df.drop(columns=cols_to_drop, inplace=True)

        return self  


def drop_columns_over_missing_fraction(
    df: pd.DataFrame, missing_threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[str]]:
    frac = df.isna().mean()
    drop_cols = frac[frac > missing_threshold].index.tolist()
    out = df.drop(columns=drop_cols, errors="ignore")
    return out, drop_cols


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not s.isna().any():
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                out[col] = s.fillna(s.median())
        elif is_bool_dtype(s):
            m = s.mode()
            out[col] = s.fillna(m.iloc[0] if len(m) else False)
        elif is_numeric_dtype(s):
            med = s.median()
            if pd.notna(med):
                if pd.api.types.is_integer_dtype(s):
                    out[col] = s.fillna(int(round(float(med))))
                else:
                    out[col] = s.fillna(med)
        else:
            m = s.mode()
            out[col] = s.fillna(m.iloc[0] if len(m) else "missing")
    return out

def remove_outliers_iqr(
    df: pd.DataFrame, factor: float = 1.5, min_unique: int = 5
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    outlier_mask_any = pd.Series(False, index=out.index)
    by_column: Dict[str, int] = {}

    for col in numeric_cols:
        if out[col].nunique(dropna=True) < min_unique:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        col_mask = (out[col] < lower) | (out[col] > upper)
        count = int(col_mask.sum())
        if count > 0:
            by_column[col] = count
            outlier_mask_any = outlier_mask_any | col_mask

    total_removed = int(outlier_mask_any.sum())
    cleaned = out.loc[~outlier_mask_any].copy()
    report = {
        "rows_before": int(len(out)),
        "rows_removed": total_removed,
        "rows_after": int(len(cleaned)),
        "by_column": by_column,
    }
    return cleaned, report

def add_datetime_epoch_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for col in list(out.columns):
        if not pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        new_name = f"{col}__epoch_s"
        if new_name in out.columns:
            continue
        s = pd.to_datetime(out[col], utc=True, errors="coerce")
        delta = s - epoch
        out[new_name] = delta.dt.total_seconds()
    return out


def add_time_features_and_finalize_numeric(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    receive_col: str = "receiveTimestamp",
    drop_text_columns: bool = True,
    long_text_mean_len: float = 120.0,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = df.copy()
    dropped: Dict[str, List[str]] = {
        "id_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "long_text_columns": [],
        "redundant_numeric_columns": [],
    }

    # Temporal features from the main timestamp column.
    if timestamp_col in out.columns:
        ts = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
        out["hour"] = ts.dt.hour.fillna(0).astype(np.int16)
        out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(np.int16)
        out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)

    ts_epoch_col = f"{timestamp_col}__epoch_s"
    recv_epoch_col = f"{receive_col}__epoch_s"
    if ts_epoch_col in out.columns and recv_epoch_col in out.columns:
        out["timestamp_delay_s"] = out[recv_epoch_col] - out[ts_epoch_col]
        out.drop(columns=[recv_epoch_col], inplace=True, errors="ignore")
        dropped["redundant_numeric_columns"].append(recv_epoch_col)

    # Drop ID-like columns (even if still present).
    id_cols = [c for c in ["insertId", "operation.id"] if c in out.columns]
    if id_cols:
        out.drop(columns=id_cols, inplace=True, errors="ignore")
        dropped["id_columns"].extend(id_cols)

    # Drop original datetime columns after extracting numeric features.
    dt_cols = list(out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    extra_dt_named = [c for c in [timestamp_col, receive_col] if c in out.columns and c not in dt_cols]
    dt_to_drop = sorted(set(dt_cols + extra_dt_named))
    if dt_to_drop:
        out.drop(columns=dt_to_drop, inplace=True, errors="ignore")
        dropped["datetime_columns"].extend(dt_to_drop)

    if drop_text_columns:
        text_cols = list(out.select_dtypes(include=["object"]).columns)
        long_text_cols: List[str] = []
        le_backed_text_cols: List[str] = []
        for col in text_cols:
            non_na = out[col].dropna().astype(str)
            mean_len = float(non_na.str.len().mean()) if len(non_na) else 0.0
            if mean_len > long_text_mean_len:
                long_text_cols.append(col)
            if f"{col}__le" in out.columns:
                le_backed_text_cols.append(col)

        to_drop_text = sorted(set(text_cols))
        if to_drop_text:
            out.drop(columns=to_drop_text, inplace=True, errors="ignore")
            dropped["text_columns"].extend(to_drop_text)
            dropped["long_text_columns"].extend(sorted(set(long_text_cols)))

    # Keep model input numeric-only (bool -> int8).
    for col in out.select_dtypes(include=["bool"]).columns:
        out[col] = out[col].astype(np.int8)

    return out, dropped


def add_anonymous_principal_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if PRINCIPAL_EMAIL_COL not in out.columns:
        return out
    s = out[PRINCIPAL_EMAIL_COL].astype(str)
    out["anonymous_principal"] = (
        s.str.contains("system:anonymous", case=False, regex=False, na=False).astype(np.int8)
    )
    return out



def add_selective_label_encoding(
    df: pd.DataFrame,
    min_unique: int = 2,
    max_unique: int = 35,
    max_mean_strlen: float = 100.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
    out = df.copy()
    mappings: Dict[str, Dict[int, str]] = {}
    skip_le = {PRINCIPAL_EMAIL_COL}
    for col in list(out.select_dtypes(include=["object"]).columns):
        if col in skip_le:
            continue
        new_col = f"{col}__le"
        if new_col in out.columns:
            continue
        series = out[col]
        valid = series.dropna()
        if len(valid) == 0:
            continue
        strv = valid.astype(str)
        if strv.str.len().mean() > max_mean_strlen:
            continue
        nuniq = series.nunique(dropna=True)
        if nuniq < min_unique or nuniq > max_unique:
            continue
        codes, uniques = pd.factorize(series, sort=True, use_na_sentinel=True)
        out[new_col] = codes.astype(np.int32)
        mappings[new_col] = {int(i): str(v) for i, v in enumerate(uniques.tolist())}
        mappings[new_col][-1] = "__MISSING__"
    return out, mappings


def save_label_mappings_txt(mappings: Dict[str, Dict[int, str]], path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    lines: List[str] = []
    for col in sorted(mappings.keys()):
        lines.append(f"[{col}]")
        for code, label in sorted(mappings[col].items(), key=lambda x: x[0]):
            lines.append(f"{code}\t{label}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def save_ml_csv(df: pd.DataFrame, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False)
