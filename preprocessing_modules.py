import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_DT_HINT = re.compile(r"date|time|timestamp", re.I)
PRINCIPAL_EMAIL_COL = "protoPayload.authenticationInfo.principalEmail"
CALLER_IP_COL = "protoPayload.requestMetadata.callerIp"


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


def add_selective_label_encoding(
    df: pd.DataFrame,
    min_unique: int = 2,
    max_unique: int = 35,
    max_mean_strlen: float = 100.0,
) -> pd.DataFrame:
    out = df.copy()
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
        codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
        out[new_col] = codes.astype(np.int32)
    return out


def save_ml_csv(df: pd.DataFrame, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False)
