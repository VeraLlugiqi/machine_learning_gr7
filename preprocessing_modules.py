"""
Data Preprocessing Module - Refactored Classes
Includes: Quality Check, Data Cleaning, Type Detection, Aggregation, Sampling, 
Feature Engineering, Transformation, Discretization, Encoding, and PCA
"""

import os
import inspect
import warnings
import re
import ipaddress
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from pandas.api.types import is_numeric_dtype
from imblearn.over_sampling import SMOTE, ADASYN


# ============================================================================
# 1. DATA QUALITY CHECK
# ============================================================================

class DataQualityChecker:
    """Analyze data quality: missing values, duplicates, empty strings, datetime validity."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.report = {}
    
    def check_missing_values(self) -> pd.Series:
        """Count missing values per column."""
        return self.df.isnull().sum()
    
    def check_duplicates(self) -> int:
        """Count duplicate rows."""
        return self.df.duplicated().sum()
    
    def check_empty_strings(self) -> Dict[str, int]:
        """Count empty string values per column."""
        empty_strings = {}
        for col in self.df.columns:
            if self.df[col].dtype == object:
                empty_strings[col] = self.df[col].astype(str).str.strip().eq("").sum()
            else:
                empty_strings[col] = 0
        return empty_strings
    
    def check_datetime_validity(self) -> Dict[str, bool]:
        """Check if datetime columns are valid."""
        datetime_issues = {}
        for col in self.df.columns:
            if "time" in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    datetime_issues[col] = True
                except:
                    datetime_issues[col] = False
        return datetime_issues
    
    def get_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        missing = self.check_missing_values()
        duplicates = self.check_duplicates()
        empty_str = self.check_empty_strings()
        dt_validity = self.check_datetime_validity()
        
        self.report = {
            "total_rows": len(self.df),
            "total_cols": len(self.df.columns),
            "missing_values": missing.to_dict(),
            "duplicate_rows": duplicates,
            "empty_strings": empty_str,
            "datetime_validity": dt_validity,
            "complete_records": len(self.df.dropna()),
            "null_records": len(self.df) - len(self.df.dropna())
        }
        return self.report
    
    def print_report(self):
        """Print quality report."""
        report = self.get_full_report()
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        print(f"\nTotal Rows: {report['total_rows']}")
        print(f"Total Columns: {report['total_cols']}")
        print(f"Complete Records (no nulls): {report['complete_records']}")
        print(f"Records with Nulls: {report['null_records']}")
        print(f"\nDuplicate Rows: {report['duplicate_rows']}")
        print("\nMissing Values per Column:")
        for col, count in report['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
        print("\nEmpty Strings per Column:")
        for col, count in report['empty_strings'].items():
            if count > 0:
                print(f"  {col}: {count}")
        print("\nDatetime Validity:")
        for col, valid in report['datetime_validity'].items():
            status = "✓ Valid" if valid else "✗ Invalid"
            print(f"  {col}: {status}")
        print("="*70 + "\n")


# ============================================================================
# 2. TYPE DETECTION
# ============================================================================

class TypeDetector:
    """Automatically detect data types in CSV."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
    
    def detect(self) -> Dict[str, str]:
        """Detect and return data types for all columns."""
        df = pd.read_csv(self.csv_path, low_memory=False)
        
        # Try to convert to datetime
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        
        detected = {}
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                detected[col] = "int"
            elif pd.api.types.is_float_dtype(dtype):
                detected[col] = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                detected[col] = "bool"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                detected[col] = "datetime"
            else:
                detected[col] = "string"
        
        return detected


# ============================================================================
# 3. DATA CLEANING
# ============================================================================

class DataCleaner:
    """Clean dataset: remove nulls, fix dates, standardize strings, handle booleans."""
    
    def __init__(self, input_csv: str):
        self.input_csv = input_csv
        self.df = pd.read_csv(input_csv, low_memory=False)
    
    def remove_empty_rows_cols(self):
        """Remove completely empty rows and columns."""
        self.df.dropna(how="all", inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)
        return self
    
    def fix_datetime_columns(self):
        """Convert timestamp columns to datetime."""
        for col in self.df.columns:
            if "timestamp" in col.lower():
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
        return self
    
    def clean_string_columns(self):
        """Trim, lowercase, and handle nulls in string columns."""
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
            self.df[col] = self.df[col].replace({
                "nan": "unknown",
                "none": "unknown",
                "": "unknown"
            })
        return self
    
    def clean_boolean_columns(self, fill_value=False):
        """Fill null values in boolean columns."""
        for col in self.df.select_dtypes(include=["bool"]).columns:
            self.df[col] = self.df[col].fillna(fill_value)
        return self
    
    def clean_numeric_columns(self, fill_strategy="median"):
        """Fill null values in numeric columns."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if fill_strategy == "median":
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        elif fill_strategy == "mean":
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        else:
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(0)
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep='first'):
        """Remove duplicate rows."""
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self
    
    def save(self, output_csv: str):
        """Save cleaned dataset."""
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        self.df.to_csv(output_csv, index=False)
        print(f"✓ Cleaned dataset saved to: {output_csv}")
        return output_csv
    
    def get_df(self) -> pd.DataFrame:
        """Get cleaned dataframe."""
        return self.df


# ============================================================================
# 4. DATA AGGREGATION
# ============================================================================

class DataAggregator:
    """Aggregate data by various dimensions."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.aggregations = {}
    
    def by_column(self, column_name: str, n_top: Optional[int] = None) -> pd.Series:
        """Count values by column."""
        result = self.df[column_name].value_counts()
        if n_top:
            result = result.head(n_top)
        self.aggregations[column_name] = result
        return result
    
    def aggregate_all(self) -> Dict[str, pd.Series]:
        """Aggregate by all object columns."""
        agg_dict = {}
        for col in self.df.select_dtypes(include=["object"]).columns:
            agg_dict[col] = self.df[col].value_counts()
        self.aggregations.update(agg_dict)
        return agg_dict
    
    def print_aggregations(self):
        """Print all aggregations."""
        for col_name, counts in self.aggregations.items():
            print(f"\n--- AGGREGATION BY {col_name.upper()} ---")
            print(counts)


# ============================================================================
# 5. DATA SAMPLING
# ============================================================================

class DataSampler:
    """Sample data from dataset."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def sample_fraction(self, fraction: float = 0.3, random_state: int = 42) -> pd.DataFrame:
        """Sample a fraction of data."""
        return self.df.sample(frac=fraction, random_state=random_state)
    
    def sample_n(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """Sample n rows."""
        return self.df.sample(n=min(n, len(self.df)), random_state=random_state)
    
    def stratified_sample(self, column: str, frac: float = 0.3, random_state: int = 42) -> pd.DataFrame:
        """Sample stratified by column."""
        return self.df.groupby(column, group_keys=False).apply(
            lambda x: x.sample(frac=min(frac, 1.0), random_state=random_state)
        )


# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================

class OutlierDetector:
    """Detect outliers using IQR and Z-score methods."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    def detect_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
    
    def detect_zscore(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        return self.df[z_scores > threshold]
    
    def get_outliers_all_numeric(self, method: str = "iqr") -> Dict[str, pd.DataFrame]:
        """Get outliers for all numeric columns."""
        outliers_dict = {}
        for col in self.numeric_cols:
            if method == "iqr":
                outliers_dict[col] = self.detect_iqr(col)
            else:
                outliers_dict[col] = self.detect_zscore(col)
        return outliers_dict
    
    def remove_outliers(self, column: str, method: str = "iqr") -> pd.DataFrame:
        """Return dataframe with outliers removed."""
        if method == "iqr":
            outlier_indices = self.detect_iqr(column).index
        else:
            outlier_indices = self.detect_zscore(column).index
        return self.df.drop(outlier_indices)


# ============================================================================
# 7. CLASS IMBALANCE HANDLING
# ============================================================================

class ClassImbalanceHandler:
    """Handle imbalanced classes using SMOTE/ADASYN."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
    
    def apply_smote(self, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to balance classes."""
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(self.X, self.y)
        return pd.DataFrame(X_resampled, columns=self.X.columns), pd.Series(y_resampled)
    
    def apply_adasyn(self, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply ADASYN to balance classes."""
        adasyn = ADASYN(random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(self.X, self.y)
        return pd.DataFrame(X_resampled, columns=self.X.columns), pd.Series(y_resampled)
    
    def get_class_distribution(self) -> Dict[Any, int]:
        """Get class distribution."""
        return self.y.value_counts().to_dict()
    
    def check_imbalance(self, threshold: float = 0.1) -> bool:
        """Check if classes are imbalanced."""
        dist = self.get_class_distribution()
        min_class = min(dist.values())
        max_class = max(dist.values())
        ratio = min_class / max_class
        return ratio < threshold


# ============================================================================
# 8. FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Extract and create features from raw data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def _extract_action(self, s):
        """Extract action from dotted notation."""
        if not isinstance(s, str):
            return np.nan
        parts = s.split(".")
        return parts[-1] if parts else s
    
    def _service_short(self, s):
        """Get service name short form."""
        if not isinstance(s, str):
            return np.nan
        return s.split(".")[0]
    
    def _resource_type_short(self, s):
        """Get resource type short form."""
        if not isinstance(s, str):
            return np.nan
        return s.split(".")[-1]
    
    def _ip_is_private(self, x):
        """Check if IP is private."""
        try:
            ip = ipaddress.ip_address(str(x))
            return int(ip.is_private)
        except Exception:
            return np.nan
    
    def _ip_first_octet(self, x):
        """Extract first octet of IP."""
        try:
            return int(str(x).split(".")[0])
        except Exception:
            return np.nan
    
    def _datetime_parts(self, col):
        """Extract datetime components."""
        out = pd.DataFrame(index=self.df.index)
        out[f"{col}__year"] = self.df[col].dt.year
        out[f"{col}__month"] = self.df[col].dt.month
        out[f"{col}__day"] = self.df[col].dt.day
        out[f"{col}__hour"] = self.df[col].dt.hour
        out[f"{col}__dow"] = self.df[col].dt.dayofweek
        out[f"{col}__is_weekend"] = (out[f"{col}__dow"].isin([5, 6])).astype(int)
        return out
    
    def extract_datetime_features(self):
        """Extract features from datetime columns."""
        datetime_cols = [c for c in self.df.columns if re.search(r"(time|date)", c, re.IGNORECASE)]
        for c in datetime_cols:
            try:
                self.df[c] = pd.to_datetime(self.df[c], errors="coerce", utc=True)
            except Exception:
                pass
        
        dt_frames = []
        for c in datetime_cols:
            if c in self.df and pd.api.types.is_datetime64_any_dtype(self.df[c]):
                dt_frames.append(self._datetime_parts(c))
        if dt_frames:
            self.df = pd.concat([self.df] + dt_frames, axis=1)
        return self
    
    def extract_method_features(self, method_col: str = "protoPayload.methodName"):
        """Extract method action features."""
        if method_col in self.df.columns:
            self.df["method_action"] = self.df[method_col].apply(self._extract_action)
        return self
    
    def extract_service_features(self, service_col: str = "protoPayload.serviceName"):
        """Extract service short name."""
        if service_col in self.df.columns:
            self.df["service_short"] = self.df[service_col].apply(self._service_short)
        return self
    
    def extract_resource_features(self, resource_col: str = "resource.type"):
        """Extract resource type short name."""
        if resource_col in self.df.columns:
            self.df["resource_type_short"] = self.df[resource_col].apply(self._resource_type_short)
        return self
    
    def extract_ip_features(self, ip_col: str = "protoPayload.requestMetadata.callerIp"):
        """Extract IP-based features."""
        if ip_col in self.df.columns:
            self.df["callerIp_is_private"] = self.df[ip_col].apply(self._ip_is_private)
            self.df["callerIp_first_octet"] = self.df[ip_col].apply(self._ip_first_octet)
        return self
    
    def get_df(self) -> pd.DataFrame:
        """Get feature-engineered dataframe."""
        return self.df


# ============================================================================
# 9. TRANSFORMATION & NORMALIZATION
# ============================================================================

class DataTransformer:
    """Transform and normalize numeric features."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.transformations = {}
    
    def log_transform(self):
        """Apply log1p transformation to positive features."""
        transformed = {}
        for col in self.numeric_cols:
            values = self.df[col].astype(float).values
            finite = np.isfinite(values)
            values = np.where(finite, values, np.nan)
            
            if np.nanmin(values) >= 0 and np.nanmax(values) > 1.0:
                transformed[f"{col}__log1p"] = np.log1p(np.nan_to_num(values, nan=0.0))
        
        self.transformations.update(transformed)
        return self
    
    def zscore_normalize(self):
        """Apply Z-score normalization."""
        normalized = {}
        for col in self.numeric_cols:
            values = self.df[col].astype(float).values
            mu = np.nanmean(values)
            sigma = np.nanstd(values)
            if sigma == 0:
                sigma = 1.0
            normalized[f"{col}__z"] = (np.nan_to_num(values, nan=mu) - mu) / sigma
        
        self.transformations.update(normalized)
        return self
    
    def minmax_scale(self, feature_range=(0, 1)):
        """Apply Min-Max scaling."""
        scaled = {}
        for col in self.numeric_cols:
            values = self.df[col].astype(float).values
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            if vmax - vmin == 0:
                scaled[f"{col}__minmax"] = np.zeros_like(values)
            else:
                scaled[f"{col}__minmax"] = (values - vmin) / (vmax - vmin) * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        self.transformations.update(scaled)
        return self
    
    def get_transformations_df(self) -> pd.DataFrame:
        """Get dataframe with transformations."""
        return pd.DataFrame(self.transformations, index=self.df.index)


# ============================================================================
# 10. DISCRETIZATION & BINARIZATION
# ============================================================================

class DiscretizationBinarizer:
    """Discretize and binarize numeric features."""
    
    def __init__(self, df: pd.DataFrame, n_bins: int = 4):
        self.df = df
        self.n_bins = n_bins
        self.numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    
    def _make_kbins(self):
        """Create KBinsDiscretizer with version compatibility."""
        kwargs = {"n_bins": self.n_bins, "encode": "ordinal", "strategy": "quantile"}
        sig = inspect.signature(KBinsDiscretizer)
        if "quantile_method" in sig.parameters:
            kwargs["quantile_method"] = "averaged_inverted_cdf"
        return KBinsDiscretizer(**kwargs)
    
    def discretize(self) -> pd.DataFrame:
        """Discretize numeric features into bins."""
        warnings.filterwarnings("ignore", message="Bins whose width are too small")
        disc_dict = {}
        
        if not self.numeric_cols:
            return pd.DataFrame()
        
        num = self.df[self.numeric_cols].copy()
        num = num.fillna(num.median(numeric_only=True))
        
        for col in num.columns:
            series = num[col].astype(float)
            uniq_count = series.dropna().nunique()
            
            if uniq_count <= 1:
                disc_dict[f"{col}__qbin"] = np.zeros(len(series))
                continue
            
            bins_here = min(self.n_bins, uniq_count)
            
            try:
                kb = self._make_kbins()
                result = kb.fit_transform(series.to_frame()).ravel()
                if len(np.unique(result)) < 2:
                    disc_dict[f"{col}__qbin"] = pd.qcut(
                        series.rank(method="first"),
                        q=min(uniq_count, self.n_bins),
                        labels=False,
                        duplicates="drop"
                    )
                else:
                    disc_dict[f"{col}__qbin"] = result
            except Exception:
                disc_dict[f"{col}__qbin"] = pd.qcut(
                    series.rank(method="first"),
                    q=min(uniq_count, self.n_bins),
                    labels=False,
                    duplicates="drop"
                )
        
        return pd.DataFrame(disc_dict)
    
    def binarize(self) -> pd.DataFrame:
        """Binarize numeric features using median threshold."""
        bin_dict = {}
        
        if not self.numeric_cols:
            return pd.DataFrame()
        
        num = self.df[self.numeric_cols].copy()
        num = num.fillna(num.median(numeric_only=True))
        med = num.median(numeric_only=True)
        
        for col in num.columns:
            m = med.get(col, 0.0)
            bin_dict[f"{col}__bin"] = (num[col] > m).astype(int)
        
        return pd.DataFrame(bin_dict)


# ============================================================================
# 11. ENCODING & SCALING WITH PCA
# ============================================================================

class EncodeScalePCA:
    """Encode categorical, scale numeric, apply variance filtering and PCA."""
    
    def __init__(self, df: pd.DataFrame, top_k: int = 20, n_components: int = 10):
        self.df = df
        self.top_k = top_k
        self.n_components = n_components
    
    def _top_k_categorize(self, series, k=None):
        """Keep top-k categories, group rest as 'OTHER'."""
        if k is None:
            k = self.top_k
        vc = series.value_counts(dropna=True)
        keep = set(vc.head(k).index.tolist())
        
        def map_val(v):
            if pd.isna(v):
                return "NA"
            return v if v in keep else "OTHER"
        
        return series.map(map_val)
    
    def _make_ohe(self):
        """Create OneHotEncoder with version compatibility."""
        params = {"handle_unknown": "ignore"}
        sig = inspect.signature(OneHotEncoder)
        if "sparse_output" in sig.parameters:
            params["sparse_output"] = False
        else:
            params["sparse"] = False
        return OneHotEncoder(**params)
    
    def encode_scale(self) -> pd.DataFrame:
        """Encode categorical and scale numeric features."""
        df = self.df.copy()
        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        
        # Categorize
        for col in cat_cols:
            df[col] = self._top_k_categorize(df[col].astype(str))
        
        # Transform
        ohe = self._make_ohe()
        scaler = StandardScaler(with_mean=True, with_std=True)
        
        pre = ColumnTransformer(
            [("num", scaler, numeric_cols), ("cat", ohe, cat_cols)],
            remainder="drop"
        )
        
        X_pre = pre.fit_transform(df)
        num_names = numeric_cols
        if len(cat_cols) > 0:
            cat_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
        else:
            cat_names = []
        pre_names = num_names + cat_names
        
        return pd.DataFrame(X_pre, columns=pre_names)
    
    def apply_variance_threshold(self, threshold=1e-4) -> pd.DataFrame:
        """Remove low-variance features."""
        X_pre = self.encode_scale()
        var = VarianceThreshold(threshold=threshold)
        X = var.fit_transform(X_pre)
        support_mask = var.get_support()
        enc_names = [name for name, keep in zip(X_pre.columns, support_mask) if keep]
        
        return pd.DataFrame(X, columns=enc_names)
    
    def apply_pca(self, threshold=1e-4) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        X = self.apply_variance_threshold(threshold)
        
        # Fill NaN values before PCA
        X = X.fillna(X.mean())
        
        k = min(self.n_components, X.shape[1]) if X.shape[1] > 0 else 0
        
        if k > 0:
            pca = PCA(n_components=k, random_state=42)
            comps = pca.fit_transform(X)
            return pd.DataFrame(comps, columns=[f"PCA_{i+1}" for i in range(k)])
        else:
            return pd.DataFrame()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")