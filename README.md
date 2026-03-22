# 📊 TUBIMI I TË DHËNAVE PËR MACHINE LEARNING - SHPJEGIME ME KOD

**Dokumentacioni me shembuj praktik të kodit për secilin hap**

## FAZA 1: Kualiteti i të dhënave

### Kodi:
```python
# preprocessing_modules.py - DataQualityChecker class

def check_missing_values(self) -> pd.Series:
    """Numëron vlerat bosh në secilin kolonë"""
    return self.df.isnull().sum()
    # Output: 
    # column1: 100
    # column2: 50
    # column3: 0
```

- `self.df.isnull()` - gjen të gjithë vlerat bosh (NaN, None)
- `.sum()` - numëron sa vlera bosh ka në secilin kolonë

```python
def check_duplicates(self) -> int:
    """Numëron rreshtat e dyfishtë"""
    return self.df.duplicated().sum()
    # Output: 5 (ka 5 rreshta të dyfishtë)
```

- `self.df.duplicated()` - gjen rreshtat që përsëriten
- `.sum()` - numëron sa rreshta të dyfishtë ka

```python
def check_empty_strings(self) -> Dict[str, int]:
    """Gjen stringjet e zbrazëta ("") në kolonat e tekstit"""
    empty_strings = {}
    for col in self.df.columns:
        if self.df[col].dtype == object:  # kolonë teksti
            # .str.strip() heq hapësirat e tepërt
            # .eq("") kontrollon nëse është bosh
            empty_strings[col] = self.df[col].astype(str).str.strip().eq("").sum()
        else:
            empty_strings[col] = 0
    return empty_strings
```
- Kontrollon kolonat e tekstit
- Heq hapësirat e tepërt (spaces)
- Numëron sa stringje janë të zbrazëta

### Rezultati (output):
```
Total Rows: 10000
Total Columns: 50
Complete Records (pa vlera bosh): 92
Records with Nulls (me vlera bosh): 9908
Duplicate Rows: 0
```

---

## FAZA 2: Tipet e të dhënave

Përcaktojmë se çfarë tipi është secilin kolonë (numër, tekst, data, etj).

### Kodi:
```python
# preprocessing_modules.py - TypeDetector class

def detect(self) -> Dict[str, str]:
    """Zbulon tipin e të dhënave për secilin kolonë"""
    df = pd.read_csv(self.csv_path, low_memory=False)
    
    # Përpiqemi të konvertojmë në datetime
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass  # nëse nuk mund të konvertohet, le të qendrojë si është
    
    detected = {}
    for col, dtype in df.dtypes.items():
        # Kontrollon secilin tip
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
```

- `pd.to_datetime()` - konvertimi në datë
- `is_integer_dtype()` - kontrollon nëse është numër i plotë
- `is_float_dtype()` - kontrollon nëse është numër me presje
- `is_datetime64_any_dtype()` - kontrollon nëse është datë

### Rezultati:
```python
{
    'age': 'int',
    'salary': 'float',
    'name': 'string',
    'active': 'bool',
    'timestamp': 'datetime'
}
```
<img width="375" height="382" alt="Screenshot 2026-03-22 at 11 23 02" src="https://github.com/user-attachments/assets/2c49d13b-afea-4eab-9941-635914c09058" />

---

## FAZA 3: PASTRIMI I TË DHËNAVE

### 3.1 HEQJA E RRESHTAVE DHE KOLONAVE BOSH

```python
def remove_empty_rows_cols(self):
    self.df.dropna(how="all", inplace=True)
    self.df.dropna(axis=1, how="all", inplace=True)
    return self
```

### 3.2 TRAJTIM I VLERAVE BOSH - STRATEGJIA KRYESORE

```python
def clean_numeric_columns(self, fill_strategy="median"):
    numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
    
    if fill_strategy == "median":
        for col in numeric_cols:
            median_value = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_value)
    
    elif fill_strategy == "mean":
        for col in numeric_cols:
            mean_value = self.df[col].mean()
            self.df[col] = self.df[col].fillna(mean_value)
    
    else:  # "zero"
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(0)
    return self
```

### 3.3 TRAJTIM I STRINGJEVE (TEKST)

```python

def clean_string_columns(self):
    for col in self.df.select_dtypes(include=["object"]).columns:
        self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        self.df[col] = self.df[col].replace({
            "nan": "unknown",      # stringi "nan"
            "none": "unknown",     # stringi "none"
            "": "unknown"          # string bosh
        })
    return self
```
### 3.4 TRAJTIM I BOOLEAN

```python
def clean_boolean_columns(self, fill_value=False):
    """Plotëson vlerat bosh në kolonat boolean"""
    for col in self.df.select_dtypes(include=["bool"]).columns:
        self.df[col] = self.df[col].fillna(fill_value)
    return self
```

### 3.5 HEQJA E RRESHTAVE TË DYFISHTË

```python
def remove_duplicates(self, subset: Optional[List[str]] = None, keep='first'):
    self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
    return self
```
---

<img width="1082" height="354" alt="Screenshot 2026-03-22 at 11 24 12" src="https://github.com/user-attachments/assets/40c8ac7c-3951-489f-9750-63ae5679c9ba" />


## FAZA 4: AGREGIMI

### Kodi:
```python

def by_column(self, column_name: str, n_top: Optional[int] = None) -> pd.Series:
    result = self.df[column_name].value_counts()
    if n_top:
        result = result.head(n_top)
    self.aggregations[column_name] = result
    return result
```
## FAZA 5: MOSTRIMI

### Kodi:
```python
def sample_fraction(self, fraction: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    return self.df.sample(frac=fraction, random_state=random_state)
```

```python
def sample_n(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    return self.df.sample(n=min(n, len(self.df)), random_state=random_state)
```

```python
def stratified_sample(self, column: str, frac: float = 0.3) -> pd.DataFrame:
    return self.df.groupby(column, group_keys=False).apply(
        lambda x: x.sample(frac=min(frac, 1.0), random_state=42)
    )
```

## FAZA 6: OUTLIERS

### Kodi - METODA 1: IQR

```python

def detect_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    Q1 = self.df[column].quantile(0.25)   
    Q3 = self.df[column].quantile(0.75)    
    IQR = Q3 - Q1                           
    
    lower_bound = Q1 - multiplier * IQR     
    upper_bound = Q3 + multiplier * IQR    
    
    return self.df[(self.df[column] < lower_bound) | 
                   (self.df[column] > upper_bound)]
```

### Kodi - METODA 2: Z-SCORE

```python

def detect_zscore(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
    
    z_scores = np.abs((self.df[column] - self.df[column].mean()) / 
                      self.df[column].std())
    
    return self.df[z_scores > threshold]
```

<img width="543" height="382" alt="Screenshot 2026-03-22 at 11 24 52" src="https://github.com/user-attachments/assets/5371344f-0d8b-43b5-b4c2-5f418c5e6077" />
<img width="538" height="380" alt="Screenshot 2026-03-22 at 11 25 00" src="https://github.com/user-attachments/assets/73a6ab8b-dbd2-4e5e-a0e8-8c89444f4a1b" />


## FAZA 7: BALANCIMI I KLASAVE

### Kodi - KONTROLLI I IMBALANCIMIT

```python

def get_class_distribution(self) -> Dict[Any, int]:
    return self.y.value_counts().to_dict()

def check_imbalance(self, threshold: float = 0.1) -> bool:
    dist = self.get_class_distribution()
    min_class = min(dist.values()) 
    max_class = max(dist.values())   
    ratio = min_class / max_class   
    
    return ratio < threshold
```
### Kodi - SMOTE

```python

def apply_smote(self, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(self.X, self.y)
    return pd.DataFrame(X_resampled, columns=self.X.columns), pd.Series(y_resampled)
```

### Kodi - ADASYN

```python
def apply_adasyn(self, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(self.X, self.y)
    return pd.DataFrame(X_resampled, columns=self.X.columns), pd.Series(y_resampled)
```

## FAZA 8: FEATURE ENGINEERING

### Kodi - VEÇORI DATETIME

```python

def _datetime_parts(self, col):
    out = pd.DataFrame(index=self.df.index)
    
    out[f"{col}__year"]   = self.df[col].dt.year      # 2024
    out[f"{col}__month"]  = self.df[col].dt.month     # 3
    out[f"{col}__day"]    = self.df[col].dt.day       # 21
    out[f"{col}__hour"]   = self.df[col].dt.hour      # 14
    out[f"{col}__dow"]    = self.df[col].dt.dayofweek # 3 (e mërkurë)
    
    out[f"{col}__is_weekend"] = (out[f"{col}__dow"].isin([5, 6])).astype(int)
    
    return out

def extract_datetime_features(self):
    datetime_cols = [c for c in self.df.columns 
                     if re.search(r"(time|date)", c, re.IGNORECASE)]
    
    dt_frames = []
    for c in datetime_cols:
        if c in self.df and pd.api.types.is_datetime64_any_dtype(self.df[c]):
            dt_frames.append(self._datetime_parts(c))
    
    if dt_frames:
        self.df = pd.concat([self.df] + dt_frames, axis=1)
    
    return self
```

### Kodi - VEÇORI IP

```python
def _ip_is_private(self, x):
    try:
        ip = ipaddress.ip_address(str(x))
        return int(ip.is_private)  # 1 = private, 0 = public
    except Exception:
        return np.nan

def _ip_first_octet(self, x):
    try:
        return int(str(x).split(".")[0])  # "192.168.1.1" → 192
    except Exception:
        return np.nan

def extract_ip_features(self, ip_col: str = "protoPayload.requestMetadata.callerIp"):
    if ip_col in self.df.columns:
        self.df["callerIp_is_private"] = self.df[ip_col].apply(self._ip_is_private)
        self.df["callerIp_first_octet"] = self.df[ip_col].apply(self._ip_first_octet)
    
    return self
```

### Kodi - VEÇORI METODE

```python
def _extract_action(self, s):
    if not isinstance(s, str):
        return np.nan
    parts = s.split(".")    # ndaj me "."
    return parts[-1] if parts else s  # kthe pjesën e fundit

def extract_method_features(self, method_col: str = "protoPayload.methodName"):
    if method_col in self.df.columns:
        self.df["method_action"] = self.df[method_col].apply(self._extract_action)
    
    return self
```

## FAZA 9: TRANSFORMIM

```python
# preprocessing_modules.py - DataTransformer class

def log_transform(self):
    transformed = {}
    
    for col in self.numeric_cols:
        values = self.df[col].astype(float).values
        finite = np.isfinite(values)  # kontrollon nëse janë numra të vlefshëm
        values = np.where(finite, values, np.nan)
        
        if np.nanmin(values) >= 0 and np.nanmax(values) > 1.0:
            transformed[f"{col}__log1p"] = np.log1p(np.nan_to_num(values, nan=0.0))
            # np.log1p(x) = log(1 + x) - më stabil për vlera të vogla
    
    self.transformations.update(transformed)
    return self
```

### Kodi - Z-SCORE NORMALIZATION

```python
def zscore_normalize(self):
    normalized = {}
    
    for col in self.numeric_cols:
        values = self.df[col].astype(float).values
        
        mu = np.nanmean(values)       # mesatarja
        sigma = np.nanstd(values)      # deviacioni standard
        
        if sigma == 0:
            sigma = 1.0  # shmang ndarjen me 0
        
        normalized[f"{col}__z"] = (np.nan_to_num(values, nan=mu) - mu) / sigma
    
    self.transformations.update(normalized)
    return self
```

### Kodi - MIN-MAX SCALING

```python
def minmax_scale(self, feature_range=(0, 1)):
    scaled = {}
    
    for col in self.numeric_cols:
        values = self.df[col].astype(float).values
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        
        if vmax - vmin == 0:
            scaled[f"{col}__minmax"] = np.zeros_like(values)
        else:
            scaled[f"{col}__minmax"] = (values - vmin) / (vmax - vmin) * \
                                       (feature_range[1] - feature_range[0]) + feature_range[0]
    
    self.transformations.update(scaled)
    return self
```

## FAZA 10: DISKRETIZIM

### Kodi - QUANTILE BINNING

```python

def discretize(self) -> pd.DataFrame:
    disc_dict = {}
    
    num = self.df[self.numeric_cols].copy()
    num = num.fillna(num.median(numeric_only=True))
    
    for col in num.columns:
        series = num[col].astype(float)
        uniq_count = series.dropna().nunique()  # vlera unike
        
        if uniq_count <= 1:
            disc_dict[f"{col}__qbin"] = np.zeros(len(series))
            continue
        
        bins_here = min(self.n_bins, uniq_count)  # numri i bins (4 si default)
        
        try:
            disc_dict[f"{col}__qbin"] = pd.qcut(
                series.rank(method="first"),
                q=bins_here,  # 4 bina
                labels=False,
                duplicates="drop"
            )
        except Exception:
            disc_dict[f"{col}__qbin"] = pd.qcut(
                series.rank(method="first"),
                q=min(uniq_count, self.n_bins),
                labels=False,
                duplicates="drop"
            )
    
    return pd.DataFrame(disc_dict)
```

### Kodi - BINARIZATION

```python
def binarize(self) -> pd.DataFrame:
    bin_dict = {}
    
    num = self.df[self.numeric_cols].copy()
    num = num.fillna(num.median(numeric_only=True))
    med = num.median(numeric_only=True)
    
    for col in num.columns:
        m = med.get(col, 0.0)
        # 1 nëse vlera > median, 0 ndryshe
        bin_dict[f"{col}__bin"] = (num[col] > m).astype(int)
    
    return pd.DataFrame(bin_dict)
```


## FAZA 11: KODIM DHE PCA

### Kodi - ONE-HOT ENCODING

```python

def _top_k_categorize(self, series, k=None):
    if k is None:
        k = self.top_k  # 20 kategori si default
    
    vc = series.value_counts(dropna=True)
    keep = set(vc.head(k).index.tolist())  # top 20 kategori
    
    def map_val(v):
        if pd.isna(v):
            return "NA"
        return v if v in keep else "OTHER"
    
    return series.map(map_val)

def encode_scale(self) -> pd.DataFrame:
    df = self.df.copy()
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    
    for col in cat_cols:
        df[col] = self._top_k_categorize(df[col].astype(str))
    
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    pre = ColumnTransformer(
        [("num", scaler, numeric_cols),
         ("cat", ohe, cat_cols)],
        remainder="drop"
    )
    
    # Aplikoj transformerët
    X_pre = pre.fit_transform(df)
    
    return pd.DataFrame(X_pre)
```

### Kodi - PCA (PRINCIPAL COMPONENT ANALYSIS)

```python
def apply_pca(self, threshold=1e-4) -> pd.DataFrame:
    """PCA për reduktim dimensionaliteti"""
    
    # Fillimisht aplikoj variance threshold
    X = self.apply_variance_threshold(threshold)
    
    # Plotësoj vlerat bosh para PCA
    X = X.fillna(X.mean())
    
    # Numri i komponentave
    k = min(self.n_components, X.shape[1]) if X.shape[1] > 0 else 0
    
    if k > 0:
        # Inicjalizoj PCA me k komponentë
        pca = PCA(n_components=k, random_state=42)
        # Aplikoj PCA
        comps = pca.fit_transform(X)
        
        # Krijo DataFrame me emrat e komponntave
        return pd.DataFrame(comps, columns=[f"PCA_{i+1}" for i in range(k)])
    else:
        return pd.DataFrame()
```

## FAZA 12: PËRGATITJE PËR ML

### Kodi - TRAIN/TEST SPLIT

```python

from sklearn.model_selection import train_test_split

def stage_ml_preparation(self, final_csv: str, 
                        target_column: Optional[str] = None,
                        test_size: float = 0.2) -> Dict[str, str]:
    
    df = pd.read_csv(final_csv, low_memory=False)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2,          # 20% për test
        random_state=42         # reproducible
    )
    
    train_path = os.path.join(self.output_dir, f"stage_12_train_{self.timestamp}.csv")
    test_path = os.path.join(self.output_dir, f"stage_12_test_{self.timestamp}.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

