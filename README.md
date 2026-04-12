# Analizimi i Audit Logs (Regjistra të Aktiviteteve) nga Kubernetes Service në Google Cloud Platform

**Universiteti:** Universiteti i Prishtines

**Fakulteti:** Fakulteti i Inxhinierise Elektrike dhe Kompjuterike

**Niveli i studimeve:** Master 

**Lënda:** Machine Learning  

**Mësimdhënësit:** Lule Ahmedi, Mërgim Hoti 

**Studentët që kanë kontribuar:**  
Art Ukshini, Leotrim Halimi, Vera Llugiqi

---

## Faza I: Përgatitja e të dhënave
# Përgatitja e të Dhënave për Machine Learning – Faza I

## 1. Përshkrimi i projektit
Ky projekt paraqet fazën e parë të punës: parapërpunimin e dataset-it me qëllim që të dhënat të bëhen të pastra, të strukturuara dhe të gatshme për fazën e dytë, ku do të zhvillohet modeli i machine learning.

Pra, në këtë fazë nuk ndërtohet ende modeli, por përgatitet dataset-i në mënyrë që në vazhdim:
- të zgjidhet target-i,
- të bëhet train/test split,
- të trajnohen modelet,
- dhe të vlerësohen rezultatet.

Qëllimi kryesor i Fazës I është:
- të kuptohen tipet e të dhënave,
- të vlerësohet cilësia e dataset-it,
- të trajtohen vlerat që mungojnë,
- të pastrohen dhe transformohen kolonat,
- të hiqen kolonat jo të dobishme,
- dhe të krijohet një dataset final i përshtatshëm për machine learning.

---

## 2. Dataset-i
Dataset-i përmban **audit logs** / ngjarje të regjistruara nga sistemet cloud / Kubernetes, me informacione si:
- `timestamp` (koha e aktivitetit / eventit),
- `principalEmail` (email-i i përdoruesit),
- `serviceName` (emri i servisit ku është kryer aktiviteti, në këtë rast: Kubernetes - k8s.io),
- `methodName` (aktiviteti apo aksioni i kryer),
- `resourceName` (emri i resursit të afektuar),
- `project_id` (ID e projektit),
- `callerSuppliedUserAgent` (shembull: shfletuesi ~ browser i përdorur ose CLI, përmes të cilit është kryer aktiviteti),
- `callerIp` (IP e përdoruesit),
- `permission` (privilegji i përdorur gjatë aktivitetit),
- `authorization.k8s.io/decision` (vendimi i autorizimit apo mosautorizimit),
- `authorization.k8s.io/reason` (arsyeja),
- `status.code/message` (kodi dhe mesazhi i statusit, nëse është kryer aktiviteti me sukses ose ka dështuar për ndonjë arsye),
- `labels` dhe `metadata` të tjera.

Këto të dhëna janë heterogjene dhe përmbajnë:
- kolona kategorike,
- kolona numerike,
- kolona të tipit datetime,
- kolona me shumë vlera që mungojnë,
- dhe kolona me vlera shumë unike (p.sh. ID).

Për këtë arsye, parapërpunimi është i domosdoshëm para kalimit në machine learning.

Numri i objekteve: 10000
Numri i atributeve: 35

---

## 3. Hapat e parapërpunimit

### 3.1 Zbulimi i tipeve të të dhënave
Në fillim bëhet identifikimi i tipeve të kolonave:
- numerike,
- kategorike,
- datetime.

### 🔹 Tipet e të dhënave

Ky hap është i rëndësishëm sepse përcakton mënyrën se si do të trajtohet secila kolonë në vazhdim.

Kolonat që kanë emra si:
- `time`
- `date`
- `timestamp`

trajtohen si kandidatë për datetime.

Kodi:
```python
cleaner = DataCleaner(df)
cleaner.fix_datetime_columns()
```

---

### 3.2 Analiza e cilësisë së të dhënave
Para transformimeve kontrollohen:
- numri i rreshtave dhe kolonave,
- vlerat null / missing,
- kolonat bosh,
- rreshtat duplikatë,
- shpërndarja bazike e kolonave.

Ky hap ndihmon për të kuptuar sa i pastër është dataset-i dhe çfarë problemesh duhet të rregullohen.

Kodi:
```python
df.shape
df.isna().mean().sort_values(ascending=False).head(10)
df.duplicated().sum()
```

---

### 3.3 Pastrimi i të dhënave
Gjatë pastrimit bëhen këto veprime:
- hiqen rreshtat duplikatë,
- hiqen kolonat plotësisht bosh,
- hiqen rreshtat plotësisht bosh,
- pastrohen stringjet nga hapësirat e panevojshme,
- stringjet bosh kthehen në vlera missing,
- standardizohet forma e disa vlerave tekstuale.

Ky hap e bën dataset-in më të qëndrueshëm dhe më të lehtë për përpunim të mëtejshëm.

Kodi:
```python
cleaner.remove_empty_rows_cols()
cleaner.normalize_string_columns()
cleaner.clean_boolean_columns()
cleaner.remove_duplicates()
cleaner.feature_selection()
df = cleaner.df
```

---

### 3.4 Trajtimi i vlerave që mungojnë
Trajtimi i missing values është një nga pjesët më të rëndësishme të projektit.

Strategjia e përdorur është:

#### Heqja e kolonave shumë të zbrazëta
Kolonat që kanë më shumë se **70% vlera që mungojnë** hiqen, sepse:
- nuk japin informacion të mjaftueshëm,
- shtojnë zhurmë,
- dhe nuk ndihmojnë në machine learning.

#### Imputimi i vlerave të mbetura
Për kolonat e mbetura përdoren këto rregulla:

- **kolonat numerike** → plotësohen me **medianën**
- **kolonat datetime** → plotësohen me vlerë qendrore të përshtatshme
- **kolonat booleane** → plotësohen me **mode** ose `False`
- **kolonat kategorike / tekstuale** → plotësohen me **mode** ose me `"missing"`

Kjo strategji ruan sa më shumë të dhëna dhe shmang humbjen e tepërt të rreshtave.

Kodi:
```python
df, dropped_missing = drop_columns_over_missing_fraction(df, missing_threshold=0.7)
df = impute_values(df)
```

### 3.5 Zbulimi dhe largimi i outlier-ëve (IQR)
Për kolonat numerike përdoret rregulli IQR për të detektuar rreshtat outlier dhe për t'i larguar nga dataset-i.

Kodi:
```python
df, outlier_report = remove_outliers_iqr(df, factor=1.5)
print("Outlier report:", outlier_report)
```

---

### 3.6 Inxhinieria e veçorive (Feature Engineering)
Për ta bërë dataset-in më të dobishëm për machine learning, krijohen disa kolona të reja nga kolonat ekzistuese.

Nga kolonat datetime krijohen kolona numerike me sekonda nga epoka (`epoch seconds`).

Gjithashtu krijohet një kolonë si:
- **anonymous_principal**  
që tregon nëse mungon informacioni për përdoruesin / principal-in.

Këto veçori të reja ndihmojnë modelin në fazën e ardhshme të kapë modele më kuptimplota në të dhëna.

Kodi:
```python
df = add_anonymous_principal_flag(df)
df = add_datetime_epoch_seconds(df)
```

---

### 3.7 Heqja e kolonave jo të dobishme
Hiqen kolonat që nuk janë të përshtatshme për machine learning, si:

- kolona me shumë missing values,
- kolona me shumë pak variacion,
- kolona me vlera pothuajse unike për çdo rresht,
- ID ose fusha shumë specifike që nuk ndihmojnë modelin.

Kjo ul zhurmën dhe e bën dataset-in më të fokusuar.

Kodi:
```python
cleaner.feature_selection()
df, dropped_missing = drop_columns_over_missing_fraction(df, missing_threshold=0.7)
```

---

### 3.8 Label Encoding + mapping file
Kolonat kategorike të përshtatshme enkodohen në forma numerike (`__le`) për machine learning.
Ruhet edhe një file tekstual që tregon cilat numra korrespondojnë me cilat label-a.

Kodi:
```python
df, mappings = add_selective_label_encoding(df)
save_label_mappings_txt(mappings, "processedfiles/label_mappings.txt")
```

### 3.9 Analiza e target-it dhe imbalance
Edhe pse trajnimi i modelit nuk bëhet ende në këtë fazë, eshte kontrollohuar një target i mundshëm për Fazën II.

Një kandidat shumë i mirë për target është:
- `labels.authorization.k8s.io/decision`

pasi ka vlera si:
- `allow`
- `forbid`

Ky target është i përshtatshëm për një problem klasifikimi.

Në këtë fazë mund të bëhet vetëm:
- kontrolli i shpërndarjes së target-it,
- numërimi i klasave,
- evidentimi i imbalance nëse ekziston.

Ky hap ndihmon për të kuptuar dataset-in më mirë pa hyrë ende në modelim.

---

## 4. Sampling
Nuk u përfshi sepse dataset-i ka madhësi të menaxhueshme dhe nuk ka nevojë për reduktim artificial.



---

## 5. Çfarë u shtua

1. **Feature Selection**
   - heq kolonat konstante
   - heq kolonat me cardinality shumë të lartë

2. **Basic Target Inspection**
   - kontrollon shpërndarjen e target-it të mundshëm
   - ndihmon për të kuptuar klasat para fazës së machine learning

---

## 6. Rezultati final
Në fund gjenerohet një dataset i pastruar dhe i përgatitur:

```text
processedfiles/ml_ready.csv
processedfiles/label_mappings.txt
```

Ky dataset:
- është i pastruar,
- ka missing values të trajtuara,
- ka rreshta outlier të larguar me IQR,
- ka veçori të reja të dobishme,
- ka më pak zhurmë,
- dhe është i gatshëm për Fazën II të machine learning.

---

### Faza II: Trajnimi i modeleve për anomaly detection

Pas parapërpunimit, dataset-i `processedfiles/ml_ready.csv` është **numerik** dhe përfshin `timestamp_delay_s` (vonesa mes receive dhe timestamp, nga pipeline-i në `data.py`).

Në këtë fazë planifikohen **tre** modele për anomaly detection; 
1. **Isolation Forest**. 
2.
3.

#### Struktura e skedarëve (Faza 2)

```text
anomaly_models/
├── __init__.py
├── common.py                 # ngarkim ml_ready, validim — përbashkët për të gjitha modelet
└── isolation_forest.py     # trajnim + CLI vetëm për Isolation Forest
train_anomaly.py              # hyrje e shkurtër; thërret IF (kompatibilitet me komandat e vjetra)
models/
└── isolation_forest/         # artefaktet e IF pas --save
    ├── isolation_forest.joblib
    ├── standard_scaler.joblib
    └── feature_columns.joblib
```

### 7.1 Përgatitja e përbashkët

- Hyrja: `processedfiles/ml_ready.csv`
- Ndarja e të dhënave:
  ```python
  y = df["labels.authorization.k8s.io/decision__le"]
  X = df.drop(columns=["labels.authorization.k8s.io/decision__le"])
  ```
- Kontrolli: pa vlera që mungojnë në `X` dhe `y`; të gjitha kolonat e `X` numerike
- Scaling: para trajnitimit përdoret `StandardScaler` mbi `X` (implementimi është brenda secilit skedar modeli)

Kodi i përbashkët: `anomaly_models/common.py` (`load_ml_ready`, `validate_features`).

### 7.2 Trajnimi me Isolation Forest

Në këtë fazë u përdor algoritmi **Isolation Forest** për të realizuar anomaly detection mbi dataset-in e përpunuar.

Isolation Forest është një model **jo-supervised**, i cili nuk përdor etiketa gjatë trajnimit. Në vend të kësaj, ai identifikon raste që devijojnë nga shpërndarja normale e të dhënave. Logjika e tij bazohet në faktin që anomalitë zakonisht janë më të lehta për t’u izoluar se rastet normale, prandaj kërkojnë më pak ndarje (splits) në pemët vendimmarrëse.

Në këtë projekt, modeli u trajnuar vetëm mbi matricën e veçorive `X`, ndërsa kolona `decision` (`y`) u përdor vetëm për krahasim dhe interpretim të rezultateve.

---

#### Ndarja e features dhe target

```python
import pandas as pd

TARGET_COL = "labels.authorization.k8s.io/decision__le"

df = pd.read_csv("processedfiles/ml_ready.csv", low_memory=False)

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])
```

---

#### Trajnimi i modelit

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)
preds = model.predict(X_scaled)
```

Në rezultatet e modelit:

* `1` përfaqëson raste normale
* `-1` përfaqëson anomalitë

---

#### Vlerësimi fillestar i rezultateve

```python
import numpy as np
import pandas as pd

print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
print("Normal count (pred == 1):", int(np.sum(preds == 1)))

ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])
print(ct)
```

Ky krahasim përdoret për të analizuar nëse rastet e identifikuara si anomali përputhen më shpesh me klasën më të rrallë (`forbid`), edhe pse modeli nuk është trajnuar drejtpërdrejt për klasifikim.

---

#### Testimi i parametrave (`contamination`)

Për të zgjedhur konfigurimin më të përshtatshëm, u provuan disa vlera të parametrin `contamination`:

```bash
python -m anomaly_models.isolation_forest --sweep
```

U testuan vlerat:

* `0.01`
* `0.03`
* `0.05`
* `0.1`

Nga rezultatet u vërejt se:

* vlerat e ulëta janë më konservative dhe identifikojnë më pak anomalitë
* vlerat e larta janë më agresive dhe rrisin numrin e rasteve të klasifikuara si anomali
* vlera `0.05` ofron kompromisin më të mirë praktik

---

#### Konfigurimi final i modelit

Në bazë të eksperimenteve, modeli final u vendos:

* `IsolationForest`
* `contamination = 0.05`
* `n_estimators = 100`

Rezultati final:

* `anomaly count = 500`
* `normal count = 9500`

Crosstab:

```
pred   -1     1
y
0     171  9288
1     329   212
```

---

#### Ruajtja e modelit

```bash
python -m anomaly_models.isolation_forest --save
```

Ky hap ruan:

* modelin (`isolation_forest.joblib`)
* scaler-in (`standard_scaler.joblib`)
* listën e feature-ve (`feature_columns.joblib`)

---

Modeli Isolation Forest rezultoi i përshtatshëm për këtë dataset, duke identifikuar një numër të arsyeshëm anomalish dhe duke kapur një pjesë të konsiderueshme të rasteve të pazakonta.

Duhet theksuar se modeli nuk është trajnuar për klasifikim të drejtpërdrejtë të klasës `forbid`, por për identifikim të devijimeve nga sjellja normale në të dhëna.
