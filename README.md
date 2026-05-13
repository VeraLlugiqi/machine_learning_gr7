# Analizimi i Audit Logs (Regjistra të Aktiviteteve) nga Kubernetes Service në Google Cloud Platform

<table border="0" cellpadding="8" cellspacing="0">
 <tr>
  <td valign="top" style="width:180px;">
    <img src="assets/University_of_Prishtina_logo.png" alt="University of Prishtina Logo" width="150" />
  </td>
  <td valign="top">
    <p><strong>Universiteti i Prishtinës / University of Prishtina</strong></p>
    <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
    <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
    <p>Profesoret: Lule Ahmedi, Mërgim Hoti</p>
    <p>Studentët që kanë kontribuar: Art Ukshini, Leotrim Halimi, Vera Llugiqi</p>
  </td>
 </tr>
</table>
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

| Metrika | `dataset.csv` (origjinali) | `processedfiles/ml_ready.csv` |
| --- | ---: | ---: |
| Rreshta | 10,489 | 10,001 |
| Features / kolona | 50 | 14 |
| Rreshta me të paktën një missing value | 9,908 | 0 |
| Kolona numerike | 2 | 14 |
| Kolona tekstuale / kategorike | 46 | 0 |

> `ml_ready.csv` ruan të njëjtin numër rreshtash, por e kthen dataset-in në formë të gatshme për machine learning duke hequr missing values dhe duke e reduktuar në vetëm kolona numerike.

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

Në këtë fazë janë të implementuara **katër** modele për anomaly detection:
1. **Isolation Forest**
2. **Local Outlier Factor (LOF)**
3. **One-Class SVM**
4. **Elliptic Envelope**


#### Struktura e skedarëve (Faza 2)

```text
anomaly_models/
├── __init__.py
├── common.py                 # ngarkim ml_ready, validim — përbashkët për të gjitha modelet
├── isolation_forest.py       # trajnim + CLI për Isolation Forest
├── isolation_forest_lime.py  # LIME për Isolation Forest
├── isolation_forest_viz.py   # dashboard interaktiv për Isolation Forest
├── local_outlier_factor.py   # trajnim + CLI për Local Outlier Factor
├── one_class_svm.py          # trajnim + CLI për One-Class SVM
└── elliptic_envelope.py      # trajnim + CLI për Elliptic Envelope
train_anomaly.py              # hyrje e shkurtër; zgjedh metodën me --method
models/
├── isolation_forest/
├── local_outlier_factor/
├── one_class_svm/
└── elliptic_envelope/
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

Kodi i përbashkët: `anomaly_models/common.py` (`load_ml_ready`, `validate_features`, `export_predictions`).

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
1     389   152
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
* opsionalisht rezultatet me `--export-results`

---

Modeli Isolation Forest rezultoi i përshtatshëm për këtë dataset, duke identifikuar një numër të arsyeshëm anomalish dhe duke kapur një pjesë të konsiderueshme të rasteve të pazakonta.

Duhet theksuar se modeli nuk është trajnuar për klasifikim të drejtpërdrejtë të klasës `forbid`, por për identifikim të devijimeve nga sjellja normale në të dhëna.

### 7.3 Trajnimi me Local Outlier Factor

Si metodë e dytë u shtua **Local Outlier Factor (LOF)**, i cili mat sa i izoluar është një rresht krahasuar me dendësinë lokale të fqinjëve të tij më të afërt.

Ky model është i dobishëm kur anomalitë nuk dallohen vetëm globalisht, por edhe si sjellje që devijon nga grupi lokal ku bën pjesë pika.

Trajnimi mund të bëhet me:

```bash
python train_anomaly.py --method local_outlier_factor --save --export-results
```

ose direkt:

```bash
python -m anomaly_models.local_outlier_factor --save --export-results
```

Parametrat kryesorë:

* `contamination = 0.05`
* `n_neighbors = 20`

---

#### Trajnimi i modelit

```python
from sklearn.neighbors import LocalOutlierFactor

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=True,
)

model.fit(X_scaled)
preds = model.predict(X_scaled)
```

Në rezultatet e modelit:

* `1` përfaqëson raste normale
* `-1` përfaqëson anomalitë

---

#### Vlerësimi i rezultateve

```python
ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])
print(ct)
print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
print("Normal count (pred == 1):", int(np.sum(preds == 1)))
```

Rezultati final:

* `anomaly count = 417`
* `normal count = 9583`

Crosstab:

```text
pred   -1     1
y
0     316  9143
1      294   257
```

Nga këto rezultate vërehet se LOF është më konservativ dhe kap vetëm një pjesë të vogël të rasteve `forbid` si anomali.

Në terma të klasës `forbid` (`label=1`), rezultati është:

* `precision = 0.0983`
* `recall = 0.0758`
* `f1-score = 0.0856`

Kjo tregon se LOF funksionon, por në këtë dataset është më pak i përshtatshëm se Isolation Forest ose One-Class SVM.

---

#### Ruajtja e modelit

```bash
python -m anomaly_models.local_outlier_factor --save --export-results
```


### 7.4 Trajnimi me One-Class SVM

**One-Class SVM** është forma e SVM-it e përdorur për anomaly detection. Ky model mëson kufirin e rajonit normal dhe i etiketon si anomali pikat që dalin jashtë këtij kufiri.

Për këtë model është më e saktë të trajnohet vetëm mbi rastet normale (`allow = 0`), sepse One-Class SVM supozohet të mësojë vetëm shpërndarjen e sjelljes normale. Kur futen edhe anomalitë në trajnim, kufiri mund të shtrembërohet dhe modeli e ka më të vështirë të dallojë `forbid`.

Trajnimi mund të bëhet me:

```bash
python train_anomaly.py --method svm --save --export-results
```

ose direkt:

```bash
python -m anomaly_models.one_class_svm --save --export-results
```

Parametrat kryesorë:

* `nu = 0.05`
* `kernel = rbf`
* `gamma = scale`
* `train_on_normal_only = true`

---

#### Trajnimi i modelit

```python
from sklearn.svm import OneClassSVM

if train_on_normal_only:
    X_train = X[y == normal_label]
else:
    X_train = X

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)

model = OneClassSVM(
    nu=0.05,
    kernel="rbf",
    gamma="scale",
    cache_size=500,
)

model.fit(X_train_scaled)
preds = model.predict(X_scaled)
```

Ky konfigurim është i qëllimshëm sepse modeli trajnohet vetëm mbi rastet normale (`allow = 0`), që është mënyra më e përshtatshme për One-Class SVM në këtë projekt.

---

#### Vlerësimi i rezultateve

```python
ct = pd.crosstab(y, preds, rownames=["y"], colnames=["pred"])
print(ct)
print("Train rows used:", int((y == 0).sum()))
print("Anomaly count (pred == -1):", int(np.sum(preds == -1)))
print("Normal count (pred == 1):", int(np.sum(preds == 1)))
```

Rezultati final:

* `train rows used = 8989`
* `anomaly count = 1011`
* `normal count = 8989`

Crosstab:

```text
pred   -1     1
y
0     158  9168
1     498     43
```

Në terma të klasës `forbid` (`label=1`), modeli ka:

* `precision = 0.5351`
* `recall = 1.0000`
* `f1-score = 0.6972`

Kjo do të thotë se One-Class SVM kap të gjitha rastet `forbid`, por një pjesë e rasteve normale i etiketon gabimisht si anomali.

---

#### Ruajtja e modelit

```bash
python -m anomaly_models.one_class_svm --save --export-results
```

### 7.5 Trajnimi me Elliptic Envelope

Në këtë fazë u shtua edhe algoritmi **Elliptic Envelope** për anomaly detection mbi dataset-in e përpunuar.

Elliptic Envelope është një model **jo-supervised** që bazohet në kovariancën robuste (Minimum Covariance Determinant). Ky algoritëm supozon që të dhënat normale ndjekin një **shpërndarje Gaussiane shumëdimensionale** dhe ndërton një kufi eliptik rreth tyre. Pikat që bien jashtë kësaj zone klasifikohen si anomali.

Kjo qasje është e ndryshme nga modelet e tjera të përdorura në projekt:
- **Isolation Forest** bazohet në pemë vendimmarrëse
- **LOF** bazohet në dendësinë lokale të fqinjëve
- **One-Class SVM** bazohet në kufirin e SVM-it

Ndërsa **Elliptic Envelope** bazohet në shpërndarjen statistikore të të dhënave, duke e bërë të përshtatshëm për rastet kur të dhënat normale kanë strukturë afërsisht Gaussiane.

Trajnimi mund të bëhet me:

```bash
python train_anomaly.py --method elliptic_envelope --save --export-results
```

ose direkt:

```bash
python -m anomaly_models.elliptic_envelope --save --export-results
```

Parametrat kryesorë:

* `contamination = 0.05`
* `support_fraction = automatic` (llogaritet automatikisht nga algoritmi)
* `random_state = 42`

#### Rezultati final

* `anomaly count = 500`
* `normal count = 9500`

Crosstab:

```
pred   -1     1
y
0     291  9168
1     339   202
```

Nga rezultatet vërehet se modeli ka identifikuar 209 nga 541 rastet `forbid` si anomali, duke kapur rreth 39% të rasteve të pazakonta.

### 7.6 Si ekzekutohen të katër algoritmet

Përmes wrapper-it:

```bash
python train_anomaly.py --method isolation_forest --save --export-results
python train_anomaly.py --method local_outlier_factor --save --export-results
python train_anomaly.py --method one_class_svm --save --export-results
python train_anomaly.py --method elliptic_envelope --save --export-results
```

Direkt nga moduli:

```bash
python -m anomaly_models.isolation_forest --save --export-results
python -m anomaly_models.local_outlier_factor --save --export-results
python -m anomaly_models.one_class_svm --save --export-results
python -m anomaly_models.elliptic_envelope --save --export-results
```

### 7.7 Tabela krahasuese e Fazës 2

| Algoritmi | Parametrat kryesorë | Anomaly count | Normal count | Crosstab (y=0 / y=1) |
|---|---|---:|---:|---|
| Isolation Forest | `contamination=0.05`, `n_estimators=100` | 500 | 9500 | `0 -> -1:171, 1:9288` ; `1 -> -1:389, 1:152` |
| Local Outlier Factor | `contamination=0.05`, `n_neighbors=20` | 417 | 9583 | `0 -> -1:316, 1:9143` ; `1 -> -1:294, 1:257` |
| One-Class SVM | `nu=0.05`, `kernel=rbf`, `gamma=scale`, `train_on_normal_only=true` | 1011 | 8989 | `0 -> -1:158, 1:9301` ; `1 -> -1:498, 1:43` |
| Elliptic Envelope | `contamination=0.05`, `support_fraction=automatic`, `random_state=42` | 500 | 9500 | `0 -> -1:291, 1:9168` ; `1 -> -1:339, 1:202` |

Në bazë të rezultateve, One-Class SVM rezulton modeli më i mirë, pasi kap pothuajse të gjitha anomalitë duke mbajtur pak gabime, duke ofruar balancën më të mirë. Isolation Forest jep performancë të qëndrueshme dhe të balancuar, por kap më pak anomalitë krahasuar me SVM. Elliptic Envelope është mesatar, pasi kap një pjesë të anomalive por jo në mënyrë optimale. Ndërsa Local Outlier Factor (LOF) përshtatet më se paku nga keta algortime me këtë dataset, duke mos arritur të identifikojë në mënyrë efektive anomalitë.

### 7.8 Rezultatet e ruajtura

- `processedfiles/isolation_forest_results.csv`
- `processedfiles/isolation_forest_anomalies_only.csv`
- `processedfiles/local_outlier_factor_results.csv`
- `processedfiles/local_outlier_factor_anomalies_only.csv`
- `processedfiles/one_class_svm_results.csv`
- `processedfiles/one_class_svm_anomalies_only.csv`
- `processedfiles/elliptic_envelope_results.csv`
- `processedfiles/elliptic_envelope_anomalies_only.csv`
- `models/isolation_forest/`
- `models/local_outlier_factor/`
- `models/one_class_svm/`
- `models/elliptic_envelope/`

---

## Faza III: Permiresimi i modeleve dhe perdorimi i veglave

Ne fazen e trete projekti kalon nga trajnimi baze i modeleve ne permiresim,
krahasim dhe interpretim strategjik te rezultateve. Qellimi i kesaj faze eshte
rritja e performances se algoritmeve, dokumentimi i qarte i rezultateve dhe
perdorimi i veglave qe e bejne projektin me te pershtatshem per analizim,
gjurmim dhe vendimmarrje. 

Modelet nuk vleresohen vetem me numrin e anomalive te gjetura, por edhe me metrika si `precision`, `recall`, `F1-score`, `accuracy` dhe `confusion matrix`. Keshtu mund te kuptohet jo vetem cili model gjen me shume anomali, por edhe cili model jep me pak alarme te rreme dhe cili eshte me i pershtatshem per monitorim te audit logs.

Moduli:

```text
anomaly_models/model_improvement.py
```

Ky modul ben:

- ndarjen `train/test` me `stratify`, qe ruan raportin mes klasave `allow` dhe `forbid`;
- trajnimin e modeleve baseline per krahasim;
- kerkimin e parametrave me `grid search`;
- vleresimin me `accuracy`, `precision`, `recall`, `f1-score`, `classification report` dhe `confusion matrix`;
- krahasimin final te modeleve ne nje tabele te vetme.

Mund te ekzekutohet direkt:

```bash
python -m anomaly_models.model_improvement
```

ose permes wrapper-it kryesor:

```bash
python train_anomaly.py --method phase3
python train_anomaly.py --method model_improvement
```

### 8.2 Algoritmet e permiresuara

Ne fazen e trete u permiresuan te gjitha modelet e anomaly detection qe ishin ne
fazen e dyte. Kjo e ben krahasimin me te drejte, sepse secili model ka versionin
baseline dhe versionin e optimizuar.

1. **Isolation Forest**
   - Faza 2: `contamination=0.05`, `n_estimators=100`
   - Faza 3: testim i kombinimeve `contamination=(0.03, 0.05, 0.08, 0.1)` dhe `n_estimators=(100, 200)`
   - konfigurimi me i mire: `contamination=0.05`, `n_estimators=200`

2. **Local Outlier Factor (LOF)**
   - Faza 2: `n_neighbors=20`, `contamination=0.05`
   - Faza 3: testim i kombinimeve `n_neighbors=(10, 20, 30)` dhe `contamination=(0.05, 0.1, 0.2)`
   - konfigurimi me i mire: `n_neighbors=10`, `contamination=0.05`

3. **One-Class SVM**
   - Faza 2: `nu=0.05`, `kernel=rbf`, `gamma=scale`, `train_on_normal_only=true`
   - Faza 3: testim i kombinimeve `kernel=(rbf, linear)`, `nu=(0.01, 0.05, 0.1)` dhe `gamma=(scale, auto)`
   - konfigurimi me i mire: `nu=0.01`, `kernel=rbf`, `gamma=scale`

4. **Elliptic Envelope**
   - Faza 2: `contamination=0.05`, `support_fraction=automatic`
   - Faza 3: testim i kombinimeve `contamination=(0.03, 0.05, 0.08, 0.1)` dhe `support_fraction=(automatic, 0.7, 0.9)`
   - konfigurimi me i mire: `contamination=0.05`, `support_fraction=0.9`

One-Class SVM vazhdon te trajnohet vetem me rastet normale (`allow = 0`),
sepse kjo e ndihmon modelin te mesoje kufirin e sjelljes normale dhe te
sinjalizoje rastet `forbid` si devijime.

### 8.3 Krahasimi i rezultateve me fazen paraprake

Krahasimi i fazave eshte bere duke perdorur rezultatet e fazes se dyte si
baseline dhe rezultatet e fazes se trete si modele te optimizuara.

Komandat kryesore per ekzekutim:

```bash
python -B train_anomaly.py --method isolation_forest
python -B train_anomaly.py --method local_outlier_factor
python -B train_anomaly.py --method one_class_svm
python -B train_anomaly.py --method elliptic_envelope
python -B train_anomaly.py --method phase3
```

Rezultatet e fazes 2:

| Algoritmi | Parametrat kryesore | Anomaly count | Normal count | Crosstab (y=0 / y=1) |
|---|---|---:|---:|---|
| Isolation Forest | `contamination=0.05`, `n_estimators=100` | 500 | 9500 | `0 -> -1:171, 1:9288`; `1 -> -1:389, 1:152` |
| Local Outlier Factor | `contamination=0.05`, `n_neighbors=20` | 417 | 9583 | `0 -> -1:316, 1:9143`; `1 -> -1:294, 1:257` |
| One-Class SVM | `nu=0.05`, `kernel=rbf`, `gamma=scale`, `train_on_normal_only=true` | 1011 | 8989 | `0 -> -1:158, 1:9301`; `1 -> -1:498, 1:43` |
| Elliptic Envelope | `contamination=0.05`, `support_fraction=automatic`, `random_state=42` | 500 | 9500 | `0 -> -1:291, 1:9168`; `1 -> -1:339, 1:202` |

Rezultatet e fazes 3 jane llogaritur me `train/test split`, prandaj perdoren per
vleresim me te qendrueshem te performances:

| Modeli | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Elliptic Envelope (optimized) | 0.9830 | 0.9111 | 0.7593 | 0.8283 |
| One-Class SVM (optimized) | 0.9770 | 0.7067 | 0.9815 | 0.8217 |
| Elliptic Envelope (baseline) | 0.9780 | 0.8478 | 0.7222 | 0.7800 |
| One-Class SVM (baseline) | 0.9490 | 0.5143 | 1.0000 | 0.6792 |
| Isolation Forest (optimized) | 0.9580 | 0.6333 | 0.5278 | 0.5758 |
| Isolation Forest (baseline) | 0.9500 | 0.5435 | 0.4630 | 0.5000 |
| Local Outlier Factor (optimized) | 0.9000 | 0.1034 | 0.1111 | 0.1071 |
| Local Outlier Factor (baseline) | 0.9015 | 0.0991 | 0.1019 | 0.1005 |

Krahasimi tregon se secili model u testua me baseline dhe me konfigurim te
optimizuar:

- **Elliptic Envelope** u rrit nga `F1=0.7800` ne `F1=0.8283` dhe doli modeli me F1-score me te larte.
- **One-Class SVM** u rrit nga `F1=0.6792` ne `F1=0.8217` dhe mbajti recall shume te larte (`0.9815`).
- **Isolation Forest** u rrit nga `F1=0.5000` ne `F1=0.5758`.
- **LOF** pati vetem permiresim te vogel nga `F1=0.1005` ne `F1=0.1071`.

Nga krahasimi shihet se permiresimi nuk eshte i njejte per te gjitha modelet.
Elliptic Envelope dhe One-Class SVM japin rezultatet me te forta, Isolation
Forest permiresohet ne menyre te moderuar, ndersa LOF mbetet modeli me
performance me te ulet per kete dataset.

Confusion matrix per Elliptic Envelope te optimizuar:

```text
[[1884    8]
 [  26   82]]
```

Interpretimi:

- `1884` raste normale u klasifikuan sakte si normale;
- `8` raste normale u shenuan gabimisht si anomali;
- `26` raste anomali nuk u kapen;
- `82` raste anomali u detektuan sakte.

Confusion matrix per One-Class SVM te optimizuar:

```text
[[1848   44]
 [   2  106]]
```

Interpretimi:

- `1848` raste normale u klasifikuan sakte si normale;
- `44` raste normale u shenuan gabimisht si anomali;
- `2` raste anomali nuk u kapen;
- `106` raste anomali u detektuan sakte.

Keto rezultate tregojne dy strategji te ndryshme: Elliptic Envelope jep me pak
alarme te rreme, ndersa One-Class SVM kap pothuajse te gjitha anomalite.

### 8.4 Si lexohen rezultatet

Ne kete projekt klasa `forbid` trajtohet si sjellje me e dyshimte ose me e
pazakonte, ndersa `allow` si sjellje normale.

- **Precision** tregon sa prej rasteve te shenuara si anomali jane vertet anomali.
  Rritja e precision do te thote me pak alarme te rreme.
- **Recall** tregon sa prej anomalive reale jane kapur nga modeli.
  Recall i larte eshte i rendesishem sepse ne audit logs nuk duam te na ikin rastet e rrezikshme.
- **F1-score** kombinon precision dhe recall.
  Ky eshte treguesi me i dobishem kur kemi klasa te pabalancuara.
- **Confusion matrix** tregon saktesisht sa raste jane klasifikuar sakte dhe sa gabim.

Modeli me i mire sipas `F1-score` ne fazen 3 eshte **Elliptic Envelope i
optimizuar** (`F1=0.8283`). **One-Class SVM i optimizuar** mbetet shume i
rendesishem sepse ka recall me te larte (`0.9815`) dhe kap pothuajse te gjitha
anomalite. Zgjedhja finale varet nga prioriteti operativ: me pak alarme te
rreme ose kapje sa me e larte e anomalive.

## Tools te perdorura ne projekt

Ne projekt jane perdorur keto vegla dhe biblioteka:

| Tool / biblioteka | Ku perdoret | Pse perdoret |
|---|---|---|
| `pandas` | `data.py`, `preprocessing_modules.py`, modelet | Lexim, pastrim, transformim dhe analizim i dataset-it |
| `numpy` | modelet | Llogaritje numerike dhe numerim i predikimeve |
| `scikit-learn` | modelim dhe vleresim | Modelet ML, scaling, metrika dhe train/test split |
| `joblib` | `anomaly_models/` | Ruajtja e modeleve, scaler-it dhe feature columns |
| `StandardScaler` | anomaly detection | Normalizim i features para modeleve |
| `ydata-profiling` | raportim eksplorues | Gjenerim i raportit HTML per EDA |
| `MLflow` | gjurmim eksperimenti | Ruajtja e parametrave, metrikave dhe modelit |
| `RandomForestClassifier` | modelim i mbikqyrur | Eksperimente krahasuese kur target-i eshte i qarte |


Për interpretim lokal të Isolation Forest është shtuar edhe një integrim me
**LIME**. Përdorimi bazë është:

```bash
python -m anomaly_models.isolation_forest_lime --anomaly-only
```

Ose për një rresht specifik:

```bash
python -m anomaly_models.isolation_forest_lime --row-index 123
```

Për vizualizim interaktiv të rezultateve të Isolation Forest është shtuar edhe
**Plotly**. Kjo gjeneron një dashboard HTML me shpërndarjen e decision score,
PCA projection dhe numrin e anomalive:

```bash
python -m anomaly_models.isolation_forest_viz
```

Perdorimi i ketyre veglave e ben projektin me te qarte per analizim dhe me te
lehte per riprodhim, sepse rezultatet lidhen me parametrat, dataset-in dhe
modelin perkates.


### MLFlow

MLflow është një mjet që përdoret për të menaxhuar dhe ndjekur procesin e zhvillimit të modeleve të machine learning. Ai përdoret për vizualizimin e rezultateve, duke treguar metrika si accuracy dhe F1-score në një ndërfaqe grafike, si dhe për krahasimin e eksperimenteve të ndryshme. Përveç kësaj, MLflow ruan parametrat dhe modelet, duke e bërë më të lehtë analizimin, organizimin dhe përzgjedhjen e modelit më të mirë.

<img width="1222" height="452" alt="Screenshot 2026-05-13 at 20 17 42" src="https://github.com/user-attachments/assets/5cf09749-6454-4dcd-8a78-7fce84d423e4" />
Ky dataset përmban gjithsej 10,000 vëzhgime (rreshta) dhe 14 variabla (kolona). Ai është plotësisht i pastër – nuk ka asnjë qelizë të dhënash të munguar (0% të dhëna të humbura) dhe as rreshta të dyfishuar. Kjo e bën dataset-in ideal për trajnimin e modeleve të makinerisë pa pasur nevojë për pastrim paraprak të të dhënave.

Nga 14 variablat, 6 prej tyre janë të tipit kategorik (p.sh. emra, etiketa, vlera tekstuale), ndërsa 8 variablat e tjerë janë numerikë (të plotë ose dhjetorë). Kjo përzierje e llojeve të variablave e bën dataset-in të përshtatshëm për probleme të ndryshme të mësimit të makinerive, duke përfshirë klasifikimin, regresionin dhe analizën e anomalive.

Madhësia totale e dataset-it në memorie është rreth 1.1 MiB, ndërsa çdo vëzhgim mesatarisht zë 112 bytes. 

<img width="1189" height="471" alt="Screenshot 2026-05-13 at 20 20 39" src="https://github.com/user-attachments/assets/da87b01e-e89c-46a8-b9cd-892a23a1c428" />

Tipari protoPayload.methodName__le nuk ka asnjë vlerë të munguar – të gjitha 10,000 rreshtat janë të plotësuar me vlera reale nga 0 deri në 15. Ai përmban vetëm 16 vlera të dallueshme, gjë që tregon se është një tipar me përsëritje të lartë. Mesatarja e tij është 4.16, ndërsa 7.9% e vlerave janë zero.

Grafiku i shpërndarjes (histogrami) për këtë tipar do të tregonte frekuencën e secilës vlerë të plotë nga 0 deri në 15. Duke qenë se vlerat janë diskrete dhe me pak variacione, grafiku më i përshtatshëm është një bar chart. Në të do të vërehej një shtyllë e lartë për vlerën 0 (786 raste), ndërsa vlerat e tjera do të kishin lartësi të ndryshme, duke formuar një shpërndarje jo uniforme.

<img width="1109" height="668" alt="Screenshot 2026-05-13 at 20 22 39" src="https://github.com/user-attachments/assets/c6fe8313-729a-42d6-8fcb-aebce5202d99" />

Ky grafik tregon marrëdhënien midis dy tipareve kryesore: timestamp_epoch_s (në boshtin horizontal) dhe timestamp_delay_s (në boshtin vertikal). Pikat e dhëna në boshtin horizontal janë afërsisht nga 1.71 deri në 1.77 – ka të ngjarë që këto të jenë vlera të shkallëzuara ose të normalizuara të kohës (p.sh. në sekonda pjesëtuar me një faktor), sepse vlerat origjinale të timestamp_epoch_s zakonisht janë numra të mëdhenj (si 1.7 miliardë). Boshti vertikal tregon timestamp_delay_s nga 0 deri në 300, që mund të jetë vonesa në sekonda.

<img width="1084" height="607" alt="Screenshot 2026-05-13 at 20 23 36" src="https://github.com/user-attachments/assets/9ab0d302-395e-4e35-bb3f-0ae9a358f686" />

Figura paraqet analizën e korrelacioneve midis 13 variablave kryesore të dataset-it. Në të përdoren disa metrika të ndryshme për matjen e lidhjes midis çifteve të tipareve Pearson's. Në anën e djathtë shfaqet një shirit ngjyrash që tregon fuqinë e korrelacionit: nga 1.00 (e kuqe e errët) për korrelacion të lartë pozitiv, në 0.00 (e bardhë) për pa korrelacion, e deri në -1.00 (e kaltër e errët) për korrelacion të lartë negativ.
Nëpërmjet hartës së nxehtësisë (heatmap) mund të identifikohen vizualisht çiftet e tipareve që janë shumë të lidhura pozitivisht (p.sh. me ngjyrë të kuqe) ose negativisht (me ngjyrë blu). Kjo ndihmon për të zbuluar varësi të panevojshme ndërmjet tipareve (multikolinearitet) dhe për të zgjedhur cilët tipare janë më të rëndësishëm për modelin.

### JOBLIB

Joblib është një librari Python e optimizuar për serializimin e objekteve të mëdha numpy/të dhëna, e përdorur zakonisht për të ruajtur modele të trajnuara në disk dhe për t'i ngarkuar ato më vonë pa pasur nevojë të ritrajnohen.

Modelet në projet mund të gjenden në folder-in: `models` dhe në algoritmin specifik si elliptic envelope, isolation forest, one class svm dhe local outlier factor.

<img width="339" height="380" alt="Screenshot 2026-05-13 at 20 35 43" src="https://github.com/user-attachments/assets/cb3c9aa7-0c86-45f9-8eb2-a8e348873746" />




