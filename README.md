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

## 5. Çfarë u shtua si përmirësim
Dy hapa të mirë që mund të përfshihen pa e tepruar janë:

1. **Feature Selection i thjeshtë**
   - heq kolonat konstante
   - heq kolonat me cardinality shumë të lartë

2. **Basic Target Inspection**
   - kontrollon shpërndarjen e target-it të mundshëm
   - ndihmon për të kuptuar klasat para fazës së machine learning

Këta hapa e bëjnë projektin më të plotë, por pa kaluar në Fazën II.

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

## 8. Struktura e projektit

```text
.
├── data.py
├── preprocessing_modules.py
├── dataset.csv
├── processedfiles/
│   ├── ml_ready.csv
│   └── label_mappings.txt
├── README.md
├── requirements.txt
```

Shikimi i datasetit perfundimtar:
```python
import pandas as pd

df = pd.read_csv("processedfiles/ml_ready.csv")
print(df.head(3))
print(df.columns.tolist())

# mapping i label encoding:
with open("processedfiles/label_mappings.txt", "r", encoding="utf-8") as f:
    print(f.read().splitlines()[:20])

```
---

## 9. Teknologjitë e përdorura
Ky projekt përdor:
- Python
- pandas
- numpy

---

## 10. Përfundim
Ky projekt realizon parapërpunimin e dataset-it në mënyrë të përshtatshme për fazën përgatitore të machine learning.

Janë zbatuar vetëm hapat që kanë kuptim për këtë dataset. Fokusi kryesor është:
- cilësia e të dhënave,
- trajtimi i missing values,
- transformimi i kolonave,
- ulja e zhurmës,
- dhe përgatitja e dataset-it për modelim në Fazën II.
