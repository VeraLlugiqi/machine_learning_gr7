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

---

### 3.2 Analiza e cilësisë së të dhënave
Para transformimeve kontrollohen:
- numri i rreshtave dhe kolonave,
- vlerat null / missing,
- kolonat bosh,
- rreshtat duplikatë,
- shpërndarja bazike e kolonave.

Ky hap ndihmon për të kuptuar sa i pastër është dataset-i dhe çfarë problemesh duhet të rregullohen.

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

---

### 3.5 Inxhinieria e veçorive (Feature Engineering)
Për ta bërë dataset-in më të dobishëm për machine learning, krijohen disa kolona të reja nga kolonat ekzistuese.

Nga kolonat datetime nxirren:
- **hour**,
- **dayofweek**,
- **is_weekend**.

Gjithashtu krijohet një kolonë si:
- **anonymous_principal**  
që tregon nëse mungon informacioni për përdoruesin / principal-in.

Këto veçori të reja ndihmojnë modelin në fazën e ardhshme të kapë modele më kuptimplota në të dhëna.

---

### 3.6 Heqja e kolonave jo të dobishme
Hiqen kolonat që nuk janë të përshtatshme për machine learning, si:

- kolona me shumë missing values,
- kolona me shumë pak variacion,
- kolona me vlera pothuajse unike për çdo rresht,
- ID ose fusha shumë specifike që nuk ndihmojnë modelin.

Kjo ul zhurmën dhe e bën dataset-in më të fokusuar.

---


### 3.7 Analiza e target-it dhe imbalance
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
```

Ky dataset:
- është i pastruar,
- ka missing values të trajtuara,
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
│   └── ml_ready.csv
├── README.md
├── requirements.txt
```

Shikimi i datasetit perfundimtar:
```python

timestamp	logName	receiveTimestamp	labels.authorization.k8s.io/decision	resource.labels.project_id	resource.labels.location	resource.labels.cluster_name	protoPayload.authenticationInfo.principalEmail	protoPayload.authorizationInfo	protoPayload.methodName	protoPayload.requestMetadata.callerIp	protoPayload.requestMetadata.callerSuppliedUserAgent	protoPayload.resourceName	protoPayload.status.code	protoPayload.status.message	anonymous_principal	timestamp__epoch_s	receiveTimestamp__epoch_s	callerIp_first_octet	logName__le	labels.authorization.k8s.io/decision__le	resource.labels.project_id__le	resource.labels.location__le	resource.labels.cluster_name__le	protoPayload.methodName__le	protoPayload.requestMetadata.callerSuppliedUserAgent__le
2024-11-03 16:38:07+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-11-03 16:38:58+00:00	forbid	project123	europe-west1	prod-cluster	admin@company.com	[{'resource': 'apis/networking.k8s.io/v1/networkpolicies', 'permission': 'io.k8s.patch'}]	io.k8s.patch	198.18.9.235	kubectl/v1.26.0 (darwin/amd64) kubernetes/b46c28f	apis/networking.k8s.io/v1/networkpolicies	0	OK	0	1730651887	1730651938	198	3	1	3	1	4	12	7
2024-07-06 11:59:19+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-07-06 12:01:26+00:00	forbid	project123	us-central1	dev-cluster	dev@company.com	[{'resource': 'apis/v1/services', 'permission': 'io.k8s.get'}]	io.k8s.get	198.19.90.169	kubectl/v1.25.0 (linux/amd64) kubernetes/a866cbe	apis/v1/services	7	forbidden: User "dev@company.com" cannot get path "apis/v1/services"	0	1720267159	1720267286	198	3	1	3	3	0	9	6
2024-03-06 04:05:32+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-03-06 04:09:49+00:00	forbid	project123	us-west1	nf-default	system:anonymous	[{'resource': 'apis/networking.k8s.io/v1/networkpolicies', 'permission': 'io.k8s.delete'}]	io.k8s.delete	198.18.39.24	kubectl/v1.25.0 (linux/amd64) kubernetes/a866cbe	apis/networking.k8s.io/v1/networkpolicies	0	OK	1	1709697932	1709698189	198	3	1	3	5	2	8	6
2024-12-17 15:24:51+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-12-17 15:24:56+00:00	forbid	project123	europe-west1	nf-default	system:anonymous	[{'resource': 'apis/rbac.authorization.k8s.io/v1/roles', 'permission': 'io.k8s.put'}]	io.k8s.put	198.19.109.172	Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36	apis/rbac.authorization.k8s.io/v1/roles	3	Invalid argument	1	1734449091	1734449096	198	3	1	3	1	2	14	1
2024-09-17 06:09:24+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-09-17 06:10:40+00:00	forbid	project123	europe-west1	nf-default	dev@company.com	[{'resource': 'global-protect/login.esp', 'permission': 'io.k8s.put'}]	io.k8s.put	198.18.83.217	kubectl/v1.25.0 (linux/amd64) kubernetes/a866cbe	global-protect/login.esp	3	Invalid argument	0	1726553364	1726553440	198	3	1	3	1	2	14	6
2024-12-06 01:12:01+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-12-06 01:13:27+00:00	allow	project123	asia-southeast1	nf-default	system:anonymous	[{'resource': 'apis/batch/v1/jobs', 'permission': 'io.k8s.get'}]	io.k8s.get	198.18.79.228	Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36	apis/batch/v1/jobs	7	forbidden: User "system:anonymous" cannot get path "apis/batch/v1/jobs"	1	1733447521	1733447607	198	3	0	3	0	2	9	2
2024-07-20 10:16:34+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-07-20 10:17:14+00:00	allow	project123	europe-west1	dev-cluster	service-account@company.iam.gserviceaccount.com	[{'resource': 'apis/v1/services', 'permission': 'io.k8s.delete'}]	io.k8s.delete	198.19.148.36	kubectl/v1.26.0 (darwin/amd64) kubernetes/b46c28f	apis/v1/services	13	Internal error	0	1721470594	1721470634	198	3	0	3	1	0	8	7
2024-07-25 20:21:32+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-07-25 20:22:21+00:00	forbid	project123	asia-southeast1	prod-cluster	dev@company.com	[{'resource': 'apis/apps/v1/deployments', 'permission': 'io.k8s.post'}]	io.k8s.post	198.18.224.49	kubectl/v1.25.0 (linux/amd64) kubernetes/a866cbe	apis/apps/v1/deployments	13	Internal error	0	1721938892	1721938941	198	3	1	3	0	4	13	6
2024-12-25 21:45:28+00:00	projects/project123/logs/cloudaudit.googleapis.com%2Factivity	2024-12-25 21:48:03+00:00	forbid	project123	asia-southeast1	test-cluster	admin@company.com	[{'resource': 'apis/v1/services', 'permission': 'io.k8s.delete'}]	io.k8s.delete	198.19.199.232	Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203	apis/v1/services	3	Invalid argument	0	1735163128	1735163283	198	3	1	3	0	6	8	3
<img width="32766" height="291" alt="image" src="https://github.com/user-attachments/assets/ea2d573d-f56c-4996-b08a-0616d7f11906" />

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
