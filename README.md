# Tomato Leaf Disease Classification Using Deep Learning

Sistem klasifikasi penyakit daun tomat berbasis Deep Learning menggunakan transfer learning dan model fusion. Proyek ini membandingkan performa MobileNetV2, EfficientNet-B0, dan Fusion Model (MobileNetV2 + EfficientNet-B0) untuk mengklasifikasikan kondisi daun tomat ke dalam 4 kelas.

## Gambaran Proyek

Tujuan utama proyek ini adalah membangun model klasifikasi citra daun tomat yang mampu mengidentifikasi penyakit secara otomatis menggunakan pendekatan transfer learning.

### Kelas Klasifikasi

- Leaf Curl
- Leaf Spot
- Yellowish
- Healthy Leaf

### Fitur Utama

- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Mixed Precision Training
- Early Stopping
- Learning Rate Scheduler
- Fusion Model Architecture
- Multi-Metric Evaluation

---

## Dataset

Dataset terdiri dari:

- Dataset Primer
- Dataset Sekunder

Distribusi data:

- Train: 70%
- Validation: 15%
- Test: 15%

Format gambar:

- JPG
- JPEG
- PNG
- BMP
- WEBP

---

## Arsitektur Model

### MobileNetV2

Model ringan berbasis transfer learning yang cocok untuk deployment pada perangkat dengan sumber daya terbatas.

### EfficientNet-B0

Model CNN dengan pendekatan compound scaling untuk meningkatkan efisiensi dan performa.

### Fusion Model

Menggabungkan feature extractor MobileNetV2 dan EfficientNet-B0 melalui proses feature concatenation sebelum klasifikasi akhir.

```text
Input Image
     │
     ▼
MobileNetV2
     │
     ├────────┐
     │        │
     ▼        ▼
EfficientNet-B0
     │
     ▼
Feature Fusion
     │
     ▼
Fully Connected Layer
     │
     ▼
Softmax Output
```

---

## Struktur Direktori

```text
project-root/
│
├── data/
│   ├── raw/
│   ├── splits/
│   └── processed/
│
├── scripts/
│   ├── 01_split_dataset_V1.py
│   └── 02_preprocess_V2.py
│
├── notebooks/
│   └── model.ipynb
│
├── reports/
│
└── README.md
```

---

## Teknologi yang Digunakan

### Deep Learning

- PyTorch
- Torchvision

### Data Processing

- NumPy
- Pandas
- Pillow

### Machine Learning

- Scikit-Learn

### Visualisasi

- Matplotlib
- Seaborn

---

## Instalasi

### Clone Repository

```bash
git clone https://github.com/username/repository.git
cd repository
```

### Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/MacOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Contoh requirements:

```txt
torch
torchvision
numpy
pandas
pillow
scikit-learn
matplotlib
seaborn
jupyter
```

---

## Persiapan Dataset

Susun dataset sebagai berikut:

```text
data/
└── raw/
    ├── primer/
    │   ├── leaf curl/
    │   ├── leaf spot/
    │   ├── yellowish/
    │   └── healthy leaf/
    │
    └── sekunder/
        ├── leaf curl/
        ├── leaf spot/
        ├── yellowish/
        └── healthy leaf/
```

---

## Pipeline Penelitian

### 1. Split Dataset

```bash
python scripts/01_split_dataset_V1.py
```

Proses:

- Menggabungkan dataset primer dan sekunder
- Label inference berdasarkan struktur folder
- Stratified split
- Pembuatan metadata CSV

### Output

```text
data/splits/
├── train/
├── validation/
├── test/
├── train.csv
├── validation.csv
├── test.csv
└── all_data.csv
```

---

### 2. Preprocessing Dataset

```bash
python scripts/02_preprocess_V2.py
```

Tahapan:

#### Train

- Random Crop
- Resize 224×224
- Augmentation
- Normalization

#### Validation & Test

- Center Crop
- Resize 224×224
- Normalization

Output:

```text
data/processed/
├── train/
├── validation/
└── test/
```

---

### 3. Pelatihan Model

```bash
jupyter notebook notebooks/model.ipynb
```

Konfigurasi umum:

```python
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 50
FINE_TUNE_EPOCHS = 25
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
```

Fitur training:

- Transfer Learning
- Fine-Tuning
- Mixed Precision
- Early Stopping
- Model Checkpointing

---

## Evaluasi Model

Metrik yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Visualisasi:

- Training History
- ROC Curve
- Confusion Matrix
- Model Comparison

---

## Reproduksi Eksperimen

```bash
# 1. Split dataset
python scripts/01_split_dataset_V1.py

# 2. Preprocess dataset
python scripts/02_preprocess_V2.py

# 3. Jalankan notebook
jupyter notebook notebooks/model.ipynb
```

---

## Ringkasan

| Komponen | Detail |
|-----------|-----------|
| Task | Multi-Class Image Classification |
| Dataset | Tomato Leaf Dataset |
| Jumlah Kelas | 4 |
| Input Size | 224×224 |
| Split | 70:15:15 |
| Backbone 1 | MobileNetV2 |
| Backbone 2 | EfficientNet-B0 |
| Model Utama | Fusion Model |
| Framework | PyTorch |
| Transfer Learning | Ya |
| Fine-Tuning | Ya |
| Evaluasi | Accuracy, Precision, Recall, F1, ROC-AUC |
