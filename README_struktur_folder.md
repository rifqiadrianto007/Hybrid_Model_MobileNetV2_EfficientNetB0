# Struktur Folder Project

Struktur ini disiapkan untuk workflow klasifikasi penyakit daun cabai dengan TensorFlow.

## Folder Utama

- `data/raw/primer/` untuk dataset primer dari kebun warga.
- `data/raw/sekunder/` untuk dataset sekunder dari Kaggle.
- `data/processed/` untuk data hasil preprocessing bila nanti ingin disimpan permanen.
- `data/splits/` untuk file split train/validation/test.
- `models/checkpoints/` untuk model terbaik per stage training.
- `models/final/` untuk model akhir yang dipilih.
- `reports/metrics/` untuk CSV metrik, classification report, atau ringkasan evaluasi.
- `reports/figures/` untuk gambar confusion matrix dan kurva training.
- `src/` untuk skrip Python tambahan bila notebook ingin dipecah menjadi modul.
- `configs/` untuk konfigurasi path atau hyperparameter bila dibutuhkan.
- `logs/` untuk log training.
- `notebooks/` untuk notebook tambahan selain `model.ipynb`.

## Struktur Dataset yang Disarankan

Simpan gambar dengan format seperti ini:

```text
data/raw/primer/leaf curl/*.jpg
data/raw/primer/leaf spot/*.jpg
data/raw/primer/yellowish/*.jpg
data/raw/primer/healthy leaf/*.jpg

data/raw/sekunder/leaf curl/*.jpg
data/raw/sekunder/leaf spot/*.jpg
data/raw/sekunder/yellowish/*.jpg
data/raw/sekunder/healthy leaf/*.jpg
```

Notebook utama sudah diarahkan ke folder lokal proyek ini melalui `Path.cwd()`, jadi cukup letakkan dataset di bawah `data/raw/` lalu jalankan notebook dari root project.