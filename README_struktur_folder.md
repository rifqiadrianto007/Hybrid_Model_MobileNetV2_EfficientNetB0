# Struktur Folder Project

Struktur ini disiapkan untuk workflow klasifikasi penyakit daun cabai dengan TensorFlow.

## Folder Utama

- `data/raw/` untuk dataset mentah yang langsung berisi folder kelas.
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
data/raw/leaf curl/*.jpg
data/raw/leaf spot/*.jpg
data/raw/yellowish/*.jpg
data/raw/healthy leaf/*.jpg
```

Notebook utama sudah diarahkan ke folder lokal proyek ini melalui `Path.cwd()`, jadi cukup letakkan dataset di bawah `data/raw/` lalu jalankan notebook dari root project.