# Klasifikasi Sentimen Ulasan Menggunakan TF-IDF dan Machine Learning

Proyek UAS Machine Learning yang dibuat untuk mengklasifikasikan sentimen dari teks ulasan berbahasa Indonesia ke dalam tiga kategori, yaitu **negative, neutral, dan positive**.

Aplikasi ini menggunakan metode TF-IDF untuk mengubah teks menjadi representasi numerik, kemudian membandingkan tiga algoritma machine learning, yaitu Multinomial Naive Bayes, Linear SVM, dan Logistic Regression. Model dengan hasil macro-F1 terbaik berdasarkan cross-validation akan digunakan dalam aplikasi web.

Dataset yang digunakan adalah SmSA dari IndoNLU, yaitu `test_preprocess.tsv` yang terdiri dari 500 dokumen dengan tiga kelas sentimen.

## Struktur Project

```text
ml/train.py
model/
  ├── model.json
  └── metrics.json
src/lib/
  ├── sentiment.ts
  └── model-store.ts
src/app/
  ├── page.tsx
  ├── perbandingan/page.tsx
  └── api/predict/route.ts
```

## Cara Menjalankan

### 1. Training Model

Training dilakukan menggunakan Python versi 3.9 atau lebih baru.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
python ml/train.py
```

Dataset akan diunduh secara otomatis ke folder `data/`.

Setelah proses training selesai, akan dihasilkan dua file, yaitu `model.json` dan `metrics.json` di folder `model/`. Grafik hasil evaluasi dan confusion matrix juga disimpan di `ml/outputs/` untuk digunakan dalam laporan.

Jika dataset tidak bisa diunduh otomatis, file TSV dapat diunduh secara manual dan digunakan dengan perintah:

```bash
python ml/train.py --data path/ke/test_preprocess.tsv
```

### 2. Menjalankan Aplikasi Web

Pastikan Node.js versi 18.18 atau lebih baru sudah terpasang.

```bash
npm install
npm run dev
```

Aplikasi dapat dibuka melalui `http://localhost:3000`.

## Metode dan Implementasi

Beberapa hal yang diterapkan dalam project ini:

- **Preprocessing teks:** Teks ulasan diproses sebelum diubah menjadi fitur TF-IDF.
- **Perbandingan model:** Menggunakan Multinomial Naive Bayes, Linear SVM, dan Logistic Regression untuk melihat model dengan performa terbaik.
- **Evaluasi model:** Pemilihan model dilakukan berdasarkan nilai macro-F1 tertinggi dari cross-validation.
- **Pembagian data:** TF-IDF dilatih di dalam setiap fold cross-validation. Data uji sebesar 20% juga disiapkan untuk evaluasi akhir dan tidak digunakan dalam proses tuning.
- **Penanganan kata negasi:** Kata seperti tidak, bukan, belum, jangan, tanpa, dan kurang tetap dipertahankan karena berpengaruh terhadap makna sentimen.
- **Implementasi pada web:** Model yang sudah dilatih diekspor ke JSON, kemudian digunakan kembali oleh aplikasi melalui TypeScript tanpa perlu menjalankan training ulang.
- **Konsistensi hasil:** Proses preprocessing, tokenisasi, TF-IDF, dan perhitungan skor pada TypeScript disesuaikan dengan implementasi Python. Hasil prediksi juga telah diuji dan menghasilkan label yang sama dengan scikit-learn untuk ketiga model.

Ketiga algoritma menggunakan perhitungan skor linear dengan bentuk `bias + Σ x·w`, sehingga model terbaik dapat digunakan tanpa perlu mengubah alur utama aplikasi.

**Catatan:** Jika Linear SVM menjadi model terbaik, nilai yang ditampilkan pada aplikasi bukan probabilitas yang telah dikalibrasi.
