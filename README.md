# Klasifikasi Sentimen Ulasan: TF-IDF + Machine Learning (Next.js)

Proyek UAS Machine Learning — topik **klasifikasi data teks**.

**Pipeline:** Preprocessing teks → TF-IDF → {Multinomial Naive Bayes, Linear SVM, Logistic Regression}
→ model terbaik (macro-F1 cross-validation tertinggi) dipakai oleh aplikasi web.

Dataset: SmSA (IndoNLU) `test_preprocess.tsv`, 500 dokumen, 3 kelas (negative / neutral / positive).

## Struktur

```
ml/train.py          # preprocessing, TF-IDF, perbandingan 3 model, tuning, ekspor model
model/               # model.json + metrics.json (dibuat oleh train.py)
src/lib/sentiment.ts # inferensi di JS (identik dengan scikit-learn)
src/lib/model-store.ts
src/app/page.tsx             # halaman Prediksi
src/app/perbandingan/page.tsx# halaman Perbandingan Model
src/app/api/predict/route.ts # API POST /api/predict
```

## Menjalankan

### 1. Training (Python 3.9+)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt
python ml/train.py                 # mengunduh dataset otomatis ke data/
```
Output: `model/model.json`, `model/metrics.json`, dan gambar grafik/confusion matrix di `ml/outputs/`
(bisa langsung dipakai di laporan paper). Jika tidak ada internet, unduh file TSV secara manual lalu:
`python ml/train.py --data path/ke/test_preprocess.tsv`

### 2. Aplikasi web (Node.js 18.18+)
```bash
npm install
npm run dev        # http://localhost:3000
```
Jika training diulang, cukup muat ulang halaman (model dimuat ulang otomatis).

## Catatan desain

- **Tidak ada kebocoran data:** TF-IDF dilatih di dalam fold CV (Pipeline), model dipilih dari CV
  pada data latih, lalu dilaporkan di data uji 20% yang tidak dipakai tuning.
- **Kata negasi** (tidak, bukan, belum, jangan, tanpa, kurang) tidak dibuang sebagai stopword.
- **Paritas Python ↔ JS:** preprocessing, tokenisasi, n-gram, TF-IDF, dan skor linear ditulis ulang di
  TypeScript, dan telah diverifikasi menghasilkan label identik dengan scikit-learn untuk ketiga model.
- Ketiga model berbentuk skor linear `bias + Σ x·w`, sehingga model mana pun yang menang bisa dipakai
  tanpa mengubah kode aplikasi.
- Probabilitas Linear SVM bukan probabilitas terkalibrasi (UI memberi catatan bila SVM yang menang).
