
"use client";

import { useState } from "react";
import type { Prediction } from "@/lib/sentiment";

const LABEL_ID: Record<string, string> = {
  positive: "Positif",
  neutral: "Netral",
  negative: "Negatif"
};

type ModelRow = {
  key: string;
  name: string;
  isBest: boolean;
  prediction: Prediction;
};

const EXAMPLES = [
  "Barangnya bagus banget, pengiriman cepat, penjual ramah. Pasti beli lagi!",
  "Kecewa berat, produk tidak sesuai deskripsi dan pengiriman lambat sekali. Tidak akan beli lagi.",
  "Barangnya standar saja, harganya lumayan, cukup sesuai dengan yang di foto."
];

const MODELS = [
  { key: "lr", name: "Logistic Regression" },
  { key: "mnb", name: "Multinomial Naive Bayes (MNB)" },
  { key: "svm", name: "Linear SVM" }
];

export default function Home() {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("svm");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [allResults, setAllResults] = useState<ModelRow[] | null>(null);

  async function analyze() {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/predict-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          selectedModel
        })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error ?? "Terjadi kesalahan.");
      }

      const rows = data.results as ModelRow[];
      const selected = rows.find(
        (r) => r.key === selectedModel
      );

      if (!selected) {
        throw new Error("Hasil model tidak ditemukan.");
      }

      setAllResults(rows);
      setResult(selected.prediction);

    } catch (e) {
      setResult(null);
      setAllResults(null);
      setError(
        e instanceof Error ? e.message : "Terjadi kesalahan."
      );
    } finally {
      setLoading(false);
    }
  }

  const sorted = result
    ? [...result.scores].sort(
        (a, b) => b.probability - a.probability
      )
    : [];

  return (
    <>
      <h1>Cek Sentimen Ulasan</h1>

      <p className="lead">
        Masukkan ulasan produk dalam bahasa Indonesia, lalu
        sistem akan menentukan apakah sentimennya positif,
        netral, atau negatif.
      </p>

      <div className="model-selector">
        <label htmlFor="model" className="small">
          Pilih Model Machine Learning
        </label>

        <select
          id="model"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={loading}
        >
          {MODELS.map((model) => (
            <option key={model.key} value={model.key}>
              {model.name}
            </option>
          ))}
        </select>
      </div>

        <label htmlFor="review" className="small">
          Tulis ulasan di sini
        </label>

        <textarea
          id="review"
          value={text}
          maxLength={1000}
          placeholder="Contoh: Barangnya bagus, pengiriman cepat, penjual ramah!"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (
              (e.ctrlKey || e.metaKey) &&
              e.key === "Enter" &&
              text.trim() &&
              !loading
            ) {
              analyze();
            }
          }}
        />

        <div className="row">
          <span className="small">Contoh:</span>

          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              className="chip"
              onClick={() => setText(ex)}
            >
              Contoh {i + 1}
            </button>
          ))}

          <span className="spacer" />

          <button
            className="small"
            onClick={() => {
              setText("");
              setResult(null);
              setAllResults(null);
              setError(null);
            }}
          >
            Hapus
          </button>

          <button
            className="primary"
            disabled={!text.trim() || loading}
            onClick={analyze}
          >
            {loading ? "Memproses…" : "Cek Sentimen"}
          </button>
        </div>
        
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      {result && (
        <section className="card" aria-live="polite">
          <div className="row" style={{ marginTop: 0 }}>
            <span className={`badge ${result.label}`}>
              {LABEL_ID[result.label] ?? result.label}
            </span>

            <span className="small">
              Diprediksi oleh model {result.modelName}
            </span>
          </div>

          <div className="bars">
            {sorted.map((s) => (
              <div className="bar-row" key={s.label}>
                <span>{LABEL_ID[s.label] ?? s.label}</span>

                <div className="track">
                  <div
                    className={`fill ${s.label}`}
                    style={{
                      width: `${s.probability * 100}%`
                    }}
                  />
                </div>

                <span style={{ textAlign: "right" }}>
                  {(s.probability * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          {result.probabilityMode === "approximate" && (
            <p className="small">
              Catatan: Linear SVM tidak menghasilkan probabilitas
              asli, jadi persentase di atas hanya skor perbandingan
              antar kelas.
            </p>
          )}

          {result.warning && (
            <div className="warn">{result.warning}</div>
          )}

          {result.topTerms.length > 0 && (
            <>
              <h2 style={{ marginTop: 22 }}>
                Kata kunci yang memengaruhi hasil
              </h2>

              <div className="terms">
                {result.topTerms.map((t) => (
                  <span
                    key={t.term}
                    className={`term ${
                      t.contribution > 0 ? "up" : "down"
                    }`}
                    title={`kontribusi ${t.contribution.toFixed(3)}`}
                  >
                    {t.contribution > 0 ? "▲" : "▼"} {t.term}
                  </span>
                ))}
              </div>

              <p className="small">
                ▲ kata yang memperkuat hasil prediksi,
                ▼ kata yang melemahkannya.
              </p>
            </>
          )}

          <h2 style={{ marginTop: 18 }}>
            Teks setelah dibersihkan
          </h2>

          <div className="mono">
            {result.cleaned || "(kosong)"}
          </div>

          <p className="small">
            {result.knownTokens} dari {result.totalTokens} kata
            ada di kosakata model.
          </p>
        </section>
      )}

      {allResults && (
        <section className="card">
          <h2>Hasil dari ketiga algoritma</h2>

          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Algoritma</th>
                  <th>Prediksi</th>
                  <th>Negatif</th>
                  <th>Netral</th>
                  <th>Positif</th>
                </tr>
              </thead>

              <tbody>
                {allResults.map((m) => (
                  <tr
                    key={m.key}
                    className={
                      m.key === selectedModel ? "win" : ""
                    }
                  >
                    <td>
                      {m.name}
                      {m.key === selectedModel && (
                        <span className="tag">DIPILIH</span>
                      )}
                      {m.isBest && (
                        <span className="tag">TERBAIK</span>
                      )}
                    </td>

                    <td style={{ textAlign: "left" }}>
                      <span
                        className={`badge ${m.prediction.label}`}
                        style={{
                          fontSize: 14,
                          padding: "2px 10px"
                        }}
                      >
                        {LABEL_ID[m.prediction.label] ??
                          m.prediction.label}
                      </span>
                    </td>

                    {["negative", "neutral", "positive"].map(
                      (l) => {
                        const s = m.prediction.scores.find(
                          (x) => x.label === l
                        );

                        return (
                          <td key={l}>
                            {s
                              ? (s.probability * 100).toFixed(1)
                              : "-"}%
                          </td>
                        );
                      }
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="small">
            Persentase Linear SVM bukan probabilitas asli,
            hanya skor perbandingan antar kelas.
          </p>
        </section>
      )}

      <section className="card">
        <h2>Bagaimana cara kerjanya?</h2>

        <ol className="steps">
          <li>
            <b>1. Bersihkan teks</b>
            Huruf dikecilkan, URL, angka, dan simbol dibuang,
            kata gaul dibakukan, kata umum dihapus.
          </li>

          <li>
            <b>2. Ubah jadi angka</b>
            Dengan TF-IDF, tiap kata diberi bobot sesuai
            seberapa penting kata itu dalam ulasan.
          </li>

          <li>
            <b>3. Tentukan sentimen</b>
            Model yang dipilih menghitung skor untuk tiap
            sentimen, lalu memilih hasil tertinggi.
          </li>
        </ol>
      </section>
    </>
  );
}
