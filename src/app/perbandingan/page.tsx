import { getMetrics, type ModelResult } from "@/lib/model-store";

export const dynamic = "force-dynamic";
export const metadata = { title: "Perbandingan Model | Analisis Sentimen" };

const LABEL_ID: Record<string, string> = { positive: "Positif", neutral: "Netral", negative: "Negatif" };
const pct = (v: number) => (v * 100).toFixed(2);

function ConfusionMatrix({ m, labels }: { m: ModelResult; labels: string[] }) {
  const max = Math.max(...m.confusion_matrix.flat(), 1);
  return (
    <div>
      <b style={{ fontSize: 14 }}>{m.name}</b>
      <table className="cm">
        <thead>
          <tr>
            <th />
            {labels.map((l) => <th key={l}>pred: {LABEL_ID[l]}</th>)}
          </tr>
        </thead>
        <tbody>
          {m.confusion_matrix.map((row, i) => (
            <tr key={i}>
              <th>aktual: {LABEL_ID[labels[i]]}</th>
              {row.map((v, j) => (
                <td key={j} style={{ background: `color-mix(in srgb, var(--accent) ${Math.round((v / max) * 75)}%, transparent)`, fontWeight: i === j ? 700 : 400 }}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Perbandingan() {
  const metrics = getMetrics();

  if (!metrics) {
    return (
      <>
        <h1>Perbandingan Model</h1>
        <div className="card">
          <p>Hasil training belum tersedia. Jalankan perintah berikut di root proyek, lalu muat ulang halaman ini:</p>
          <div className="mono">python ml/train.py</div>
        </div>
      </>
    );
  }

  const { dataset, models, winner, labels } = metrics;
  const win = models.find((m) => m.key === winner)!;

  return (
    <>
      <h1>Perbandingan Model</h1>
      <p className="lead">
        Tiga algoritma dibandingkan dengan preprocessing dan TF-IDF yang sama. Model terbaik{" "}
        (<b>{win.name}</b>) otomatis dipakai di halaman Prediksi.
      </p>

      <section className="card">
        <h2>Dataset & skema evaluasi</h2>
        <p className="small" style={{ marginTop: 0 }}>
          {dataset.total} dokumen · {dataset.train} data latih / {dataset.test} data uji (stratified) ·{" "}
          {metrics.settings.cv_folds}-fold CV pada data latih untuk tuning · kosakata model terbaik {metrics.vocab_size} fitur
        </p>
        <div className="grid2">
          {labels.map((l) => (
            <div key={l} className="metric-row" style={{ gridTemplateColumns: "70px 1fr 40px" }}>
              <span>{LABEL_ID[l]}</span>
              <div className="track"><div className={`fill ${l}`} style={{ width: `${(dataset.class_distribution[l] / dataset.total) * 100}%` }} /></div>
              <span style={{ textAlign: "right" }}>{dataset.class_distribution[l]}</span>
            </div>
          ))}
        </div>
        <p className="small">Contoh preprocessing:</p>
        <div className="mono">{metrics.preprocessing_example.raw}</div>
        <div className="mono" style={{ marginTop: 6 }}>→ {metrics.preprocessing_example.clean}</div>
      </section>

      <section className="card">
        <h2>Hasil evaluasi</h2>
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>CV macro-F1</th><th>Akurasi</th><th>Presisi</th><th>Recall</th><th>F1 macro</th><th>Waktu latih</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.key} className={m.key === winner ? "win" : ""}>
                  <td>{m.name}{m.key === winner && <span className="tag">TERBAIK</span>}</td>
                  <td>{pct(m.cv_f1_macro)} ± {pct(m.cv_f1_std)}</td>
                  <td>{pct(m.test.accuracy)}</td>
                  <td>{pct(m.test.precision_macro)}</td>
                  <td>{pct(m.test.recall_macro)}</td>
                  <td>{pct(m.test.f1_macro)}</td>
                  <td>{m.fit_seconds}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small">
          Kolom Akurasi, Presisi, Recall, dan F1 (macro) dihitung pada data uji. Pemilihan model memakai CV macro-F1 pada data latih
          agar data uji tidak ikut menentukan pilihan.
        </p>

        <div style={{ marginTop: 14 }}>
          {models.map((m) => (
            <div key={m.key} style={{ marginBottom: 14 }}>
              <b style={{ fontSize: 14 }}>{m.name}</b>
              {([["CV macro-F1", m.cv_f1_macro], ["Akurasi (uji)", m.test.accuracy], ["F1 macro (uji)", m.test.f1_macro]] as const).map(([n, v]) => (
                <div className="metric-row" key={n}>
                  <span>{n}</span>
                  <div className="track"><div className="fill accent" style={{ width: `${v * 100}%` }} /></div>
                  <span style={{ textAlign: "right" }}>{pct(v)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Metrik per kelas</h2>
        <div className="scroll">
          <table>
            <thead>
              <tr><th>Model</th><th>Kelas</th><th>Presisi</th><th>Recall</th><th>F1</th><th>Support</th></tr>
            </thead>
            <tbody>
              {models.flatMap((m) =>
                labels.map((l) => (
                  <tr key={m.key + l} className={m.key === winner ? "win" : ""}>
                    <td>{m.name}</td><td style={{ textAlign: "left" }}>{LABEL_ID[l]}</td>
                    <td>{pct(m.per_class[l].precision)}</td><td>{pct(m.per_class[l].recall)}</td>
                    <td>{pct(m.per_class[l].f1)}</td><td>{m.per_class[l].support}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Confusion matrix (data uji)</h2>
        <div className="cmgrid">
          {models.map((m) => <ConfusionMatrix key={m.key} m={m} labels={labels} />)}
        </div>
      </section>

      <section className="card">
        <h2>Hyperparameter terbaik</h2>
        <div className="scroll">
          <table>
            <thead><tr><th>Model</th><th style={{ textAlign: "left" }}>Parameter</th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.key}>
                  <td>{m.name}</td>
                  <td className="mono" style={{ textAlign: "left", border: "none" }}>
                    {Object.entries(m.best_params).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Kata paling berpengaruh per kelas ({win.name})</h2>
        <div className="grid2">
          {labels.map((l) => (
            <div key={l}>
              <b style={{ fontSize: 14 }}>{LABEL_ID[l]}</b>
              <div className="terms">
                {metrics.top_terms[l].map((t) => <span key={t.term} className={`term ${l === "positive" ? "up" : l === "negative" ? "down" : ""}`}>{t.term}</span>)}
              </div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
