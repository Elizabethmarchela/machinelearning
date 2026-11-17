export interface ModelFile {
  version: number;
  model_key: "mnb" | "svm" | "lr";
  model_name: string;
  labels: string[];
  ngram_range: [number, number];
  sublinear_tf: boolean;
  vocabulary: Record<string, number>;
  idf: number[];
  weights: number[][];
  bias: number[];
  probability_mode: "posterior" | "approximate";
  stopwords: string[];
  slang: Record<string, string>;
}

export interface Prediction {
  label: string;
  scores: { label: string; probability: number }[];
  cleaned: string;
  knownTokens: number;
  totalTokens: number;
  topTerms: { term: string; contribution: number }[];
  probabilityMode: "posterior" | "approximate";
  modelName: string;
  warning?: string;
}

export function createClassifier(model: ModelFile) {
  const stop = new Set(model.stopwords);
  const slang = new Map(Object.entries(model.slang));
  const vocab = new Map(Object.entries(model.vocabulary));
  const terms: string[] = [];
  for (const [term, idx] of vocab) terms[idx] = term;
  const [nMin, nMax] = model.ngram_range;
  const K = model.labels.length;
  const V = model.idf.length;

  const meanW = new Float64Array(V);
  for (let j = 0; j < V; j++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += model.weights[k][j];
    meanW[j] = s / K;
  }

  function cleanText(text: string): string {
    let t = text.toLowerCase();
    t = t.replace(/https?:\/\/\S+|www\.\S+/g, " ");
    t = t.replace(/@\S+/g, " ");
    t = t.replace(/[^a-z\s]/g, " ");
    t = t.replace(/(.)\1{2,}/g, "$1");
    const tokens = t
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => slang.get(w) ?? w)
      .filter((w) => !stop.has(w));
    return tokens.join(" ");
  }

  function predict(text: string): Prediction {
    const cleaned = cleanText(text);
    const toks = cleaned.match(/\b[a-z]{2,}\b/g) ?? [];

    const counts = new Map<number, number>();
    for (let n = nMin; n <= Math.min(nMax, toks.length); n++) {
      for (let i = 0; i + n <= toks.length; i++) {
        const idx = vocab.get(toks.slice(i, i + n).join(" "));
        if (idx !== undefined) counts.set(idx, (counts.get(idx) ?? 0) + 1);
      }
    }

    const x: [number, number][] = [];
    let norm = 0;
    for (const [j, c] of counts) {
      const tf = model.sublinear_tf ? 1 + Math.log(c) : c;
      const v = tf * model.idf[j];
      x.push([j, v]);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const e of x) e[1] /= norm;

    const raw = model.bias.slice();
    for (let k = 0; k < K; k++) for (const [j, v] of x) raw[k] += v * model.weights[k][j];

    const mx = Math.max(...raw);
    const ex = raw.map((s) => Math.exp(s - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((e) => e / sum);
    const best = probs.indexOf(Math.max(...probs));

    const topTerms = x
      .map(([j, v]) => ({ term: terms[j], contribution: v * (model.weights[best][j] - meanW[j]) }))
      .filter((t) => Math.abs(t.contribution) > 1e-9)
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 10);

    const knownTokens = toks.filter((t) => vocab.has(t)).length;
    let warning: string | undefined;
    if (toks.length === 0) warning = "Tidak ada kata bermakna setelah preprocessing.";
    else if (x.length === 0) warning = "Tidak ada kata yang dikenali model, hasil tidak dapat diandalkan.";

    return {
      label: model.labels[best],
      scores: model.labels.map((label, k) => ({ label, probability: probs[k] })),
      cleaned,
      knownTokens,
      totalTokens: toks.length,
      topTerms,
      probabilityMode: model.probability_mode,
      modelName: model.model_name,
      warning,
    };
  }

  return { predict, cleanText };
}
