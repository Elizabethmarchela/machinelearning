import fs from "node:fs";
import path from "node:path";
import { createClassifier, type ModelFile } from "./sentiment";

const MODEL_DIR = path.join(process.cwd(), "model");

type Classifier = ReturnType<typeof createClassifier>;
let cache: { mtime: number; clf: Classifier } | null = null;

export function getClassifier(): Classifier | null {
  const file = path.join(MODEL_DIR, "model.json");
  try {
    const mtime = fs.statSync(file).mtimeMs;
    if (!cache || cache.mtime !== mtime) {
      const model = JSON.parse(fs.readFileSync(file, "utf8")) as ModelFile;
      cache = { mtime, clf: createClassifier(model) };
    }
    return cache.clf;
  } catch {
    return null;
  }
}

const ALL_KEYS = ["mnb", "svm", "lr"] as const;
const allCache: Record<string, { mtime: number; clf: Classifier; name: string }> = {};

export function getAllClassifiers() {
  const winner = getMetrics()?.winner;
  const out: { key: string; name: string; clf: Classifier; isBest: boolean }[] = [];
  for (const key of ALL_KEYS) {
    const file = path.join(MODEL_DIR, `model_${key}.json`);
    try {
      const mtime = fs.statSync(file).mtimeMs;
      if (!allCache[key] || allCache[key].mtime !== mtime) {
        const model = JSON.parse(fs.readFileSync(file, "utf8")) as ModelFile;
        allCache[key] = { mtime, clf: createClassifier(model), name: model.model_name };
      }
      out.push({ key, name: allCache[key].name, clf: allCache[key].clf, isBest: key === winner });
    } catch {
      return null;
    }
  }
  return out;
}

export interface ModelResult {
  key: string;
  name: string;
  best_params: Record<string, unknown>;
  cv_f1_macro: number;
  cv_f1_std: number;
  test: {
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    f1_weighted: number;
  };
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix: number[][];
  fit_seconds: number;
  predict_ms_per_doc: number;
}

export interface Metrics {
  settings: { seed: number; test_size: number; cv_folds: number; scoring: string };
  dataset: {
    source: string;
    total: number;
    train: number;
    test: number;
    class_distribution: Record<string, number>;
    avg_tokens_raw: number;
    avg_tokens_clean: number;
  };
  preprocessing_example: { raw: string; clean: string };
  labels: string[];
  models: ModelResult[];
  winner: string;
  winner_reason: string;
  vocab_size: number;
  top_terms: Record<string, { term: string; weight: number }[]>;
}

export function getMetrics(): Metrics | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "metrics.json"), "utf8")) as Metrics;
  } catch {
    return null;
  }
}
