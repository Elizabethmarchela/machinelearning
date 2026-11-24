import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = ("https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/"
            "smsa_doc-sentiment-prosa/test_preprocess.tsv")
LABELS = ["negative", "neutral", "positive"]
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOKEN_PATTERN = r"\b[a-z]{2,}\b"

STOPWORDS = sorted({
    "yang", "di", "ke", "dari", "dan", "atau", "itu", "ini", "untuk", "dengan", "pada",
    "dalam", "adalah", "ada", "juga", "oleh", "akan", "para", "sebagai", "saya", "aku",
    "gue", "gua", "kamu", "anda", "dia", "mereka", "kami", "kita", "nya", "lah", "kah",
    "pun", "sih", "deh", "dong", "kok", "nih", "tuh", "lalu", "kemudian", "saat",
    "ketika", "sudah", "sedang", "agar", "supaya", "bahwa", "karena", "sebuah", "suatu",
    "tersebut", "hal", "yaitu", "yakni", "dah", "kan", "lagi", "si", "sang", "pas",
})

SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "tdk": "tidak", "tak": "tidak", "bgt": "banget", "bngt": "banget",
    "yg": "yang", "udah": "sudah", "udh": "sudah", "sdh": "sudah", "krn": "karena",
    "karna": "karena", "jg": "juga", "dgn": "dengan", "utk": "untuk", "tp": "tapi",
    "sy": "saya", "blm": "belum", "bs": "bisa", "jgn": "jangan", "klo": "kalau",
    "kalo": "kalau", "emg": "memang", "emang": "memang", "aja": "saja", "doang": "saja",
}

_STOP = set(STOPWORDS)


def clean_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    tokens = [SLANG.get(w, w) for w in t.split()]
    tokens = [w for w in tokens if w not in _STOP]
    return " ".join(tokens)

def load_dataset(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] mengunduh dataset -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    texts, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        text, label = line.rsplit("\t", 1)
        label = label.strip().lower()
        if label not in LABELS:
            continue
        texts.append(text.strip())
        labels.append(label)
    return texts, np.array(labels)


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN)),
        ("clf", clf),
    ])


TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [True, False],
}

CANDIDATES = {
    "mnb": ("Multinomial Naive Bayes", MultinomialNB(),
            {"clf__alpha": [0.1, 0.3, 0.5, 1.0]}),
    "svm": ("Linear SVM", LinearSVC(max_iter=20000, random_state=SEED),
            {"clf__C": [0.1, 0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]}),
    "lr": ("Logistic Regression", LogisticRegression(max_iter=5000, random_state=SEED),
        {"clf__C": [1, 3, 10, 30, 100], "clf__class_weight": [None, "balanced"]}),
}


def evaluate(key, name, clf, grid, X_tr, y_tr, X_te, y_te):
    print(f"\n=== {name} ===")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(clf), {**TFIDF_GRID, **grid}, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = gs.predict(X_te)
    pred_ms = (time.perf_counter() - t0) / len(X_te) * 1000

    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=LABELS, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    result = {
        "key": key, "name": name,
        "best_params": {k.split("__")[1]: v for k, v in gs.best_params_.items()},
        "cv_f1_macro": float(gs.best_score_),
        "cv_f1_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
        "test": {"accuracy": float(accuracy_score(y_te, pred)),
        "precision_macro": float(pm), "recall_macro": float(rm),
        "f1_macro": float(fm), "f1_weighted": float(fw)},
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]),
        "f1": float(f[i]), "support": int(s[i])}
        for i, l in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y_te, pred, labels=LABELS).tolist(),
        "fit_seconds": round(fit_s, 2),
        "predict_ms_per_doc": round(pred_ms, 4),
    }
    print(f"best params : {result['best_params']}")
    print(f"CV macro-F1 : {result['cv_f1_macro']:.4f} (+/- {result['cv_f1_std']:.4f})")
    print(f"Test acc    : {result['test']['accuracy']:.4f} | macro-F1 {result['test']['f1_macro']:.4f}")
    return result, gs.best_estimator_

def export_model(key, name, pipe, out_path: Path):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    if key == "mnb":
        W, b = clf.feature_log_prob_, clf.class_log_prior_
        mode = "posterior"
    else:
        W, b = clf.coef_, clf.intercept_
        mode = "posterior" if key == "lr" else "approximate"
    if W.shape[0] == 1:
        W, b = np.vstack([-W, W]), np.concatenate([-b, b])
    assert list(clf.classes_) == LABELS
    model = {
        "version": 1,
        "model_key": key, "model_name": name,
        "labels": LABELS,
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": bool(vec.sublinear_tf),
        "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": np.round(vec.idf_, 6).tolist(),
        "weights": np.round(W, 6).tolist(),
        "bias": np.round(b, 6).tolist(),
        "probability_mode": mode,
        "stopwords": STOPWORDS,
        "slang": SLANG,
    }
    out_path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(f"[export] {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, "
        f"{len(model['vocabulary'])} fitur)")
    return W, vec


def top_terms(W, vec, n=10):
    inv = {i: t for t, i in vec.vocabulary_.items()}
    centered = W - W.mean(axis=0, keepdims=True)
    out = {}
    for k, label in enumerate(LABELS):
        idx = np.argsort(-centered[k])[:n]
        out[label] = [{"term": inv[int(i)], "weight": round(float(centered[k][i]), 4)} for i in idx]
    return out


def make_figures(results, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib tidak ada, lewati gambar")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    x = np.arange(len(results)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, [r["cv_f1_macro"] for r in results], w, label="CV macro-F1")
    ax.bar(x, [r["test"]["accuracy"] for r in results], w, label="Test accuracy")
    ax.bar(x + w, [r["test"]["f1_macro"] for r in results], w, label="Test macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1)
    ax.set_ylabel("Skor"); ax.legend(); ax.set_title("Perbandingan Model")
    fig.tight_layout(); fig.savefig(fig_dir / "perbandingan_model.png", dpi=200); plt.close(fig)

    for r in results:
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(r["name"], fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout(); fig.savefig(fig_dir / f"confusion_{r['key']}.png", dpi=200); plt.close(fig)
    print(f"[fig] gambar disimpan di {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "test_preprocess.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    texts, y = load_dataset(args.data)
    clean = [clean_text(t) for t in texts]
    dist = {l: int((y == l).sum()) for l in LABELS}
    print(f"[data] {len(texts)} dokumen | distribusi: {dist}")
    print(f"[prep] contoh:\n  raw  : {texts[0][:120]}\n  clean: {clean[0][:120]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        clean, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    results, estimators = [], {}
    for key, (name, clf, grid) in CANDIDATES.items():
        res, est = evaluate(key, name, clf, grid, X_tr, y_tr, X_te, y_te)
        results.append(res); estimators[key] = est

    best = max(results, key=lambda r: (round(r["cv_f1_macro"], 6), r["test"]["f1_macro"]))
    print(f"\n>>> MODEL TERBAIK: {best['name']} "
        f"(CV macro-F1 {best['cv_f1_macro']:.4f}, test macro-F1 {best['test']['f1_macro']:.4f})")

    W, vec = export_model(best["key"], best["name"], estimators[best["key"]], args.out / "model.json")
    for r in results:
        export_model(r["key"], r["name"], estimators[r["key"]], args.out / f"model_{r['key']}.json")

    metrics = {
        "settings": {"seed": SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS, "scoring": "f1_macro"},
        "dataset": {
            "source": DATA_URL, "total": len(texts),
            "train": len(X_tr), "test": len(X_te),
            "class_distribution": dist,
            "avg_tokens_raw": round(float(np.mean([len(t.split()) for t in texts])), 1),
            "avg_tokens_clean": round(float(np.mean([len(c.split()) for c in clean])), 1),
        },
        "preprocessing_example": {"raw": texts[0], "clean": clean[0]},
        "labels": LABELS,
        "models": results,
        "winner": best["key"],
        "winner_reason": "macro-F1 cross-validation (5-fold) tertinggi pada data latih",
        "vocab_size": len(vec.vocabulary_),
        "top_terms": top_terms(W, vec),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] {args.out / 'metrics.json'}")
    make_figures(results, ROOT / "ml" / "outputs")


if __name__ == "__main__":
    main()
