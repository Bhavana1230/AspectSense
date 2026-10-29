"""Neural sentiment back-ends.  Each exposes ``score_many(texts) -> [float in -1..1]``
(= P(positive) - P(negative)).  The lexicon engine lives in aspects.py."""
import csv
import importlib.util
import os
import threading

import joblib

from .preprocess import norm_token, tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFIDF_PATH = os.path.join(ROOT, "models", "tfidf.joblib")
TRAIN_CSV = os.path.join(ROOT, "data", "train.csv")
TRANSFORMER_MODEL = os.environ.get("ABSA_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
_lock = threading.Lock()
_cache = {}


def prep_text(t: str) -> str:
    """Normalised, slang-expanded token string used by the TF-IDF model."""
    return " ".join(w for w, _, _ in tokenize(t) if w not in ",;")


def load_rows(path=TRAIN_CSV):
    with open(path, encoding="utf-8") as f:
        return [(r["text"], r["label"]) for r in csv.DictReader(f)]


def build_tfidf_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=prep_text, analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("char", TfidfVectorizer(preprocessor=prep_text, analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
    ])
    return Pipeline([("tfidf", feats),
                     ("lr", LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced"))])


def train_tfidf(csv_path=TRAIN_CSV, out_path=TFIDF_PATH):
    rows = load_rows(csv_path)
    pipe = build_tfidf_pipeline()
    pipe.fit([r[0] for r in rows], [r[1] for r in rows])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    return pipe


class TfidfEngine:
    name = "tfidf"

    def __init__(self):
        self.pipe = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else train_tfidf()
        self.classes = list(self.pipe.classes_)

    def score_many(self, texts):
        probs = self.pipe.predict_proba(texts)
        ip, ineg = self.classes.index("positive"), self.classes.index("negative")
        return [float(p[ip] - p[ineg]) for p in probs]


class TransformerEngine:
    """Multilingual XLM-RoBERTa sentiment model (downloads once from the Hugging Face Hub)."""
    name = "transformer"

    def __init__(self, model_name=TRANSFORMER_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        id2 = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if all(v.startswith("label_") for v in id2.values()):       # LABEL_0/1/2 = neg/neu/pos
            id2 = {0: "negative", 1: "neutral", 2: "positive"}
        self.ipos = next(i for i, v in id2.items() if v.startswith("pos"))
        self.ineg = next(i for i, v in id2.items() if v.startswith("neg"))

    def score_many(self, texts, batch=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
                p = self.torch.softmax(self.model(**enc).logits, dim=-1)
                out += [float(r[self.ipos] - r[self.ineg]) for r in p]
        return out


def transformer_available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


def available_engines():
    tf = transformer_available()
    return {"lexicon": True, "tfidf": True, "transformer": tf, "hybrid": tf}


def get_neural(name):
    """Return (cached) neural engine for 'tfidf' / 'transformer'."""
    with _lock:
        if name not in _cache:
            _cache[name] = TfidfEngine() if name == "tfidf" else TransformerEngine()
        return _cache[name]
