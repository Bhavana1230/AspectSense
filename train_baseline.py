"""Train / re-train the TF-IDF + Logistic Regression baseline.  Usage: python train_baseline.py"""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from absa.sentiment import build_tfidf_pipeline, load_rows, train_tfidf

rows = load_rows()
X, y = [r[0] for r in rows], [r[1] for r in rows]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
pipe = build_tfidf_pipeline().fit(Xtr, ytr)
print("Hold-out split of the synthetic set (template overlap makes this optimistic):")
print(classification_report(yva, pipe.predict(Xva), digits=3))
train_tfidf()
print("saved models/tfidf.joblib (trained on all rows)")
