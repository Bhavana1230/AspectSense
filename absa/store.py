"""Append-only JSONL log + aggregate statistics for the dashboard."""
import json
import os
import threading
import time
from collections import Counter

from .lexicon import ASPECT_LABELS

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "log.jsonl")
_lock = threading.Lock()


def log_result(r: dict):
    rec = {"ts": int(time.time()), "text": r["text"][:300], "lang": r["language"]["name"], "engine": r["engine"],
           "overall": r["overall"]["label"], "score": r["overall"]["score"],
           "aspects": {a["aspect"]: a["label"] for a in r["aspects"]}}
    with _lock, open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_log():
    if not os.path.exists(LOG):
        return []
    with _lock, open(LOG, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def clear_log():
    with _lock:
        open(LOG, "w").close()


def stats():
    rows = read_log()
    matrix = {a: Counter() for a in ASPECT_LABELS}
    for r in rows:
        for a, lab in r["aspects"].items():
            matrix[a][lab] += 1
    return {
        "total": len(rows),
        "overall": dict(Counter(r["overall"] for r in rows)),
        "languages": dict(Counter(r["lang"] for r in rows)),
        "engines": dict(Counter(r["engine"] for r in rows)),
        "aspect_matrix": {a: dict(c) for a, c in matrix.items()},
        "avg_score": round(sum(r["score"] for r in rows) / len(rows), 3) if rows else 0,
        "recent": rows[-12:][::-1],
    }
