"""Evaluate engines on data/test_set.json.
Usage:  python evaluate.py                    (lexicon + tfidf [+ transformer/hybrid if installed])
        python evaluate.py --engines lexicon,hybrid
"""
import argparse
import json
import os
import time

from sklearn.metrics import f1_score

from absa.aspects import ENGINE_CFG, analyze
from absa.preprocess import detect_language
from absa.sentiment import available_engines, get_neural

HERE = os.path.dirname(os.path.abspath(__file__))


def run(engine, items):
    neural = get_neural(ENGINE_CFG[engine][0]) if ENGINE_CFG[engine][0] else None
    yt, yp, per_lang = [], [], {}
    tp = fp = fn = 0
    asp_ok = asp_det = 0
    t0 = time.perf_counter()
    for it in items:
        r = analyze(it["text"], engine, neural)
        pred = "neutral" if r["overall"]["label"] == "mixed" else r["overall"]["label"]
        yt.append(it["overall"]); yp.append(pred)
        per_lang.setdefault(it["lang"], []).append(pred == it["overall"])
        gold = it["aspects"]; got = {a["aspect"]: a["label"] for a in r["aspects"]}
        tp += len(set(gold) & set(got)); fp += len(set(got) - set(gold)); fn += len(set(gold) - set(got))
        for a in set(gold) & set(got):
            asp_det += 1; asp_ok += gold[a] == got[a]
    dt = (time.perf_counter() - t0) / len(items) * 1000
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return {
        "overall_acc": sum(a == b for a, b in zip(yt, yp)) / len(yt),
        "overall_macro_f1": f1_score(yt, yp, average="macro"),
        "aspect_P": P, "aspect_R": R, "aspect_F1": 2 * P * R / (P + R) if P + R else 0,
        "aspect_sentiment_acc": asp_ok / asp_det if asp_det else 0,
        "per_lang_acc": {k: sum(v) / len(v) for k, v in per_lang.items()},
        "ms_per_review": dt,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="lexicon,tfidf,transformer,hybrid")
    a = ap.parse_args()
    avail = available_engines()
    out = {}
    for setname, fname in (("main", "test_set.json"), ("challenge", "test_hard.json")):
        items = json.load(open(os.path.join(HERE, "data", fname), encoding="utf-8"))
        lang_acc = sum(detect_language(i["text"])["code"] == i["lang"] for i in items) / len(items)
        out[setname] = {"n": len(items), "language_detection_acc": lang_acc, "engines": {}}
        print("\n== %s set: n=%d  language-detection acc=%.3f" % (setname, len(items), lang_acc))
        print("%-12s %7s %7s %7s %7s %7s %8s" % ("engine", "acc", "mF1", "asp-P", "asp-R", "asp-F1", "asp-sent"))
        for e in a.engines.split(","):
            if not avail.get(e):
                print("%-12s skipped (install torch + transformers)" % e); continue
            m = run(e, items); out[setname]["engines"][e] = m
            print("%-12s %7.3f %7.3f %7.3f %7.3f %7.3f %8.3f   %s  %.1f ms" % (
                e, m["overall_acc"], m["overall_macro_f1"], m["aspect_P"], m["aspect_R"], m["aspect_F1"],
                m["aspect_sentiment_acc"], {k: round(v, 2) for k, v in m["per_lang_acc"].items()}, m["ms_per_review"]))
    json.dump(out, open(os.path.join(HERE, "data", "metrics.json"), "w"), indent=1)
