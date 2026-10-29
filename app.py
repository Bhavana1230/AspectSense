"""AspectSense - Flask REST API + web UI.   Run:  python app.py   ->  http://127.0.0.1:5000"""
import csv
import io
import os

from flask import Flask, jsonify, render_template, request

from absa import store
from absa.aspects import ENGINE_CFG, analyze
from absa.sentiment import available_engines, get_neural

app = Flask(__name__)
MAX_CHARS, MAX_ROWS = 2000, 500
DEFAULT_ENGINE = os.environ.get("ENGINE", "lexicon")


def run_one(text, engine, log=True):
    avail = available_engines()
    note = None
    if engine not in ENGINE_CFG:
        engine = "lexicon"
    if not avail.get(engine):
        note, engine = "transformer libs not installed - fell back to lexicon", "lexicon"
    nname = ENGINE_CFG[engine][0]
    try:
        neural = get_neural(nname) if nname else None
    except Exception as exc:                      # model download / load failure
        note, engine, neural = "could not load '%s' (%s) - fell back to lexicon" % (nname, type(exc).__name__), "lexicon", None
    res = analyze(text[:MAX_CHARS], engine, neural)
    if note:
        res["note"] = note
    if log:
        store.log_result(res)
    return res


@app.get("/")
def index():
    return render_template("index.html", default_engine=DEFAULT_ENGINE)


@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.get("/api/engines")
def engines():
    return jsonify({"engines": available_engines(), "default": DEFAULT_ENGINE})


@app.post("/api/analyze")
def api_analyze():
    d = request.get_json(silent=True) or {}
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    return jsonify(run_one(text, d.get("engine", DEFAULT_ENGINE)))


@app.post("/api/batch")
def api_batch():
    engine = request.form.get("engine") or (request.get_json(silent=True) or {}).get("engine") or DEFAULT_ENGINE
    texts = []
    if "file" in request.files:
        raw = request.files["file"].read().decode("utf-8-sig", errors="replace")
        rd = csv.reader(io.StringIO(raw))
        rows = list(rd)
        if rows:
            col = rows[0].index("text") if "text" in rows[0] else 0
            start = 1 if "text" in rows[0] else 0
            texts = [r[col] for r in rows[start:] if len(r) > col and r[col].strip()]
    else:
        texts = [t for t in ((request.get_json(silent=True) or {}).get("texts") or []) if str(t).strip()]
    if not texts:
        return jsonify({"error": "no texts found (CSV needs a 'text' column or one review per line)"}), 400
    out = [run_one(str(t), engine) for t in texts[:MAX_ROWS]]
    summary = {"count": len(out), "overall": {}, "aspects": {}}
    for r in out:
        summary["overall"][r["overall"]["label"]] = summary["overall"].get(r["overall"]["label"], 0) + 1
        for a in r["aspects"]:
            summary["aspects"].setdefault(a["aspect"], {}).setdefault(a["label"], 0)
            summary["aspects"][a["aspect"]][a["label"]] += 1
    rows = [{"text": r["text"], "language": r["language"]["name"], "overall": r["overall"]["label"],
             "aspects": "; ".join("%s=%s" % (a["aspect"], a["label"]) for a in r["aspects"])} for r in out]
    return jsonify({"summary": summary, "rows": rows, "truncated": len(texts) > MAX_ROWS})


@app.get("/api/stats")
def api_stats():
    return jsonify(store.stats())


@app.post("/api/clear")
def api_clear():
    store.clear_log()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
