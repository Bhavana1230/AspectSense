"""Aspect-based sentiment pipeline.

text -> language -> sentences -> clauses -> aspect terms/cues -> sentiment
(lexicon window and/or neural clause score) -> per-aspect aggregation.
"""
import math
import time

from . import lexicon as L
from .preprocess import detect_language, sentences, tokenize

POS_T, NEG_T = 0.25, -0.25
# engine name -> (neural engine name or None, weight of neural score)
ENGINE_CFG = {
    "lexicon": (None, 0.0),
    "tfidf": ("tfidf", 1.0),
    "transformer": ("transformer", 1.0),
    "hybrid": ("transformer", 0.5),
}


def label_of(score: float) -> str:
    return "positive" if score >= POS_T else "negative" if score <= NEG_T else "neutral"


# ------------------------------------------------------------ lexicon scoring
def score_tokens(toks, lang):
    """Return sentiment items [{start,end,score,word}] found in a token list."""
    items, i, n = [], 0, len(toks)
    while i < n:
        hit = None
        for ln in (3, 2):
            if i + ln <= n:
                key = " ".join(toks[i:i + ln])
                if key in L.PHRASES:
                    hit = (L.PHRASES[key], i, i + ln - 1, key)
                    break
        if hit is None:
            v = L.SENT.get(toks[i])
            if v is not None:
                hit = (v, i, i, toks[i])
        if hit is None:
            i += 1
            continue
        score, s, e, word = hit
        mult = 1.0
        for j in range(max(0, s - 2), s):
            m = L.INTENS.get(toks[j])
            if m:
                mult = m
        neg = any(t in L.NEGATORS for t in toks[max(0, s - 3):s])
        if lang != "en" and any(t in L.NEGATORS for t in toks[e + 1:e + 3]):
            neg = True
        score *= mult
        if neg:
            score *= -0.75
        items.append({"start": s, "end": e, "score": score, "word": word, "negated": neg})
        i = e + 1
    return items


def has_sentiment(toks, lang) -> bool:
    return any(abs(it["score"]) >= 0.5 for it in score_tokens(toks, lang))


# ------------------------------------------------------------ clause splitting
def split_clauses(toks, lang):
    """Split a token list ([(tok,start,end)], commas included) into clauses."""
    words = [t[0] for t in toks]
    # 1. contrast markers
    cuts = []
    for i, w in enumerate(words):
        if w in L.CONTRAST:
            cuts.append(i)
        elif w in L.SOFT_CONTRAST and i > 0 and words[i - 1] in L.COPULA:
            cuts.append(i)
    segs, prev = [], 0
    for c in cuts:
        segs.append((prev, c))
        prev = c + 1
    segs.append((prev, len(toks)))
    # 2. conjunctions / commas, only when both sides carry sentiment
    out = []
    for a, b in segs:
        start = a
        for i in range(a, b):
            if words[i] in L.CONJ or words[i] in ",;":
                left = [w for w in words[start:i] if w not in ",;"]
                right = [w for w in words[i + 1:b] if w not in ",;"]
                if left and right and has_sentiment(left, lang) and has_sentiment(right, lang):
                    out.append((start, i))
                    start = i + 1
        out.append((start, b))
    clauses = []
    for a, b in out:
        sub = [t for t in toks[a:b] if t[0] not in ",;"]
        if sub:
            clauses.append(sub)
    return clauses


# ------------------------------------------------------------ aspect matching
def find_aspects(words):
    """Explicit aspect terms [(start,end,aspect)]; falls back to implicit cues."""
    n, hits, i = len(words), [], 0
    while i < n:
        matched = False
        for ln in (3, 2):
            if i + ln <= n:
                key = " ".join(words[i:i + ln])
                if key in L.MULTIWORD_TERMS:
                    hits.append((i, i + ln - 1, L.MULTIWORD_TERMS[key]))
                    i += ln
                    matched = True
                    break
        if matched:
            continue
        a = L.TERM_LEX.get(words[i])
        if a:
            hits.append((i, i, a))
        i += 1
    if hits:
        return hits, "term"
    cues = []
    for i, w in enumerate(words):
        a = L.CUE_LEX.get(w)
        if a:
            cues.append((i, i, a))
    return cues, "cue"


def _gap(item, hit):
    if item["end"] < hit[0]:
        return hit[0] - item["end"] - 1
    if item["start"] > hit[1]:
        return item["start"] - hit[1] - 1
    return 0


def lexicon_aspect_scores(words, lang, hits):
    """Assign every sentiment item to its nearest aspect hit -> {aspect: raw_sum}."""
    items = score_tokens(words, lang)
    aspects = sorted({h[2] for h in hits})
    raw = {a: 0.0 for a in aspects}
    evid = {a: False for a in aspects}
    multi = len(aspects) > 1
    for it in items:
        best = min(hits, key=lambda h: _gap(it, h))
        if multi and _gap(it, best) > 4:
            continue
        raw[best[2]] += it["score"]
        evid[best[2]] = True
    # coordinated aspects ("service aur delivery dono achhe"): share the nearest scored neighbour
    for h in hits:
        if not evid[h[2]]:
            donors = [g for g in hits if evid[g[2]] and g[2] != h[2] and
                      min(abs(g[0] - h[1]), abs(h[0] - g[1])) <= 4 and
                      any(w in L.CONJ for w in words[min(g[1], h[1]):max(g[0], h[0]) + 1])]
            if donors:
                d = min(donors, key=lambda g: min(abs(g[0] - h[1]), abs(h[0] - g[1])))
                raw[h[2]], evid[h[2]] = raw[d[2]], True
    return raw, evid, items


# ------------------------------------------------------------ main entry
def analyze(text: str, engine: str = "lexicon", neural=None) -> dict:
    t0 = time.perf_counter()
    text = (text or "").strip()
    lang = detect_language(text)
    lc = lang["code"] if lang["code"] == "en" else "x"
    neural_name, w_n = ENGINE_CFG.get(engine, ENGINE_CFG["lexicon"])
    if neural_name and neural is None:
        raise ValueError("neural engine required for '%s'" % engine)

    clause_objs, all_items = [], []
    for s, e in sentences(text):
        toks = tokenize(text[s:e])
        toks = [(w, a + s, b + s) for w, a, b in toks]
        for cl in split_clauses(toks, lc):
            words = [t[0] for t in cl]
            hits, kind = find_aspects(words)
            raw, evid, items = lexicon_aspect_scores(words, lc, hits) if hits else ({}, {}, score_tokens(words, lc))
            all_items.extend(items)
            clause_objs.append({
                "text": text[cl[0][1]:cl[-1][2]], "words": words, "hits": hits, "kind": kind,
                "raw": raw, "evid": evid,
                "lex_score": math.tanh(sum(i["score"] for i in items) / 2.0),
            })

    # neural scores: whole text + every clause that has aspects (one batch)
    n_overall, n_clause = 0.0, {}
    if neural_name:
        idx = [i for i, c in enumerate(clause_objs) if c["hits"]]
        scores = neural.score_many([text] + [clause_objs[i]["text"] for i in idx])
        n_overall = scores[0]
        n_clause = {i: s for i, s in zip(idx, scores[1:])}

    mentions = {}
    clauses_out = []
    for ci, c in enumerate(clause_objs):
        asp_here = sorted({h[2] for h in c["hits"]})
        for a in asp_here:
            lex = math.tanh(c["raw"][a] / 2.0)
            have_lex = c["evid"][a]
            if neural_name is None:
                if not have_lex:
                    continue
                sc = lex
            elif len(asp_here) == 1 or not have_lex:
                sc = (1 - w_n) * lex + w_n * n_clause[ci] if have_lex else n_clause[ci]
            else:
                sc = (1 - w_n) * lex + w_n * n_clause[ci] if w_n < 1 else n_clause[ci]
            mentions.setdefault(a, []).append({"clause": c["text"], "score": round(sc, 3)})
        clauses_out.append({
            "text": c["text"], "aspects": asp_here,
            "score": round(c["lex_score"] if neural_name is None else n_clause.get(ci, c["lex_score"]), 3),
        })

    aspects_out = []
    for a in L.ASPECT_LABELS:
        if a not in mentions:
            continue
        ms = mentions[a]
        mean = sum(m["score"] for m in ms) / len(ms)
        mixed = any(m["score"] >= POS_T for m in ms) and any(m["score"] <= NEG_T for m in ms)
        aspects_out.append({
            "aspect": a, "score": round(mean, 3),
            "label": "mixed" if mixed else label_of(mean), "mentions": ms,
        })

    lex_overall = math.tanh(sum(i["score"] for i in all_items) / 2.5)
    overall = (1 - w_n) * lex_overall + w_n * n_overall if neural_name else lex_overall
    has_pos = any(a["label"] == "positive" for a in aspects_out)
    has_neg = any(a["label"] == "negative" for a in aspects_out)
    mixed_flag = (has_pos and has_neg) or any(a["label"] == "mixed" for a in aspects_out)
    olabel = label_of(overall)
    if mixed_flag and abs(overall) < 0.5:
        olabel = "mixed"

    return {
        "text": text, "language": lang, "engine": engine,
        "overall": {"label": olabel, "score": round(overall, 3)},
        "aspects": aspects_out, "clauses": clauses_out,
        "trace": {
            "tokens": [t[0] for t in tokenize(text) if t[0] not in ",;"][:60],
            "lexicon_hits": [{"word": i["word"], "score": round(i["score"], 2), "negated": i["negated"]}
                             for i in all_items][:30],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    }
