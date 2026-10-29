"""Text normalisation, tokenisation, language detection and clause segmentation."""
import re
import unicodedata

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_APOS = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'"})
_DEV = re.compile(r"[\u0900-\u097F]")
TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097f\u20b9']+|[,;]", re.I)
SENT_RE = re.compile(r"[^.!?\u0964\u0965\n]+[.!?\u0964\u0965]*")

SLANG = {
    "gr8": "great", "gud": "good", "awsm": "awesome", "nyc": "nice", "pls": "please",
    "plz": "please", "thx": "thanks", "tnx": "thanks", "u": "you", "ur": "your",
    "r": "are", "bcz": "because", "luv": "love", "nt": "not", "dnt": "dont",
    "cant": "can't", "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
    "isnt": "isn't", "wasnt": "wasn't", "wont": "won't", "arent": "aren't",
    "nhi": "nahi", "nahin": "nahi", "bohot": "bahut", "bhot": "bahut", "bahot": "bahut",
}


def norm_token(t: str) -> str:
    """Lower-case, strip Devanagari nukta / zero-width chars, unify apostrophes."""
    t = unicodedata.normalize("NFD", t).replace("\u093c", "")
    t = unicodedata.normalize("NFC", t).translate(_ZW).translate(_APOS).lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # soooo -> soo
    return t.strip("'")


def is_dev(t: str) -> bool:
    return bool(_DEV.search(t))


def skeleton(t: str) -> str:
    """Spelling-variant key for romanised words: achha/accha/acha -> acha."""
    return re.sub(r"(.)\1+", r"\1", t)


_MATRAS = "ाििीुूेैोौंँः"


def stem_dev(t: str) -> str:
    """Strip one trailing vowel sign so inflected Devanagari forms collapse."""
    return t[:-1] if len(t) > 2 and t[-1] in _MATRAS else t


def tokenize(raw: str):
    """Return [(norm_token, start, end)] over the raw string (commas kept as tokens)."""
    out = []
    for m in TOKEN_RE.finditer(raw):
        w = m.group()
        if w in ",;":
            out.append((w, m.start(), m.end()))
            continue
        n = norm_token(w)
        n = SLANG.get(n, n)
        if n:
            out.append((n, m.start(), m.end()))
    return out


def sentences(raw: str):
    return [(m.start(), m.end()) for m in SENT_RE.finditer(raw)
            if m.group().strip(" .!?\u0964\u0965")]


# ---------------------------------------------------------------- language
_HI_DEV = set("है हैं नहीं नही बहुत और लेकिन था थी थे का की के हूँ हूं मैं में को से हुआ हुई भी यह कुछ सब मिला आई आया".split())
_MR_DEV = set("आहे नाही खूप होता होती होतं होते आणि पण मला काय आहेत झाली झाला आली आला गेले मिळाली सगळं काही जास्त".split())
_HINGLISH = set("""hai hain tha thi nahi bahut aur lekin magar achha accha acha achhi achi badhiya bakwas bekar
bekaar kharab mast ekdum kaafi kafi bilkul paisa paise mehnga sasta kya raha rahi aaya aayi aaye mera meri kuch sab
bhi ka ki ke ko se mein toh yeh ye hua hui thoda thodi barbaad ghatiya wala wali vasool turant der
khup aahe ahe aani hota hoti mala chhan chan jast gele vel var sagla kay nako""".split())


def detect_language(raw: str):
    toks = [t for t, _, _ in tokenize(raw) if t not in ",;"]
    dev_chars = len(_DEV.findall(raw))
    latin_chars = len(re.findall(r"[A-Za-z]", raw))
    if dev_chars and dev_chars >= latin_chars * 0.3:
        hi = sum(t in _HI_DEV for t in toks)
        mr = sum(t in _MR_DEV for t in toks)
        if mr > hi:
            return {"code": "mr", "name": "Marathi", "script": "Devanagari"}
        return {"code": "hi", "name": "Hindi", "script": "Devanagari"}
    hits = sum(t in _HINGLISH for t in toks)
    if toks and hits >= 1 and hits / len(toks) >= 0.12:
        return {"code": "hinglish", "name": "Hinglish", "script": "Latin"}
    return {"code": "en", "name": "English", "script": "Latin"}
