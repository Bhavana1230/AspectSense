"""Hand-built multilingual lexicons (English / Hindi / Marathi / Hinglish).

All keys are passed through ``norm_token`` so Devanagari nukta forms and
romanised spelling variants match.  Lookup falls back to a spelling-skeleton
(romanised) or a trailing-matra stem (Devanagari).
"""
from .preprocess import is_dev, norm_token, skeleton, stem_dev

# ----------------------------------------------------------- sentiment words
_SENT_SPEC = [
    # ---- positive
    (2.0, "excellent amazing awesome perfect fantastic superb wonderful outstanding delicious brilliant flawless "
          "zabardast jabardast shandar shaandaar behtareen behtarin lajawab kamaal kamal jhakaas zakkas swadisht "
          "जबरदस्त शानदार बेहतरीन लाजवाब कमाल स्वादिष्ट झक्कास उत्कृष्ट उम्दा उत्तम uttam"),
    (1.5, "great love loved loves lovely beautiful impressive happy satisfied stylish tasty gorgeous "
          "badhiya badiya badhia mast sundar khush mazedar mazedaar bhari behtar "
          "बढिया मस्त सुंदर खुश भारी बेहतर सुन्दर समाधानी"),
    (1.2, "good nice friendly helpful premium comfortable reliable smooth sturdy durable responsive polite prompt "
          "recommend recommended pleasant fresh crisp clean sleek fast quick speedy snappy tidy worth "
          "achha accha acha achhi acchi achi achhe acche ache pasand maza chhan chan chhaan changla changli chhangla "
          "chaangla awadla avadla awadli madadgar wegvan "
          "अच्छा अच्छी अच्छे पसंद मजा छान चांगला चांगली चांगले चांगलं चांगल्या आवडला आवडली आवडलं आवडले "
          "मदतगार टिकाऊ वेळेवर velevar"),
    (1.0, "thanks thank shukriya dhanyavad धन्यवाद शुक्रिया आभार turant तुरंत tez तेज वेगवान"),
    (0.8, "easy decent affordable help madad मदद sahi सही jaldi जल्दी लवकर sasta sasti saste सस्ता सस्ती सस्ते "
          "swasta स्वस्त"),
    (0.6, "cheap"),
    (0.2, "ok okay fine theek thik ठीक"),
    # ---- negative
    (-2.5, "worst scam fraud dhokha dhoka धोखा धोका"),
    (-2.2, "bakwas bakwaas ghatiya wahiyat vahiyat बकवास घटिया वाहियात निकृष्ट"),
    (-2.0, "terrible awful horrible useless disgusting pathetic defective fake garbage trash waste hate hated "
           "nakli kachra barbaad barbad नकली कचरा बर्बाद"),
    (-1.8, "bekar bekaar kharab kharaab faltu toota tuta tooti tuti tute "
           "बेकार खराब फालतू टूटा टूटी टूटे टूटलेली तुटला तुटली तुटलेली तुटलेले तुटलेला vaya वाया"),
    (-1.5, "bad poor ugly disappointed disappointing disappointment broken damaged rude unhelpful flimsy dirty stale "
           "laggy buggy crash crashes crashing crashed hang hangs slow late delayed overpriced unresponsive annoying "
           "frustrating worse bura buri bure pareshan paresan gussa ghussa ganda gandi nirash vait vaait ghan ghaan "
           "trass "
           "बुरा बुरी बुरे परेशान गुस्सा गंदा गंदी निराश निराशा वाईट घाण त्रास क्रैश क्रॅश हैंग लैग "),
    (-1.2, "expensive costly noisy uncomfortable bland cold lag lags heating overheats overheating drains drain "
           "mediocre unreliable mehnga mehngi mehenga mahanga dheema dhima sust der ushir ushira ushiri mahag mahaag "
           "dikkat shikayat "
           "महंगा महंगी महंगे धीमा धीमी धीमे सुस्त देर दिक्कत शिकायत महाग उशीर उशिरा उशिर संथ"),
    (-1.0, "issue issues problem problems complaint sumar सुमार"),
    (-0.3, "meh"),
]

_PHRASE_SPEC = [
    (2.0, "paisa vasool|paise vasool|paisa wasool|पैसा वसूल|पैसे वसूल|top notch|must buy"),
    (1.8, "value for money"),
    (1.2, "on time|samay par|समय पर|vel var|time pe|time par|वक्त पर"),
    (-2.2, "waste of money|stopped working|not working|doesn't work|does not work|didn't work|pura kharab"),
    (-0.4, "nothing special|khaas nahi|khas nahi|खास बात नहीं|खास काही नाही"),
]

_INTENS_SPEC = [
    (1.8, "extremely behad बेहद incredibly"),
    (1.7, "incredibly"),
    (1.6, "absolutely utterly ekdum एकदम atishay अतिशय"),
    (1.5, "very super bahut बहुत khup khoop खूप farach फारच"),
    (1.4, "totally highly bilkul बिल्कुल"),
    (1.3, "really so too truly kaafi kafi काफी"),
    (1.2, "quite pretty"),
    (0.8, "fairly"),
    (0.7, "somewhat bit little kinda thoda thodi थोडा थोड़ा थोड़ी थोडी"),
    (0.6, "slightly"),
]

NEGATORS = set(norm_token(w) for w in """not no never nor cannot can't don't doesn't didn't isn't wasn't aren't won't
wouldn't couldn't shouldn't hardly barely without nahi mat nako नहीं नही न मत नाही नाहीत नको""".split())

CONTRAST = set(norm_token(w) for w in """but however although though whereas while lekin lakin magar parantu kintu
लेकिन मगर परंतु किंतु""".split())
# 'par' / 'pan' are 'but' only after a copula (hai par ..); otherwise 'on' / 'also'
SOFT_CONTRAST = set(norm_token(w) for w in "par pan पर पण".split())
COPULA = set(norm_token(w) for w in """hai hain tha thi the h aahe ahe hota hoti hote nahi
है हैं था थी थे आहे आहेत होता होती होतं होते नाही नहीं""".split())
CONJ = set(norm_token(w) for w in "and aur plus tatha ani aani और तथा आणि आणखी &".split())

# ----------------------------------------------------------------- aspects
# 'terms' name the aspect explicitly; 'cues' imply it (only used if no term is present)
ASPECTS = {
    "quality": {
        "terms": "quality build material taste food meal dish product item stuff flavor flavour ingredients fabric "
                 "texture durability kwality gunvatta khana khaana swad maal saman samaan cheez kapda jevan vastu "
                 "क्वालिटी गुणवत्ता खाना स्वाद माल सामान प्रोडक्ट चीज कपड़ा मटेरियल जेवण वस्तू चव दर्जा",
        "cues": "delicious tasty bland stale fresh flimsy sturdy durable fake defective nakli swadisht "
                "स्वादिष्ट टिकाऊ नकली",
    },
    "price": {
        "terms": "price pricing cost value money rate budget charges fee fees paisa paise daam dam kimat kimmat keemat "
                 "rupees rs ₹ कीमत दाम पैसा पैसे किंमत रुपये मूल्य भाव",
        "cues": "expensive cheap costly overpriced affordable pricey worth mehnga mehngi sasta sasti saste mahag "
                "महंगा महंगी महंगे सस्ता सस्ती सस्ते महाग स्वस्त",
    },
    "service": {
        "terms": "service staff support helpline behaviour behavior manager waiter agent executive representative "
                 "seller team attitude seva vyavhar vyavahar karmchari "
                 "सर्विस सेवा स्टाफ व्यवहार कर्मचारी कर्मचाऱ्यांचे कर्मचाऱ्यांचा वागणे वागणूक सपोर्ट मैनेजर विक्रेता",
        "cues": "rude polite unhelpful friendly courteous madadgar मदतगार",
    },
    "delivery": {
        "terms": "delivery shipping shipment shipped courier package packaging packing parcel dispatch arrived "
                 "arrival box packed deliver delivered "
                 "डिलीवरी डिलिवरी डिलिव्हरी पैकिंग पैकेजिंग पॅकिंग कुरियर पार्सल पैकेट डिब्बा पैकेज",
        "cues": "late delayed lateness ushir ushira उशिरा उशीर",
    },
    "design": {
        "terms": "design look looks color colour size display screen style appearance ui interface ux layout theme "
                 "fit shape rang dizain डिजाइन डिझाइन रंग साइज डिस्प्ले स्क्रीन लुक इंटरफेस दिसणे",
        "cues": "beautiful ugly stylish sleek gorgeous sundar सुंदर",
    },
    "performance": {
        "terms": "performance speed battery camera app application software sound audio processor charging charge "
                 "heating signal network connectivity working wifi bluetooth mic "
                 "परफॉर्मेंस स्पीड बैटरी कैमरा ऐप ॲप ऐप्लिकेशन साउंड चार्जिंग नेटवर्क कामगिरी गती कॅमेरा बॅटरी",
        "cues": "slow fast laggy lag crash crashes crashing crashed hang hangs buggy glitch smooth responsive "
                "heating overheats overheating drains drain क्रैश क्रॅश लैग धीमा धीमी धीमे संथ",
    },
}

ASPECT_LABELS = list(ASPECTS)


# ------------------------------------------------------------ lookup helpers
class Lex:
    """Exact -> skeleton (romanised) -> trailing-matra stem (Devanagari) lookup."""

    def __init__(self, table):
        self.t = table
        self.sk = {skeleton(k): v for k, v in table.items() if len(k) >= 4 and not is_dev(k)}
        self.st = {stem_dev(k): v for k, v in table.items() if is_dev(k) and len(k) >= 3}

    def get(self, tok):
        v = self.t.get(tok)
        if v is not None:
            return v
        if is_dev(tok):
            return self.st.get(stem_dev(tok))
        if len(tok) > 3 and tok.endswith("s") and tok[:-1] in self.t:
            return self.t[tok[:-1]]
        if len(tok) >= 4:
            return self.sk.get(skeleton(tok))
        return None


def _table(spec):
    d = {}
    for score, words in spec:
        for w in words.split():
            d[norm_token(w)] = score
    return d


SENT = Lex(_table(_SENT_SPEC))
INTENS = _table(_INTENS_SPEC)
PHRASES = {}
for _s, _p in _PHRASE_SPEC:
    for _ph in _p.split("|"):
        PHRASES[" ".join(norm_token(w) for w in _ph.split())] = _s


def _aspect_table(key):
    multi, single = {}, {}
    for asp, d in ASPECTS.items():
        for w in d[key].split():
            single[norm_token(w)] = asp
    # two-word aspect terms
    return multi, single


_ASP_TERMS = _aspect_table("terms")[1]
_ASP_CUES = _aspect_table("cues")[1]
MULTIWORD_TERMS = {
    "customer care": "service", "customer service": "service", "customer support": "service",
    "battery life": "performance", "ग्राहक सेवा": "service", "कस्टमर केयर": "service",
    "ग्राहक सर्विस": "service",
}
TERM_LEX = Lex(_ASP_TERMS)
CUE_LEX = Lex(_ASP_CUES)
