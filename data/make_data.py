"""Generate the synthetic clause-level training set (data/train.csv) and the
hand-written evaluation set (data/test_set.json).  Run:  python data/make_data.py"""
import csv
import json
import os
import random

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))

LANG = {
    "en": dict(
        subj=["the food", "the service", "delivery", "the battery", "the price", "the design", "the packaging",
              "the staff", "the app", "the quality", "the camera", "customer support", "the screen"],
        pos="good great excellent amazing lovely nice superb fantastic perfect wonderful impressive smooth".split(),
        neg="bad terrible awful poor horrible disappointing worst useless pathetic rubbish annoying".split(),
        neu="okay average fine alright normal so-so".split(),
        inten=["very", "really", "extremely", "so"],
        pat=["{s} was {a}", "{s} is {a}", "{i} {a} {s}", "{s} was {i} {a}", "{a} {s}"],
        negp="{s} was not {a}", neu_sent=["I ordered it yesterday", "the parcel came in a box", "I bought this last week",
                                          "it has a black cover", "I am using it for two days"]),
    "hinglish": dict(
        subj=["khana", "service", "delivery", "battery", "price", "design", "packing", "staff ka behaviour", "app",
              "quality", "camera", "paisa", "customer care"],
        pos="achha badhiya mast zabardast shandar kamaal sahi behtareen lajawab".split(),
        neg="bura bekar bakwas ghatiya kharab faltu wahiyat bekaar".split(),
        neu="theek average thik-thak normal chalta-hai".split(),
        inten=["bahut", "ekdum", "kaafi", "bilkul"],
        pat=["{s} {a} hai", "{s} {i} {a} tha", "{s} {a} thi", "{i} {a} {s}", "{s} {a}"],
        negp="{s} {a} nahi hai", neu_sent=["mera order kal aaya", "maine ye last week liya", "parcel box mein aaya",
                                           "do din se use kar raha hoon"]),
    "hi": dict(
        subj=["खाना", "सर्विस", "डिलीवरी", "बैटरी", "कीमत", "डिज़ाइन", "पैकिंग", "स्टाफ का व्यवहार", "ऐप",
              "क्वालिटी", "कैमरा", "ग्राहक सेवा"],
        pos="अच्छा बढ़िया शानदार बेहतरीन लाजवाब कमाल मस्त उम्दा".split(),
        neg="बुरा बेकार बकवास घटिया खराब फालतू वाहियात".split(),
        neu="ठीक-ठाक औसत सामान्य".split(),
        inten=["बहुत", "एकदम", "काफी", "बिल्कुल"],
        pat=["{s} {a} है", "{s} {i} {a} था", "{s} {a} थी", "{i} {a} {s}", "{s} {a}"],
        negp="{s} {a} नहीं है", neu_sent=["मैंने कल ऑर्डर किया", "पार्सल डिब्बे में आया", "दो दिन से इस्तेमाल कर रहा हूँ"]),
    "mr": dict(
        subj=["जेवण", "सेवा", "डिलिव्हरी", "बॅटरी", "किंमत", "डिझाइन", "पॅकिंग", "कर्मचाऱ्यांचे वागणे", "ॲप",
              "दर्जा", "कॅमेरा", "ग्राहक सेवा"],
        pos="छान उत्तम मस्त सुंदर झक्कास भारी चांगले".split(),
        neg="वाईट कचरा खराब बेकार फालतू सुमार".split(),
        neu="ठीक सामान्य".split(),
        inten=["खूप", "फारच", "अतिशय"],
        pat=["{s} {a} आहे", "{s} {i} {a} होतं", "{s} {a} होती", "{i} {a} {s}", "{s} {a}"],
        negp="{s} {a} नाही", neu_sent=["मी काल ऑर्डर केलं", "पार्सल बॉक्समध्ये आलं", "दोन दिवसांपासून वापरतोय"]),
}


def gen():
    rows = set()
    for lg, d in LANG.items():
        for s in d["subj"]:
            for lab, adjs in (("positive", d["pos"]), ("negative", d["neg"]), ("neutral", d["neu"])):
                for a in adjs:
                    for p in d["pat"]:
                        if "{i}" in p:
                            t = p.format(s=s, a=a, i=random.choice(d["inten"]))
                        else:
                            t = p.format(s=s, a=a)
                        rows.add((t, lab))
            for a in d["pos"]:                      # negated positive -> negative
                rows.add((d["negp"].format(s=s, a=a), "negative"))
            for a in d["neg"][:3]:                  # negated mild negative -> positive / neutral
                rows.add((d["negp"].format(s=s, a=a), "positive" if lg in ("en", "hinglish") else "neutral"))
        for t in d["neu_sent"]:
            rows.add((t, "neutral"))
    rows = sorted(rows)
    random.shuffle(rows)
    return rows


# ---------------------------------------------------------------- test set
T = []


def add(lang, text, overall, **aspects):
    T.append({"lang": lang, "text": text, "overall": overall, "aspects": aspects})


# English
add("en", "The food was delicious but the service was very slow.", "neutral", quality="positive", service="negative")
add("en", "Great battery life and a beautiful display, totally worth the price.", "positive", performance="positive", design="positive", price="positive")
add("en", "Delivery was late and the packaging was damaged. Very disappointed.", "negative", delivery="negative")
add("en", "The phone is okay, nothing special.", "neutral")
add("en", "Customer support was rude and unhelpful.", "negative", service="negative")
add("en", "Not bad for the price.", "positive", price="positive")
add("en", "Build quality is excellent but it is quite expensive.", "neutral", quality="positive", price="negative")
add("en", "The app keeps crashing and the interface looks ugly.", "negative", performance="negative", design="negative")
add("en", "Fast delivery and friendly staff, will order again!", "positive", delivery="positive", service="positive")
add("en", "I love the color and the size is perfect.", "positive", design="positive")
add("en", "The product arrived on time but stopped working after two days.", "negative", delivery="positive", performance="negative")
add("en", "Total waste of money. Never buying again.", "negative", price="negative")
# Hinglish (romanised Hindi / Marathi)
add("hinglish", "Khana bahut tasty tha lekin service ekdum bekar thi", "neutral", quality="positive", service="negative")
add("hinglish", "Delivery time pe aayi aur packaging bhi badhiya hai", "positive", delivery="positive")
add("hinglish", "Paisa vasool product hai, quality mast hai", "positive", price="positive", quality="positive")
add("hinglish", "Bahut mehnga hai aur quality bhi achhi nahi", "negative", price="negative", quality="negative")
add("hinglish", "Camera achha hai but battery bakwas hai", "neutral", performance="mixed")
add("hinglish", "Staff ka behaviour bahut accha tha", "positive", service="positive")
add("hinglish", "Delivery boy bahut late aaya, bilkul bekar experience", "negative", delivery="negative")
add("hinglish", "Design mast hai par speed thodi slow hai", "neutral", design="positive", performance="negative")
add("hinglish", "Ye phone ekdum ghatiya nikla, paisa barbaad", "negative", price="negative")
add("hinglish", "Sab kuch theek tha, kuch khaas nahi", "neutral")
add("hinglish", "Customer care ne turant help ki, great service!", "positive", service="positive")
add("hinglish", "Packaging kharab thi aur delivery bhi der se hui", "negative", delivery="negative")
add("hinglish", "Jevan khup chhan hota pan service khup slow hoti", "neutral", quality="positive", service="negative")
add("hinglish", "Delivery vel var aali ani packing chhan hoti", "positive", delivery="positive")
add("hinglish", "Kimmat khup jast aahe, paise vaya gele", "negative", price="negative")
# Hindi
add("hi", "खाना बहुत स्वादिष्ट था लेकिन सर्विस बहुत धीमी थी", "neutral", quality="positive", service="negative")
add("hi", "डिलीवरी समय पर हुई और पैकिंग भी बढ़िया थी", "positive", delivery="positive")
add("hi", "कीमत बहुत महंगी है, पैसा वसूल नहीं हुआ", "negative", price="negative")
add("hi", "कैमरा बेहतरीन है पर बैटरी बेकार है", "neutral", performance="mixed")
add("hi", "स्टाफ का व्यवहार बहुत अच्छा था", "positive", service="positive")
add("hi", "प्रोडक्ट घटिया निकला और डिलीवरी भी देर से आई", "negative", quality="negative", delivery="negative")
add("hi", "डिज़ाइन शानदार है और रंग भी सुंदर है", "positive", design="positive")
add("hi", "ऐप बार-बार क्रैश होता है, बहुत परेशान हूँ", "negative", performance="negative")
add("hi", "सब ठीक है, कोई खास बात नहीं", "neutral")
add("hi", "बहुत सस्ता और अच्छा सामान", "positive", price="positive", quality="positive")
add("hi", "ग्राहक सेवा ने तुरंत मदद की, शुक्रिया", "positive", service="positive")
add("hi", "पैकिंग खराब थी और सामान टूटा हुआ मिला", "negative", delivery="negative", quality="negative")
# Marathi
add("mr", "जेवण खूप छान होतं पण सेवा खूप संथ होती", "neutral", quality="positive", service="negative")
add("mr", "डिलिव्हरी वेळेवर आली आणि पॅकिंग छान होतं", "positive", delivery="positive")
add("mr", "किंमत खूप जास्त आहे, पैसे वाया गेले", "negative", price="negative")
add("mr", "कॅमेरा उत्तम आहे पण बॅटरी वाईट आहे", "neutral", performance="mixed")
add("mr", "कर्मचाऱ्यांचे वागणे खूप चांगले होते", "positive", service="positive")
add("mr", "वस्तू खराब निघाली आणि डिलिव्हरी उशिरा आली", "negative", quality="negative", delivery="negative")
add("mr", "डिझाइन सुंदर आहे आणि रंग पण छान आहे", "positive", design="positive")
add("mr", "ॲप सारखा क्रॅश होतो, खूप त्रास झाला", "negative", performance="negative")
add("mr", "सगळं ठीक आहे, खास काही नाही", "neutral")
add("mr", "स्वस्त आणि चांगली वस्तू", "positive", price="positive", quality="positive")
add("mr", "ग्राहक सेवा खूप मदतगार होती", "positive", service="positive")
add("mr", "पॅकिंग वाईट होतं आणि वस्तू तुटलेली मिळाली", "negative", delivery="negative", quality="negative")


# ------------------------------------------------- challenge set (unseen phrasing)
H = []


def hard(lang, text, overall, **aspects):
    H.append({"lang": lang, "text": text, "overall": overall, "aspects": aspects})


hard("en", "The pasta was bland and overcooked, though the waiter was very courteous.", "neutral", quality="negative", service="positive")
hard("en", "Shipping took forever but the box arrived intact.", "neutral", delivery="mixed")
hard("en", "Honestly the screen is gorgeous and the speakers are crisp.", "positive", design="positive", performance="positive")
hard("en", "I wouldn't recommend this to anyone.", "negative")
hard("en", "Oh great, another app update that broke everything.", "negative", performance="negative")
hard("en", "Service was fine, nothing to complain about.", "neutral", service="neutral")
hard("en", "Overpriced for what you get, but the staff made up for it with their warmth.", "neutral", price="negative", service="positive")
hard("en", "The jacket feels premium and the stitching is solid.", "positive", quality="positive")
hard("hinglish", "Pizza ekdum fresh tha par delivery boy ka attitude kharab tha", "neutral", quality="positive", service="negative")
hard("hinglish", "Mobile hang bahut karta hai, battery bhi jaldi khatam ho jaati hai", "negative", performance="negative")
hard("hinglish", "Value for money hai, bilkul recommend karunga", "positive", price="positive")
hard("hinglish", "Itna paisa dene ke baad bhi ye haal, sharam aani chahiye", "negative")
hard("hinglish", "Seller ne galat product bheja aur refund bhi nahi diya", "negative", service="negative")
hard("hinglish", "Thoda mehnga zaroor hai lekin kaam ekdum mast karta hai", "neutral", price="negative")
hard("hi", "सर्विस तो ठीक थी पर खाना ठंडा और बेस्वाद था", "negative", service="neutral", quality="negative")
hard("hi", "बैटरी शानदार चलती है और कैमरा भी लाजवाब है", "positive", performance="positive")
hard("hi", "डिलीवरी में चार दिन लग गए, बहुत निराशा हुई", "negative", delivery="negative")
hard("hi", "कपड़े की क्वालिटी उम्मीद से बेहतर निकली", "positive", quality="positive")
hard("hi", "स्टाफ ने कोई मदद नहीं की", "negative", service="negative")
hard("mr", "जेवण थंड आणि बेचव होतं", "negative", quality="negative")
hard("mr", "ॲपचा वेग खूप चांगला आहे", "positive", performance="positive")
hard("mr", "किंमत थोडी जास्त आहे पण दर्जा उत्कृष्ट आहे", "neutral", price="negative", quality="positive")
hard("mr", "बॅटरी लवकर संपते, अजिबात आवडलं नाही", "negative", performance="negative")
hard("mr", "सेवा ठीकठाक होती", "neutral", service="neutral")

if __name__ == "__main__":
    rows = gen()
    with open(os.path.join(HERE, "train.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    with open(os.path.join(HERE, "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(T, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "test_hard.json"), "w", encoding="utf-8") as f:
        json.dump(H, f, ensure_ascii=False, indent=1)
    print("train rows:", len(rows), "| test items:", len(T), "| challenge items:", len(H))
