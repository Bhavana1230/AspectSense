import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from absa.aspects import analyze
from absa.preprocess import detect_language


def asp(text):
    return {a["aspect"]: a["label"] for a in analyze(text)["aspects"]}


def test_language_detection():
    assert detect_language("The service was great")["code"] == "en"
    assert detect_language("Khana bahut tasty tha lekin service bekar thi")["code"] == "hinglish"
    assert detect_language("खाना बहुत स्वादिष्ट था")["code"] == "hi"
    assert detect_language("जेवण खूप छान होतं")["code"] == "mr"


def test_contrast_gives_opposite_aspects():
    assert asp("The food was delicious but the service was very slow.") == {"quality": "positive", "service": "negative"}


def test_negation_flips():
    assert asp("Not bad for the price.") == {"price": "positive"}
    assert asp("Bahut mehnga hai aur quality bhi achhi nahi") == {"price": "negative", "quality": "negative"}


def test_hindi_and_marathi():
    assert asp("डिलीवरी समय पर हुई और पैकिंग भी बढ़िया थी") == {"delivery": "positive"}
    assert asp("जेवण खूप छान होतं पण सेवा खूप संथ होती") == {"quality": "positive", "service": "negative"}


def test_same_aspect_conflict_is_mixed():
    assert asp("Camera achha hai but battery bakwas hai") == {"performance": "mixed"}


def test_overall_mixed_flag():
    assert analyze("The food was delicious but the service was very slow.")["overall"]["label"] == "mixed"


def test_main_set_regression():
    items = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "test_set.json"), encoding="utf-8"))
    ok = sum(("neutral" if analyze(i["text"])["overall"]["label"] == "mixed" else analyze(i["text"])["overall"]["label"]) == i["overall"] for i in items)
    assert ok / len(items) >= 0.9
