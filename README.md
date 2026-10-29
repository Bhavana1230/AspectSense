# AspectSense — Aspect-Based + Multilingual + Transformer Sentiment Analysis

Review likho (English / Hindi / Marathi / Hinglish) → har **aspect** (quality, price, service,
delivery, design, performance) ka alag sentiment + overall sentiment.

## Run (5 commands)
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # open http://127.0.0.1:5000
python -m pytest -q tests                              # optional: unit tests
python evaluate.py --engines lexicon,tfidf             # metrics on bundled test sets
```
Pages: `/` analyzer · `/dashboard` charts + CSV batch upload.

## Engines
| engine | needs | notes |
|---|---|---|
| `lexicon` | nothing | default; instant; explainable (trace panel shows every hit) |
| `tfidf` | scikit-learn | TF-IDF (word+char n-gram) + Logistic Regression, trained on `data/train.csv` (auto-trains on first use) |
| `transformer` | `pip install torch transformers` | multilingual XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`), ~1.1 GB download on first use, then offline |
| `hybrid` | same as transformer | 0.5·XLM-R clause score + 0.5·lexicon aspect-window score (best for romanised Hinglish) |

`ENGINE=hybrid python app.py` sets the default. `ABSA_MODEL=<hf-model-id>` swaps the transformer
(any 3-class sentiment model with negative/neutral/positive labels works).

## API
```bash
curl -X POST localhost:5000/api/analyze -H "Content-Type: application/json" \
     -d '{"text":"Khana tasty tha lekin service bekar thi","engine":"lexicon"}'
curl -F file=@reviews.csv -F engine=lexicon localhost:5000/api/batch     # CSV with a "text" column
curl localhost:5000/api/stats
```

## Layout
```
app.py                Flask REST API + pages
absa/preprocess.py    normalisation, tokenizer, language detection
absa/lexicon.py       sentiment + aspect lexicons (en/hi/mr/hinglish)
absa/aspects.py       clause split → aspect find → scoring → aggregation
absa/sentiment.py     TF-IDF and XLM-R engines
absa/store.py         JSONL log + dashboard stats
data/make_data.py     regenerates train.csv, test_set.json, test_hard.json
evaluate.py           accuracy / macro-F1 / aspect P-R-F1
train_baseline.py     retrain TF-IDF model
```

## Extend
* New aspect → add entry in `ASPECTS` (`absa/lexicon.py`) with `terms` and `cues`.
* New words → add to `_SENT_SPEC` with a score (−2.5 … +2).
* Real data → replace `data/train.csv` (columns `text,label` with positive/neutral/negative), run `python train_baseline.py`.

## Honest limits
Lexicon + test sets are small and hand-written by the authors; numbers show the pipeline works, not state-of-the-art accuracy.
Sarcasm, unseen vocabulary and implicit aspects are the main failure modes (see `data/test_hard.json`).
