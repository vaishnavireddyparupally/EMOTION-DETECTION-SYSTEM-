# Speech-to-Text Prediction Using NLP

Final project (Innomatics Research Labs) — converts spoken/typed reviews into text and
predicts sentiment (**Positive / Negative / Neutral**) with a confidence score, following
the workflow:

```
Speech → Text → Preprocessing → Text Vectorization → ML Model → Prediction + Confidence Score
```

## Project structure

```
Speech_to_Text_NLP/
├── dataset.csv            # Review text + sentiment label (1,357 rows)
├── generate_dataset.py    # Script that built dataset.csv (template-based, reproducible)
├── model_building.ipynb   # Full step-by-step notebook (matches the project spec, 1–14)
├── train_pipeline.py      # Same pipeline as the notebook, as a plain .py script
├── app.py                 # Streamlit app: speech-to-text, prediction, confidence, gestures
├── best_model.pkl         # Saved best model (see model_report.txt)
├── vectorizer.pkl         # Saved fitted TF-IDF vectorizer
├── model_report.txt       # Accuracy / F1 for each candidate model
├── wordcloud.png           # Word cloud generated from the dataset
└── requirements.txt
```

## How it was built

- **Dataset**: 1,357 short review-style sentences, evenly split across Positive / Negative /
  Neutral, generated from templates + word banks (`generate_dataset.py`) so the whole
  pipeline is reproducible without needing an external dataset download.
- **Preprocessing**: lowercasing, URL/punctuation/number removal, stop-word removal
  (NLTK), lemmatization (WordNet).
- **Vectorization**: TF-IDF (unigrams + bigrams, top 3,000 features).
- **Models compared**: Logistic Regression and KNN, each tuned with `GridSearchCV`.
- **Result**: see `model_report.txt` for the exact accuracy/F1 obtained — the notebook
  does not assume any numbers, it prints the real evaluation output.
- **Saved artifacts**: best model → `best_model.pkl`, fitted vectorizer → `vectorizer.pkl`
  (the same vectorizer object is reused at prediction time, as the spec requires).

## Running it

```bash
pip install -r requirements.txt

# 1. (optional) regenerate the dataset
python generate_dataset.py

# 2. (optional) retrain from scratch — or just open model_building.ipynb
python train_pipeline.py

# 3. launch the app
streamlit run app.py
```

## App features (`app.py`)

- Dark-themed Streamlit UI with two tabs:
  - **🎤 Speech-to-Text Prediction** — record from the microphone (`SpeechRecognition` +
    Google's free recognizer) or upload a WAV file, or just type text. Shows the
    recognized text, predicted sentiment, and confidence score.
  - **✋ Gesture Detection (optional)** — the bonus MediaPipe + OpenCV feature from the
    spec. Uses the current MediaPipe **Tasks API** (`HandLandmarker`), since the older
    `mp.solutions` namespace was removed in recent MediaPipe versions. On first run it
    downloads the small hand-landmarker model bundle automatically.
- Microphone/webcam access requires running `streamlit run app.py` **locally** (a
  browser-based hosted sandbox can't reach your mic/camera) — the "type text" mode always
  works everywhere.

## Notes

- `dataset.csv` is synthetic (template-generated) so the project is fully self-contained
  and reproducible offline — swap in a real labeled dataset by keeping the same
  `text` / `target` column names and everything downstream (notebook + app) still works.
- Per the spec's note, the workflow can be adapted — extra features, a different
  vectorizer (Word2Vec/GloVe/etc.), or a different model can be dropped in without
  changing the overall structure.
