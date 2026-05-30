import os
import re
import pickle
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------- NLTK ----------------
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
stemmer = PorterStemmer()

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

MODEL = None
VECTORIZER = None

# ---------------- LOAD MODEL ----------------
def load_artifacts():
    global MODEL, VECTORIZER

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "model.pkl")
    vec_path   = os.path.join(base, "vectorizer.pkl")

    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)

    with open(vec_path, "rb") as f:
        VECTORIZER = pickle.load(f)

    log.info("✅ Model & vectorizer loaded")

# ---------------- PREPROCESS ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def preprocess(text):
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def full_preprocess(text):
    return preprocess(clean_text(text))

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "Empty text"}), 400

        processed = full_preprocess(text)

        if not processed:
            return jsonify({
                "sentiment": "Neutral",
                "confidence": 0.5
            })

        vec = VECTORIZER.transform([processed])
        pred = MODEL.predict(vec)[0]
        proba = MODEL.predict_proba(vec)[0]

        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = float(max(proba))

        return jsonify({
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    log.info("Starting server...")
    load_artifacts()
    app.run(host="0.0.0.0", port=5000)