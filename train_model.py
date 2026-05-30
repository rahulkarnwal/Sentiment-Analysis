import pandas as pd
import re
import pickle
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------- NLTK ----------------
nltk.download("stopwords", quiet=True)

# Keep important words like "not"
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
stemmer = PorterStemmer()

# ---------------- CONFIG ----------------
COLUMNS = ["target", "id", "date", "flag", "user", "text"]
LABEL_MAP = {0: 0, 4: 1}

# ---------------- LOAD ----------------
def load_dataset(path):
    df = pd.read_csv(path, names=COLUMNS, encoding="latin-1", header=None)
    df["sentiment"] = df["target"].map(LABEL_MAP)
    df = df[["text", "sentiment"]].dropna()
    return df

# ---------------- CLEAN ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ---------------- PREPROCESS ----------------
def preprocess(text):
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

# ---------------- MAIN ----------------
def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(BASE, "..", "data", "sentiment140.csv")
    OUT  = os.path.join(BASE, "..", "backend")

    SAMPLE_SIZE = 200000

    log.info("Loading dataset...")
    df = load_dataset(DATA)

    # Balanced sampling
    log.info("Sampling...")
    df_pos = df[df["sentiment"] == 1].sample(n=SAMPLE_SIZE//2, random_state=42)
    df_neg = df[df["sentiment"] == 0].sample(n=SAMPLE_SIZE//2, random_state=42)
    df = pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True)

    # Clean + preprocess
    log.info("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(preprocess)
    df = df[df["text"] != ""]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    # Vectorize
    log.info("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test  = vectorizer.transform(X_test)

    # Train model (better than RandomForest for text)
    log.info("Training model...")
    model = LogisticRegression(max_iter=500, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(OUT, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    log.info("Model + vectorizer saved!")

if __name__ == "__main__":
    main()