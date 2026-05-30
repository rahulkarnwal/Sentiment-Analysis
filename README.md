# SentiScope — Sentiment Analysis Web Application

> A production-ready sentiment analysis system powered by a Logistic Regression model trained on 1.6 million tweets from the Sentiment140 dataset.

---

## 🏗️ Project Structure

```
sentiment-analysis-app/
│
├── backend/
│   ├── app.py              # Flask REST API
│   ├── model.pkl           # Trained model (generated)
│   └── vectorizer.pkl      # TF-IDF vectorizer (generated)
│
├── frontend/
│   ├── index.html          # Main UI page
│   ├── style.css           # Dark-theme styling
│   └── script.js           # API integration + UX logic
│
├── model/
│   └── train_model.py      # Full ML training pipeline
│
├── data/                   # Place dataset here (see Step 2)
│   └── training.1600000.processed.noemoticon.csv
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup & Running Instructions

### Step 1 — Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

---

### Step 2 — Download the Sentiment140 Dataset

1. Visit the dataset page on Kaggle:  
   **https://www.kaggle.com/datasets/kazanova/sentiment140**

2. Download the file:  
   `training.1600000.processed.noemoticon.csv`

3. Place it inside the `data/` folder:  
   ```
   sentiment-analysis-app/data/training.1600000.processed.noemoticon.csv
   ```

> **Note:** You need a free Kaggle account to download datasets.

---

### Step 3 — Train the Model

```bash
# Navigate to the model directory
cd model/

# Run the training script
python train_model.py
```

**What happens:**
- Loads and cleans 200,000 tweets (balanced sample for speed)
- Preprocesses: tokenization → stopword removal → stemming
- Extracts TF-IDF features (5,000 features, unigrams + bigrams)
- Trains Logistic Regression with saga solver
- Evaluates on 20% test split
- Saves `model.pkl` and `vectorizer.pkl` to `backend/`

**Expected output:**
```
Accuracy: ~79% (varies by sample)
Precision / Recall / F1: ~0.79 for each class
```

**To use the full 1.6M dataset** (slower, slightly better accuracy):
```bash
SAMPLE_SIZE=1600000 python train_model.py
```

**To specify a custom dataset path:**
```bash
DATASET_PATH=/path/to/csv OUTPUT_DIR=/path/to/backend python train_model.py
```

---

### Step 4 — Start the Backend API Server

```bash
# Navigate to backend
cd backend/

# Run Flask server
python app.py
```

The API will be available at: **http://localhost:5000**

**Test it with curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this, it made my day!"}'
```

**Expected response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.9123,
  "explanation": "The text expresses strong positive emotions...",
  "processed_text": "absolut love made day"
}
```

---

### Step 5 — Open the Frontend

Simply open the HTML file in your browser:

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or use a local server for best experience:
```bash
cd frontend/
python -m http.server 8080
# Then visit: http://localhost:8080
```

---

## 🔌 API Reference

### `GET /`
Health check — returns service info.

### `GET /health`
Returns `{"status": "healthy", "model_loaded": true}`.

### `POST /predict`

**Request:**
```json
{
  "text": "Your text here (max 5000 chars)"
}
```

**Response:**
```json
{
  "sentiment": "Positive | Negative | Neutral",
  "confidence": 0.0,
  "explanation": "Human-readable description",
  "processed_text": "stemmed tokens..."
}
```

**Error Response:**
```json
{
  "error": "Descriptive error message"
}
```

| Status | Meaning |
|--------|---------|
| 200    | Success |
| 400    | Bad request (missing/empty text) |
| 415    | Wrong Content-Type |
| 503    | Model not loaded |

---

## 🧠 ML Pipeline Details

| Component     | Choice |
|---------------|--------|
| Dataset       | Sentiment140 (1.6M tweets) |
| Cleaning      | Remove URLs, @mentions, #hashtags, special chars |
| Tokenizer     | NLTK word_tokenize |
| Stopwords     | NLTK English stopwords |
| Stemming      | Porter Stemmer |
| Features      | TF-IDF, max 5000, bigrams, sublinear TF |
| Model         | Logistic Regression (saga solver, C=1.0) |
| Train/Test    | 80% / 20% stratified split |
| Expected Acc  | ~78–82% |

**Why Logistic Regression?**
- Fast training even on large datasets
- Probabilistic output (confidence scores)
- Interpretable
- Good baseline for text classification

**Neutral sentiment** is inferred when max prediction probability < 55% — indicating the model isn't confident in either direction.

---

## 🚀 Production Deployment

### Use Gunicorn instead of Flask dev server:
```bash
cd backend/
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Serve frontend with Nginx:
```nginx
server {
    listen 80;
    root /path/to/frontend;
    index index.html;

    location /predict {
        proxy_pass http://localhost:5000;
    }
}
```

### Environment Variables:
```bash
FLASK_ENV=production    # Disable debug mode
DATASET_PATH=...        # Custom dataset location
OUTPUT_DIR=...          # Where to save models
SAMPLE_SIZE=200000      # Training sample size
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: model.pkl` | Run `train_model.py` first |
| `CORS error in browser` | Ensure Flask-CORS is installed and server is running |
| `Cannot connect to API` | Check Flask server is on port 5000 |
| `UnicodeDecodeError` | Use `encoding="latin-1"` (already handled in script) |
| `NLTK data not found` | Run `python -c "import nltk; nltk.download('all')"` |
| Low accuracy (<70%) | Try full dataset: `SAMPLE_SIZE=1600000 python train_model.py` |

---

## 📊 Sample Results

| Text | Sentiment | Confidence |
|------|-----------|------------|
| "I love this so much!" | Positive | 94% |
| "This is terrible, I hate it" | Negative | 91% |
| "The meeting is at 3pm" | Neutral | — |
| "Not the worst day ever" | Neutral | 52% |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

Built with ❤️ using Python · Flask · scikit-learn · NLTK · Sentiment140
