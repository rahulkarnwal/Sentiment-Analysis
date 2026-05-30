/**
 * =============================================================
 *  SentiScope — Frontend JavaScript
 *  Connects the UI to the Flask Sentiment Analysis API
 * =============================================================
 */

// ── Configuration ─────────────────────────────────────────────
const API_URL = "http://127.0.0.1:5000/predict";

// ── DOM Element References ─────────────────────────────────────
const textInput       = document.getElementById("text-input");
const analyzeBtn      = document.getElementById("analyze-btn");
const loadingDiv      = document.getElementById("loading");
const errorBox        = document.getElementById("error-box");
const errorMessage    = document.getElementById("error-message");
const resultDiv       = document.getElementById("result");
const sentimentLabel  = document.getElementById("sentiment-label");
const sentimentIcon   = document.getElementById("sentiment-icon");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar   = document.getElementById("confidence-bar");
const explanationText = document.getElementById("explanation-text");
const processedText   = document.getElementById("processed-text");
const charCount       = document.getElementById("char-count");

// ── Sentiment Icons & Colors ───────────────────────────────────
const SENTIMENT_CONFIG = {
  Positive: { icon: "◉", cssClass: "positive", emoji: "😊" },
  Negative: { icon: "◎", cssClass: "negative", emoji: "😞" },
  Neutral:  { icon: "◌", cssClass: "neutral",  emoji: "😐" }
};

// ══════════════════════════════════════════════════════════════
//  MAIN: analyzeSentiment()
// ══════════════════════════════════════════════════════════════
async function analyzeSentiment() {
  const text = textInput.value.trim();

  if (!text) {
    showError("Please enter some text to analyze.");
    textInput.focus();
    return;
  }
  if (text.length < 3) {
    showError("Please enter at least 3 characters for meaningful analysis.");
    return;
  }

  setLoadingState(true);
  clearError();
  hideResult();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }

    displayResult(data);

  } catch (err) {
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      showError("Cannot connect to the API server. Make sure the Flask server is running on port 5000.");
    } else {
      showError(err.message || "An unexpected error occurred.");
    }
    console.error("[SentiScope] API Error:", err);

  } finally {
    setLoadingState(false);
  }
}

// ══════════════════════════════════════════════════════════════
//  displayResult()
// ══════════════════════════════════════════════════════════════
function displayResult(data) {
  const { sentiment, confidence, explanation, processed_text } = data;
  const config = SENTIMENT_CONFIG[sentiment] || SENTIMENT_CONFIG["Neutral"];

  sentimentLabel.textContent = sentiment;
  sentimentIcon.textContent  = config.emoji;

  sentimentIcon.style.transform = "scale(1.4)";
  setTimeout(() => {
    sentimentIcon.style.transform = "scale(1)";
    sentimentIcon.style.transition = "transform 0.4s cubic-bezier(0.34,1.56,0.64,1)";
  }, 50);

  const pct = Math.round(confidence * 100);
  confidenceValue.textContent = `${pct}%`;

  const topDiv = document.querySelector(".result__top");
  topDiv.className = `result__top ${config.cssClass}`;

  confidenceBar.style.width = "0%";
  confidenceBar.className = `confidence-bar ${config.cssClass}`;

  requestAnimationFrame(() => {
    setTimeout(() => { confidenceBar.style.width = `${pct}%`; }, 50);
  });

  explanationText.textContent = explanation || "—";
  processedText.textContent   = processed_text || "(empty after processing)";

  resultDiv.className = `result visible ${config.cssClass}-active`;

  setTimeout(() => {
    resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 100);
}

// ══════════════════════════════════════════════════════════════
//  UI State Helpers
// ══════════════════════════════════════════════════════════════
function setLoadingState(isLoading) {
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    loadingDiv.classList.add("visible");
    analyzeBtn.querySelector(".btn-text").textContent = "Analyzing…";
    analyzeBtn.querySelector(".btn-icon").textContent  = "…";
  } else {
    loadingDiv.classList.remove("visible");
    analyzeBtn.querySelector(".btn-text").textContent = "Analyze Sentiment";
    analyzeBtn.querySelector(".btn-icon").textContent  = "→";
  }
}

function showError(message) {
  errorMessage.textContent = message;
  errorBox.classList.add("visible");
}

function clearError() {
  errorBox.classList.remove("visible");
  errorMessage.textContent = "";
}

function hideResult() {
  resultDiv.className = "result";
}

// ══════════════════════════════════════════════════════════════
//  Example Pills
// ══════════════════════════════════════════════════════════════
document.querySelectorAll(".example-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    textInput.value = pill.getAttribute("data-text");
    updateCharCount();
    textInput.focus();
    clearError();
    hideResult();
    textInput.style.borderColor = "var(--accent)";
    textInput.style.boxShadow = "0 0 0 3px var(--accent-glow)";
    setTimeout(() => {
      textInput.style.borderColor = "";
      textInput.style.boxShadow = "";
    }, 800);
  });
});

// ══════════════════════════════════════════════════════════════
//  Character Counter
// ══════════════════════════════════════════════════════════════
function updateCharCount() {
  const len = textInput.value.length;
  charCount.textContent = len;
  if (len > 4500)      charCount.style.color = "var(--negative)";
  else if (len > 3500) charCount.style.color = "#f59e0b";
  else                 charCount.style.color = "";
}

textInput.addEventListener("input", updateCharCount);

// ══════════════════════════════════════════════════════════════
//  Keyboard Shortcut: Ctrl+Enter → Analyze
// ══════════════════════════════════════════════════════════════
textInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    analyzeSentiment();
  }
});

// Auto-clear error when user types
textInput.addEventListener("input", () => {
  if (errorBox.classList.contains("visible")) clearError();
});

// ══════════════════════════════════════════════════════════════
//  On Page Load
// ══════════════════════════════════════════════════════════════
window.addEventListener("DOMContentLoaded", () => {
  analyzeBtn.title = "Analyze Sentiment (Ctrl+Enter)";
  console.log(
    "%c SentiScope %c Ready ",
    "background:#7c6fff;color:#fff;padding:4px 8px;border-radius:4px 0 0 4px;font-weight:bold;",
    "background:#22c55e;color:#fff;padding:4px 8px;border-radius:0 4px 4px 0;font-weight:bold;"
  );
  console.log("API endpoint:", API_URL);
  console.log("Tip: Use Ctrl+Enter to analyze quickly.");
});
