"""
src/predict.py
---------------
Loads trained model + vectorizer and predicts sentiment
for any new text input (single string or batch list).

Also provides VADER-based rule-based sentiment for comparison.
"""

import os
import pickle

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.preprocess         import preprocess_text
from src.feature_extraction import transform, load_vectorizer

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"
LE_PATH    = "models/label_encoder.pkl"
VEC_PATH   = "models/tfidf_vectorizer.pkl"


def load_artifacts():
    """Load model, label encoder, and vectorizer from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    vectorizer = load_vectorizer(VEC_PATH)
    return model, le, vectorizer


def predict_ml(texts: list, model, le, vectorizer) -> list:
    """
    ML-based sentiment prediction.

    Args:
        texts:      list of raw text strings
        model:      trained classifier
        le:         LabelEncoder
        vectorizer: fitted TfidfVectorizer

    Returns:
        list of predicted sentiment labels (e.g. "Positive")
    """
    processed = [preprocess_text(t) for t in texts]
    X         = transform(processed, vectorizer)
    preds     = model.predict(X)
    proba     = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for i, label_idx in enumerate(preds):
        label = le.inverse_transform([label_idx])[0]
        conf  = round(float(proba[i].max()) * 100, 1) if proba is not None else None
        results.append({"text": texts[i], "sentiment": label, "confidence": conf})
    return results


def predict_vader(texts: list) -> list:
    """
    Rule-based VADER sentiment analysis (no training needed).
    Useful for quick checks and comparison with ML predictions.

    Compound score:
        >= 0.05  → Positive
        <= -0.05 → Negative
        else     → Neutral
    """
    sia     = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores    = sia.polarity_scores(text)
        compound  = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({
            "text":      text,
            "sentiment": label,
            "compound":  round(compound, 4),
            "pos":       scores["pos"],
            "neu":       scores["neu"],
            "neg":       scores["neg"],
        })
    return results


def predict_single(text: str, use_ml: bool = True) -> dict:
    """
    Convenience function for predicting a single text.

    Args:
        text:   raw social media post
        use_ml: True → ML model, False → VADER

    Returns:
        dict with 'text', 'sentiment', and score info
    """
    if use_ml:
        model, le, vectorizer = load_artifacts()
        return predict_ml([text], model, le, vectorizer)[0]
    else:
        return predict_vader([text])[0]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "Zomato delivery was super fast today! Loved the packaging!",
        "Netflix keeps buffering. Very frustrating experience!",
        "Placed my order. Waiting for delivery.",
        "Amazon delivered a damaged product. Refund process is a nightmare.",
        "Really happy with my purchase. Exactly as described online.",
    ]

    print("=" * 65)
    print("  SENTIMENT PREDICTION DEMO")
    print("=" * 65)

    # VADER (no model needed)
    print("\n📊 VADER (Rule-Based) Predictions:\n")
    vader_results = predict_vader(test_texts)
    for r in vader_results:
        emoji = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}.get(r["sentiment"], "")
        print(f"{emoji} [{r['sentiment']:8s}] (compound={r['compound']:+.3f}) → {r['text'][:60]}")

    # ML Model
    try:
        print("\n🤖 ML Model Predictions:\n")
        model, le, vectorizer = load_artifacts()
        ml_results = predict_ml(test_texts, model, le, vectorizer)
        for r in ml_results:
            emoji = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}.get(r["sentiment"], "")
            conf  = f"(conf={r['confidence']}%)" if r["confidence"] else ""
            print(f"{emoji} [{r['sentiment']:8s}] {conf} → {r['text'][:60]}")
    except FileNotFoundError:
        print("⚠️  Model not found. Run src/train_model.py first.")
