"""
src/preprocess.py
------------------
NLP preprocessing pipeline:
  - Tokenization
  - Stopword removal (NLTK)
  - Lemmatization (WordNetLemmatizer)
  - Final processed text construction
"""

import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.clean_text import clean_text

# ── Download required NLTK resources (runs once) ─────────────────────────────
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
# Keep sentiment-carrying negations
KEEP_WORDS = {"not", "no", "never", "nor", "neither", "without"}
STOP_WORDS -= KEEP_WORDS

lemmatizer = WordNetLemmatizer()


def tokenize(text: str) -> list:
    """Split cleaned text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords while keeping negation words."""
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize(tokens: list) -> list:
    """Reduce each token to its base/root form."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(raw_text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps:
        1. Clean raw text
        2. Tokenize
        3. Remove stopwords
        4. Lemmatize
        5. Rejoin into a single string

    Args:
        raw_text: original social media post

    Returns:
        processed text string ready for vectorization
    """
    cleaned   = clean_text(raw_text)
    tokens    = tokenize(cleaned)
    tokens    = [t for t in tokens if t.isalpha()]  # keep only alphabetic tokens
    tokens    = remove_stopwords(tokens)
    tokens    = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "text") -> pd.DataFrame:
    """
    Apply full preprocessing to every row of a DataFrame.

    Args:
        df:       input DataFrame
        text_col: name of the column containing raw text

    Returns:
        DataFrame with an added 'processed_text' column
    """
    df = df.copy()
    df["cleaned_text"]   = df[text_col].apply(clean_text)
    df["processed_text"] = df[text_col].apply(preprocess_text)
    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Zomato delivery was super fast today! Loved the packaging 🎉",
        "Netflix keeps buffering even on high-speed internet. Very frustrating!",
        "Placed my Zomato order. Waiting for delivery.",
    ]
    print("=" * 60)
    print("PREPROCESSING MODULE – TEST OUTPUT")
    print("=" * 60)
    for s in samples:
        print(f"\nOriginal  : {s}")
        print(f"Processed : {preprocess_text(s)}")
