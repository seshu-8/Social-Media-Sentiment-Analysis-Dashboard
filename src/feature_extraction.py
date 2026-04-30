"""
src/feature_extraction.py
--------------------------
TF-IDF vectorization for converting processed text
into numerical feature vectors that ML models can use.

TF-IDF = Term Frequency × Inverse Document Frequency
  - Gives higher weight to words that are unique and important
  - Reduces weight of very common words (even after stopword removal)
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Default vectorizer config ─────────────────────────────────────────────────
VECTORIZER_PARAMS = {
    "max_features": 5000,      # keep top 5000 most informative words
    "ngram_range":  (1, 2),    # unigrams + bigrams (e.g. "not good")
    "min_df":       2,         # ignore words appearing in < 2 docs
    "max_df":       0.95,      # ignore words appearing in > 95% of docs
    "sublinear_tf": True,      # apply log normalization to TF
}


def build_vectorizer() -> TfidfVectorizer:
    """Return a new TF-IDF vectorizer with default params."""
    return TfidfVectorizer(**VECTORIZER_PARAMS)


def fit_transform(texts, vectorizer: TfidfVectorizer = None):
    """
    Fit vectorizer on training texts and transform them.

    Args:
        texts:      iterable of processed text strings
        vectorizer: optional pre-created vectorizer

    Returns:
        (X_sparse_matrix, fitted_vectorizer)
    """
    if vectorizer is None:
        vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def transform(texts, vectorizer: TfidfVectorizer):
    """
    Transform new texts using an already-fitted vectorizer.

    Args:
        texts:      iterable of processed text strings
        vectorizer: previously fitted TfidfVectorizer

    Returns:
        X_sparse_matrix
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer,
                    path: str = "models/tfidf_vectorizer.pkl"):
    """Serialize vectorizer to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved → {path}")


def load_vectorizer(path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """Load a previously saved vectorizer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_top_features(vectorizer: TfidfVectorizer, n: int = 20) -> list:
    """Return the top-n feature names learned by the vectorizer."""
    return vectorizer.get_feature_names_out()[:n].tolist()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "zomato delivery fast loved packaging",
        "netflix buffering frustrating internet",
        "placed order waiting delivery",
        "product quality exceeded expectations recommended",
        "received damaged product refund nightmare",
    ]
    X, vec = fit_transform(sample_texts)
    print("=" * 60)
    print("FEATURE EXTRACTION – TEST OUTPUT")
    print("=" * 60)
    print(f"Matrix shape     : {X.shape}")
    print(f"Vocabulary size  : {len(vec.vocabulary_)}")
    print(f"Top 10 features  : {get_top_features(vec, 10)}")
