"""
src/train_model.py
-------------------
Trains THREE models and selects the best one:
  1. Logistic Regression   – fast, interpretable, good baseline
  2. Naive Bayes           – excellent for text, very fast
  3. Random Forest         – ensemble, handles non-linearity

Saves the best model + vectorizer to models/ directory.
Exports evaluation reports to outputs/ directory.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import MultinomialNB
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing    import LabelEncoder

from src.preprocess          import preprocess_dataframe
from src.feature_extraction  import fit_transform, save_vectorizer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/social_media_data.csv"
MODEL_DIR        = "models"
OUTPUT_DIR       = "outputs"
IMAGE_DIR        = "images"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR,  exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV dataset and verify required columns exist."""
    df = pd.read_csv(path)
    required = {"text", "sentiment"}
    assert required.issubset(df.columns), f"CSV must contain columns: {required}"
    print(f"✅ Loaded {len(df)} rows from {path}")
    print(df["sentiment"].value_counts(), "\n")
    return df


def encode_labels(df: pd.DataFrame) -> tuple:
    """Encode sentiment strings → integers (0/1/2)."""
    le = LabelEncoder()
    y  = le.fit_transform(df["sentiment"])
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return y, le


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       label_names: list) -> dict:
    """
    Train all three models, evaluate, return results dict.

    Returns:
        {model_name: {"model": obj, "accuracy": float, "report": str}}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, target_names=label_names)

        results[name] = {
            "model":    model,
            "accuracy": accuracy,
            "report":   report,
            "y_pred":   y_pred,
        }
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"  Accuracy : {accuracy * 100:.2f}%")
        print(f"{'='*55}")
        print(report)

    return results


def save_confusion_matrices(results: dict, y_test, label_names: list):
    """Plot and save confusion matrix for each model."""
    for name, res in results.items():
        cm   = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix – {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fname = name.lower().replace(" ", "_") + "_confusion_matrix.png"
        path  = os.path.join(IMAGE_DIR, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✅ Confusion matrix saved → {path}")


def save_accuracy_comparison(results: dict):
    """Bar chart comparing model accuracies."""
    names  = list(results.keys())
    scores = [r["accuracy"] * 100 for r in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores,
                  color=["#4CAF50", "#2196F3", "#FF9800"],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "model_accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Accuracy chart saved → {path}")


def save_best_model(best_name: str, best_model, le: LabelEncoder):
    """Pickle best model and label encoder."""
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    le_path    = os.path.join(MODEL_DIR, "label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\n🏆 Best model : {best_name}")
    print(f"✅ Model saved → {model_path}")
    print(f"✅ LabelEncoder saved → {le_path}")


def run_training():
    """End-to-end training pipeline."""
    print("\n" + "=" * 60)
    print("  SOCIAL MEDIA SENTIMENT ANALYSIS – MODEL TRAINING")
    print("=" * 60)

    # 1. Load & preprocess
    df = load_data()
    df = preprocess_dataframe(df, text_col="text")

    # 2. Encode labels
    y, le = encode_labels(df)
    label_names = list(le.classes_)

    # 3. TF-IDF features
    X, vectorizer = fit_transform(df["processed_text"])
    save_vectorizer(vectorizer)
    print(f"✅ TF-IDF matrix shape: {X.shape}")

    # 4. Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 5. Train & evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test, label_names)

    # 6. Save visuals
    save_confusion_matrices(results, y_test, label_names)
    save_accuracy_comparison(results)

    # 7. Pick best model
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    save_best_model(best_name, best_model, le)

    # 8. Save classification reports
    report_path = os.path.join(OUTPUT_DIR, "classification_reports.txt")
    with open(report_path, "w") as f:
        for name, res in results.items():
            f.write(f"\n{'='*55}\n{name}\nAccuracy: {res['accuracy']*100:.2f}%\n")
            f.write(res["report"])
    print(f"✅ Reports saved → {report_path}")

    print("\n✅ Training complete!")
    return results


if __name__ == "__main__":
    run_training()
