"""
src/evaluate.py
----------------
Generates evaluation charts and business-insight visualizations
from the full dataset with predicted sentiments.

Outputs saved to images/ and outputs/ directories.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess          import preprocess_dataframe
from src.feature_extraction  import load_vectorizer, transform
from src.predict             import predict_vader

IMAGE_DIR  = "images"
OUTPUT_DIR = "outputs"
os.makedirs(IMAGE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}


def load_predicted_df(data_path: str = "data/social_media_data.csv") -> pd.DataFrame:
    """Load dataset and add ML + VADER predictions."""
    df  = pd.read_csv(data_path)
    df  = preprocess_dataframe(df)

    # ML predictions
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        vectorizer = load_vectorizer()
        X = transform(df["processed_text"], vectorizer)
        df["ml_sentiment"] = le.inverse_transform(model.predict(X))
    except FileNotFoundError:
        df["ml_sentiment"] = df["sentiment"]   # fallback

    # VADER predictions
    vader_res            = predict_vader(df["text"].tolist())
    df["vader_sentiment"] = [r["sentiment"] for r in vader_res]
    df["vader_compound"]  = [r["compound"]  for r in vader_res]

    return df


# ── Chart generators ──────────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, col: str = "sentiment"):
    """Pie + bar chart of overall sentiment distribution."""
    counts = df[col].value_counts()
    labels = counts.index.tolist()
    colors = [COLORS.get(l, "#9E9E9E") for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overall Sentiment Distribution", fontsize=15, fontweight="bold")

    # Pie
    axes[0].pie(counts, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Proportional View")

    # Bar
    axes[1].bar(labels, counts, color=colors, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Number of Posts")
    axes[1].set_title("Count View")
    for i, (label, count) in enumerate(zip(labels, counts)):
        axes[1].text(i, count + 5, str(count), ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "sentiment_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved → {path}")


def plot_sentiment_by_brand(df: pd.DataFrame):
    """Grouped bar chart: sentiment per brand."""
    brand_sent = df.groupby(["brand", "sentiment"]).size().unstack(fill_value=0)
    brand_sent = brand_sent.reindex(
        columns=[c for c in ["Positive", "Negative", "Neutral"] if c in brand_sent.columns]
    )
    brand_sent.plot(kind="bar", figsize=(12, 6),
                    color=[COLORS.get(c, "#9E9E9E") for c in brand_sent.columns],
                    edgecolor="white")
    plt.title("Sentiment Distribution by Brand", fontsize=14, fontweight="bold")
    plt.xlabel("Brand")
    plt.ylabel("Number of Posts")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "sentiment_by_brand.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved → {path}")


def plot_sentiment_by_platform(df: pd.DataFrame):
    """Stacked bar chart: sentiment per platform."""
    plt_sent = df.groupby(["platform", "sentiment"]).size().unstack(fill_value=0)
    plt_sent = plt_sent.reindex(
        columns=[c for c in ["Positive", "Negative", "Neutral"] if c in plt_sent.columns]
    )
    plt_sent.plot(kind="bar", stacked=True, figsize=(10, 6),
                  color=[COLORS.get(c, "#9E9E9E") for c in plt_sent.columns],
                  edgecolor="white")
    plt.title("Sentiment Distribution by Platform", fontsize=14, fontweight="bold")
    plt.xlabel("Platform")
    plt.ylabel("Number of Posts")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "sentiment_by_platform.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved → {path}")


def plot_sentiment_over_time(df: pd.DataFrame):
    """Line chart: daily sentiment trends."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    daily = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    for sent, color in COLORS.items():
        if sent in daily.columns:
            ax.plot(daily.index, daily[sent], label=sent, color=color,
                    linewidth=2, marker="o", markersize=3)
    ax.set_title("Sentiment Trend Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Posts")
    ax.legend(title="Sentiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "sentiment_trend_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved → {path}")


def plot_vader_compound_distribution(df: pd.DataFrame):
    """Histogram of VADER compound scores colored by sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for sent, color in COLORS.items():
        subset = df[df["vader_sentiment"] == sent]["vader_compound"]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=sent, edgecolor="white")
    ax.axvline(0.05,  color="green",  linestyle="--", linewidth=1.2, label="Positive threshold")
    ax.axvline(-0.05, color="red",    linestyle="--", linewidth=1.2, label="Negative threshold")
    ax.set_title("VADER Compound Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Compound Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "vader_compound_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved → {path}")


def generate_insights_report(df: pd.DataFrame):
    """Write a plain-text business insights report."""
    total    = len(df)
    counts   = df["sentiment"].value_counts()
    pos_pct  = counts.get("Positive", 0) / total * 100
    neg_pct  = counts.get("Negative", 0) / total * 100
    neu_pct  = counts.get("Neutral",  0) / total * 100

    top_neg_brand = df[df["sentiment"] == "Negative"]["brand"].value_counts().idxmax()
    top_pos_brand = df[df["sentiment"] == "Positive"]["brand"].value_counts().idxmax()

    report = f"""
====================================================================
   SOCIAL MEDIA SENTIMENT ANALYSIS – BUSINESS INSIGHTS REPORT
====================================================================

OVERVIEW
--------
Total Posts Analyzed : {total}
Positive Sentiment   : {counts.get('Positive', 0):,} posts  ({pos_pct:.1f}%)
Negative Sentiment   : {counts.get('Negative', 0):,} posts  ({neg_pct:.1f}%)
Neutral Sentiment    : {counts.get('Neutral',  0):,} posts  ({neu_pct:.1f}%)

BRAND INSIGHTS
--------------
Brand with most POSITIVE mentions : {top_pos_brand}
Brand with most NEGATIVE mentions : {top_neg_brand}

PLATFORM BREAKDOWN
------------------
{df.groupby('platform')['sentiment'].value_counts().to_string()}

TOP BRANDS POSITIVE RATE
------------------------
{(df[df['sentiment']=='Positive'].groupby('brand').size() /
  df.groupby('brand').size() * 100).round(1).to_string()}

RECOMMENDATIONS
---------------
1. {top_neg_brand} should focus on improving customer support & delivery.
2. Monitor spikes in negative sentiment for immediate crisis response.
3. Amplify positive user posts in marketing campaigns.
4. Use neutral posts to identify unanswered customer queries.
5. Track weekly sentiment trends to measure campaign effectiveness.

====================================================================
"""
    path = os.path.join(OUTPUT_DIR, "insights_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"✅ Insights report saved → {path}")
    print(report)


def run_all_charts(data_path: str = "data/social_media_data.csv"):
    """Generate all charts and the insights report."""
    print("\n" + "=" * 60)
    print("  GENERATING EVALUATION CHARTS & INSIGHTS")
    print("=" * 60 + "\n")
    df = load_predicted_df(data_path)

    plot_sentiment_distribution(df)
    plot_sentiment_by_brand(df)
    plot_sentiment_by_platform(df)
    plot_sentiment_over_time(df)
    plot_vader_compound_distribution(df)
    generate_insights_report(df)

    # Save predicted CSV
    out_path = os.path.join(OUTPUT_DIR, "predictions_output.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions CSV saved → {out_path}")


if __name__ == "__main__":
    run_all_charts()
