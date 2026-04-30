# 🏗️ Project Architecture – Social Media Sentiment Analysis Dashboard

## Overview

This document describes the complete system architecture, data flow,
module responsibilities, and design decisions for the project.

---

## System Architecture (Text Block Diagram)

```
╔══════════════════════════════════════════════════════════════════╗
║               SOCIAL MEDIA SENTIMENT ANALYSIS SYSTEM            ║
╚══════════════════════════════════════════════════════════════════╝

INPUT LAYER
┌────────────────────────────────────────────────────────────────┐
│  Synthetic Dataset (CSV)                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Twitter  │  │Instagram │  │ Facebook │  │Google Reviews│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│  Columns: id, platform, brand, text, sentiment, likes, date     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
CLEANING LAYER (src/clean_text.py)
┌────────────────────────────────────────────────────────────────┐
│  Input: "Zomato delivery was super fast! 🎉 @zomato #Zomato"   │
│                                                                  │
│  → lowercase       → remove URLs    → remove @mentions          │
│  → remove #tags    → remove emojis  → remove punctuation        │
│  → remove numbers  → strip spaces                               │
│                                                                  │
│  Output: "zomato delivery was super fast"                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
NLP PREPROCESSING LAYER (src/preprocess.py)
┌────────────────────────────────────────────────────────────────┐
│  Input: "zomato delivery was super fast"                        │
│                                                                  │
│  → Tokenize: ["zomato", "delivery", "was", "super", "fast"]    │
│  → Remove stopwords: ["zomato", "delivery", "super", "fast"]   │
│    (keep: not, no, never, nor)                                  │
│  → Lemmatize: ["zomato", "deliveri", "super", "fast"]          │
│  → Rejoin: "zomato deliveri super fast"                         │
│                                                                  │
│  Output: processed_text ready for vectorization                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
FEATURE EXTRACTION LAYER (src/feature_extraction.py)
┌────────────────────────────────────────────────────────────────┐
│  Algorithm: TF-IDF (Term Frequency × Inverse Document Freq.)   │
│                                                                  │
│  Settings:                                                       │
│  • max_features = 5000 (top 5000 words)                         │
│  • ngram_range  = (1, 2) (unigrams + bigrams)                   │
│  • min_df       = 2 (ignore words in < 2 documents)             │
│  • sublinear_tf = True (log normalization)                      │
│                                                                  │
│  Output: Sparse matrix  [n_samples × 5000]                      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
MODEL TRAINING LAYER (src/train_model.py)
┌────────────────────────────────────────────────────────────────┐
│  Train/Test Split: 80% / 20% (stratified)                      │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Logistic Regression │  │  Naive Bayes │  │Random Forest│  │
│  │  max_iter=1000       │  │  alpha=0.5   │  │ n=100 trees │  │
│  └──────────────────────┘  └──────────────┘  └─────────────┘  │
│                                                                  │
│  Best model selected by accuracy → saved as best_model.pkl     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
PREDICTION LAYER (src/predict.py)
┌────────────────────────────────────────────────────────────────┐
│  ML Pipeline:  new_text → clean → preprocess → TF-IDF →        │
│                model.predict() → label                         │
│                                                                  │
│  VADER Pipeline: new_text → compound score → threshold →        │
│                  Positive / Negative / Neutral                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
VISUALIZATION LAYER (src/evaluate.py)
┌────────────────────────────────────────────────────────────────┐
│  Charts:                                                         │
│  • Sentiment distribution (pie + bar)                           │
│  • Sentiment by brand (grouped bar)                             │
│  • Sentiment by platform (stacked bar)                          │
│  • Sentiment trend over time (line chart)                       │
│  • VADER compound score histogram                               │
│  • Model accuracy comparison                                    │
│  • Confusion matrices                                           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
DASHBOARD LAYER (app/dashboard.py)
┌────────────────────────────────────────────────────────────────┐
│  Streamlit + Plotly Interactive Dashboard                        │
│                                                                  │
│  Tabs:                                                           │
│  1. Overview      – KPI metrics + distribution charts           │
│  2. Brand Analysis– Grouped bars + scorecard table              │
│  3. Trend Analysis– Timeline + VADER histogram                  │
│  4. Live Predictor– Real-time text → sentiment                  │
│  5. Data Explorer – Full table + CSV download                   │
│  6. Insights      – Auto-generated business recommendations     │
│                                                                  │
│  Sidebar Filters: Platform | Brand | Sentiment | Date Range     │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
CSV File
  └─▶ Pandas DataFrame
        └─▶ Text Cleaning  (src/clean_text.py)
              └─▶ NLP Preprocessing  (src/preprocess.py)
                    └─▶ TF-IDF Vectorization  (src/feature_extraction.py)
                          └─▶ ML Model Training  (src/train_model.py)
                                └─▶ Sentiment Prediction  (src/predict.py)
                                      └─▶ Charts + Dashboard
```

---

## Key Design Decisions

### Why TF-IDF instead of Word2Vec or BERT?
- TF-IDF is fast, interpretable, and works well with scikit-learn
- No GPU required — works on any laptop
- Achieves 85–90% accuracy for 3-class sentiment on clean text
- BERT would be overkill for a beginner project and requires GPU

### Why three models?
- Allows fair comparison during evaluation
- Naive Bayes is the traditional text classification baseline
- Logistic Regression often outperforms on TF-IDF features
- Random Forest adds ensemble diversity

### Why keep negation words in stopword removal?
- "not good", "never satisfied", "no response" all carry negative sentiment
- Removing "not" would turn "not good" → "good" → wrong prediction

### Why synthetic data?
- No API keys or rate limits needed
- Reproducible, controlled experiments
- Beginner-friendly — no data scraping setup required
- Mirrors realistic social media text patterns
