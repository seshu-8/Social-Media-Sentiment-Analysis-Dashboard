# 📊 Social Media Sentiment Analysis Dashboard

> **An industry-grade NLP & Machine Learning project** that analyzes social media posts, tweets, and customer reviews to classify sentiment as Positive, Negative, or Neutral — with an interactive real-time dashboard.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-green)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 What Is This Project?

**Sentiment Analysis** is the process of using Natural Language Processing (NLP) and Machine Learning to automatically detect the emotional tone behind text — whether a customer is happy, frustrated, or neutral about a product, brand, or service.

**Real-world applications:**
- 🛒 **E-Commerce** (Amazon, Flipkart): Track product reviews & return complaints
- 🍕 **Food Delivery** (Zomato, Swiggy): Monitor delivery experience feedback
- 🎬 **Streaming** (Netflix): Analyze show reviews & subscription churn signals
- 🏦 **Banks & Fintech**: Detect complaints before they escalate
- 📱 **Startups**: Monitor brand reputation across social platforms

---

## 🚀 Live Dashboard Features

| Feature | Description |
|---|---|
| 📊 Overview | Sentiment distribution (pie + bar) |
| 🏢 Brand Analysis | Per-brand sentiment scorecard |
| 📈 Trend Analysis | Daily sentiment timeline |
| 🤖 Live Predictor | Type any text → instant sentiment |
| 🔍 Data Explorer | Full dataset with download option |
| 💡 Business Insights | Auto-generated brand health report |

---

## 🏗️ Project Architecture

```
Raw Social Media Text
        │
        ▼
┌─────────────────────┐
│   Text Cleaning     │  → lowercase, remove URLs, mentions, emojis
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  NLP Preprocessing  │  → tokenize, remove stopwords, lemmatize
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │  → convert text → numerical features (5000 features)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ML Classifier     │  → Logistic Regression / Naive Bayes / Random Forest
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Sentiment Prediction│  → Positive / Negative / Neutral
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Streamlit Dashboard │  → Interactive charts, live predictor, insights
└─────────────────────┘
```

---

## 📁 Folder Structure

```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/
│   ├── generate_dataset.py     ← Creates synthetic social media dataset
│   └── social_media_data.csv   ← Generated dataset (1500 posts)
│
├── src/
│   ├── __init__.py
│   ├── clean_text.py           ← URL/emoji/punctuation removal
│   ├── preprocess.py           ← Tokenization, stopwords, lemmatization
│   ├── feature_extraction.py   ← TF-IDF vectorization
│   ├── train_model.py          ← Train 3 ML models, save best
│   ├── predict.py              ← ML + VADER predictions
│   └── evaluate.py             ← Charts & business insights
│
├── app/
│   └── dashboard.py            ← Streamlit interactive dashboard
│
├── models/
│   ├── best_model.pkl          ← Serialized best classifier
│   ├── tfidf_vectorizer.pkl    ← Fitted TF-IDF vectorizer
│   └── label_encoder.pkl       ← Label encoder
│
├── outputs/
│   ├── preprocessed_data.csv
│   ├── predictions_output.csv
│   ├── classification_reports.txt
│   └── insights_report.txt
│
├── images/
│   ├── sentiment_distribution.png
│   ├── sentiment_by_brand.png
│   ├── sentiment_by_platform.png
│   ├── sentiment_trend_over_time.png
│   ├── model_accuracy_comparison.png
│   └── *_confusion_matrix.png
│
├── notebooks/
│   └── sentiment_analysis_eda.ipynb
│
├── docs/
│   └── architecture.md
│
├── main.py                     ← Pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 – Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard
```

### Step 2 – Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A – Full Pipeline (Recommended for first run)
```bash
python main.py --all
```
This runs: generate → preprocess → train → evaluate → predict

### Option B – Step by step
```bash
# 1. Generate synthetic dataset
python main.py --generate

# 2. Preprocess text
python main.py --preprocess

# 3. Train ML models
python main.py --train

# 4. Generate charts
python main.py --evaluate

# 5. Demo predictions
python main.py --predict

# 6. Launch dashboard
python main.py --dashboard
```

### Launch Dashboard directly
```bash
streamlit run app/dashboard.py
```
Then open → **http://localhost:8501**

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~88% |
| Naive Bayes | ~85% |
| Random Forest | ~87% |

> Accuracy may vary slightly due to random dataset generation.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| NLP | NLTK (tokenizer, stopwords, lemmatizer, VADER) |
| Features | TF-IDF Vectorizer (scikit-learn) |
| Models | Logistic Regression, Naive Bayes, Random Forest |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Serialization | Pickle |

---

## 🎓 Learning Outcomes

After building this project, you will understand:
- ✅ End-to-end NLP pipeline design
- ✅ Text cleaning and preprocessing techniques
- ✅ TF-IDF feature extraction
- ✅ Training and comparing multiple ML classifiers
- ✅ Evaluation metrics (accuracy, F1, confusion matrix)
- ✅ Building interactive dashboards with Streamlit & Plotly
- ✅ Structuring professional GitHub projects

---

## 📋 Interview Q&A

**Q: Why TF-IDF over Bag of Words?**
A: TF-IDF penalizes very common words and rewards rare, informative words — making it more suitable for sentiment classification.

**Q: Why keep negation words when removing stopwords?**
A: Words like "not", "never", "no" completely flip sentiment. Removing them would cause "not good" to be classified as "good".

**Q: How is VADER different from ML-based sentiment?**
A: VADER is a rule-based lexicon model requiring no training — great for social media but less accurate for domain-specific text. ML models learn patterns from labeled data and generalize better.

**Q: How would you scale this to handle real Twitter/Instagram data?**
A: Use Twitter API v2 or Instagram Graph API for data collection, replace synthetic CSV with real-time stream, add a database (PostgreSQL/MongoDB), and deploy on AWS/GCP with a CI/CD pipeline.

---

## 👤 Author

**[Your Name]**
- GitHub: [@your-username](https://github.com/seshu-8)
- LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/seshu-babu-konijeti-74968b2b9?utm_source=share_via&utm_content=profile&utm_medium=member_android)
- Email: seshubabukv1200@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ **If this project helped you, please give it a star!**
