# 📅 7-Day GitHub Proof Plan

## Day 1 – Environment Setup & Folder Structure

**What to do:**
```bash
mkdir Social-Media-Sentiment-Analysis-Dashboard
cd Social-Media-Sentiment-Analysis-Dashboard
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
git init
git add .
git commit -m "chore: initial project setup with folder structure and requirements"
git push origin main
```

**Screenshot to save:** `images/day1_folder_structure.png`
- Take a screenshot of VS Code showing the full project folder tree

---

## Day 2 – Dataset Generation

```bash
python main.py --generate
git add data/
git commit -m "feat: add synthetic social media dataset generator (1500 posts)"
git push
```

**Screenshot:** `images/day2_dataset_preview.png`
- Run `python -c "import pandas as pd; print(pd.read_csv('data/social_media_data.csv').head(10).to_string())"`
- Screenshot the terminal output

---

## Day 3 – Text Cleaning & Preprocessing

```bash
python main.py --preprocess
git add src/clean_text.py src/preprocess.py outputs/preprocessed_data.csv
git commit -m "feat: add text cleaning and NLP preprocessing pipeline"
git push
```

**Screenshot:** `images/day3_preprocessing_output.png`
- Show before/after text cleaning in terminal

---

## Day 4 – TF-IDF Feature Extraction & Model Training

```bash
python main.py --train
git add src/ models/
git commit -m "feat: TF-IDF vectorization + train Logistic Regression, Naive Bayes, Random Forest"
git push
```

**Screenshot:** `images/day4_training_output.png`
- Screenshot accuracy scores for all 3 models in terminal

---

## Day 5 – Model Evaluation & Charts

```bash
python main.py --evaluate
git add images/ outputs/
git commit -m "feat: add evaluation charts, confusion matrices, and business insights report"
git push
```

**Screenshots to save:**
- `images/sentiment_distribution.png`
- `images/model_accuracy_comparison.png`
- `images/logistic_regression_confusion_matrix.png`

---

## Day 6 – Dashboard Creation

```bash
streamlit run app/dashboard.py
# Take screenshots of each tab
git add app/
git commit -m "feat: add interactive Streamlit dashboard with live predictor and Plotly charts"
git push
```

**Screenshots:**
- `images/dashboard_overview.png`
- `images/dashboard_live_predictor.png`
- `images/dashboard_brand_analysis.png`

---

## Day 7 – Final Polish & GitHub README

```bash
git add README.md docs/ .gitignore
git commit -m "docs: add professional README with architecture, results, and setup guide"
git push
```

**Final checks:**
- ✅ Repository has a clear description
- ✅ Topics/tags added on GitHub
- ✅ README has screenshots embedded
- ✅ All code is commented
- ✅ requirements.txt is up to date

---

## GitHub Repository Setup Commands

```bash
# 1. Create repo on github.com first (name: Social-Media-Sentiment-Analysis-Dashboard)

# 2. Connect local to remote
git remote add origin https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Analysis-Dashboard.git

# 3. Push
git branch -M main
git push -u origin main
```

**Best GitHub topics to add:**
`nlp`, `sentiment-analysis`, `machine-learning`, `streamlit`, `python`, 
`scikit-learn`, `tfidf`, `social-media`, `data-science`, `dashboard`

---

## Proof Checklist

- [ ] GitHub repo created and public
- [ ] All code pushed with meaningful commits
- [ ] README has project description, tech stack, architecture diagram
- [ ] Screenshots of dataset, charts, dashboard in README
- [ ] models/ contains trained .pkl files
- [ ] outputs/ contains classification report and insights
- [ ] images/ contains all chart PNGs
- [ ] requirements.txt is complete
- [ ] .gitignore is set up
- [ ] Project runs end-to-end without errors

---

## Interview Preparation Checklist

Practice explaining:
- [ ] What is TF-IDF and why you chose it
- [ ] Why 3 models were trained and compared
- [ ] Why negation words are kept in preprocessing
- [ ] How VADER differs from ML-based sentiment
- [ ] How you would scale this to handle real Twitter data
- [ ] What the confusion matrix tells you about your model
- [ ] What precision, recall, and F1-score mean
- [ ] How you would add a new data source (e.g., Instagram API)
