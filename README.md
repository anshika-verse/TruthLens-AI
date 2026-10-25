# TruthLens AI — Explainable Fake News Detection System

TruthLens AI is a Machine Learning and Natural Language Processing application designed to classify news text as either **Real** or **Fake** using supervised text classification models and TF-IDF feature extraction.

The project emphasizes model interpretability, transparent evaluation metrics (accuracy, precision, recall, F1, confusion matrix), and clear disclaimers regarding the limits of statistical NLP classifiers.

---

## 🚀 Features

- **Standardized Text Preprocessing:** Regex-based cleaning, HTML stripping, URL removal, and punctuation normalization.
- **TF-IDF Vectorization:** Feature extraction with unigram and bigram representation.
- **Model Comparison:** Evaluates three baseline classifiers:
  1. Logistic Regression
  2. Multinomial Naive Bayes
  3. Linear SVM
- **Explainability:** Identifies top influential vocabulary features that mathematically contributed to the prediction.
- **Interactive Streamlit Web Dashboard:** Clean, responsive UI with model analysis, performance breakdown, and ethical disclaimers.
- **Safe Setup Mode:** Graceful handling of missing models or unplaced datasets without crashing.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.14)
- Git (optional)

### 2. Virtual Environment Setup
Activate the local virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
Or create a fresh one:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📊 Dataset Setup & Model Training

### 1. Download Dataset
Download the **Fake and Real News Dataset** from Kaggle:
👉 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place the two extracted files into the `data/` folder:
```
data/
├── Fake.csv
└── True.csv
```

### 2. Train and Evaluate Models
Run the training pipeline:
```bash
python train.py
```
This script will:
- Validate dataset integrity and eliminate duplicate content.
- Perform stratified train (70%) / validation (15%) / test (15%) splits.
- Fit TF-IDF vectorizer strictly on training data (leakage prevention).
- Train and compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- Select the best performing model on the validation split.
- Evaluate the champion model on the held-out test split.
- Save the fitted pipeline to `models/truthlens_model.joblib` and metrics to `models/evaluation_results.json`.

---

## 💻 Running the Streamlit Web Application

Launch the interactive web app:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## ⚠️ Responsible AI & Model Limitations Notice

1. **Statistical Correlation vs. Truth:** Text classification models detect linguistic patterns, vocabulary choices, and structural quirks present in the training set. They do not have live access to reality or real-time ground truth.
2. **Dataset-Specific Biases:** Datasets like the Kaggle Fake/Real news corpus frequently feature stylistic publisher markers (e.g., `"WASHINGTON (Reuters)"` in real articles). Models may latch onto these formatting cues.
3. **Fact Verification:** Model output should never be treated as an authoritative verdict. High-stakes claims must always be verified using established, independent journalistic sources.
