# TruthLens AI
## Explainable Fake News Detection System Using Machine Learning and NLP

**Project type:** Machine Learning + Natural Language Processing (NLP)  
**Proposed level:** Advanced BTech / portfolio project  
**Primary language:** Python  
**Initial interface:** Streamlit  
**Optional advanced architecture:** FastAPI + React  
**Project status:** Proposed design and implementation roadmap

---

## 1. Project Overview

**TruthLens AI** is a news-text analysis application designed to estimate whether a news headline or article resembles the *fake* or *real* examples in its labeled training data. It combines Natural Language Processing (NLP), machine learning, model evaluation, and explainability features in an interactive application.

The system is intended to help users investigate news—not to make an unquestionable determination of truth. A text classifier can learn writing styles, publisher patterns, or dataset-specific artifacts instead of reliable factual signals. For this reason, the application should communicate uncertainty and clearly distinguish a model prediction from independently verified evidence.

### 1.1 Problem Statement

Online misinformation can spread quickly, while readers may not have the time or skills to investigate every claim. A text-classification model can provide an initial signal for further review, but its predictions must be interpreted carefully. TruthLens AI aims to create an accessible tool that analyzes news text, explains some factors influencing its prediction, and can later be extended with source retrieval and evidence-based claim verification.

### 1.2 Project Objectives

- Build an NLP-based binary news-text classifier using labeled data.
- Compare multiple machine-learning algorithms under a consistent evaluation setup.
- Provide predictions through an easy-to-use web interface.
- Show appropriate evaluation metrics rather than relying only on accuracy.
- Add explainability to identify text features that influenced a model's prediction.
- Investigate model generalization using independent data where feasible.
- Provide a foundation for optional article extraction, source retrieval, and evidence comparison.
- Handle limitations, privacy, and user safety transparently.

### 1.3 Expected Users

- Students learning NLP and machine learning.
- Readers who want an initial signal before investigating a news item.
- Researchers or educators demonstrating text classification and model limitations.

---

## 2. Proposed Features

### 2.1 Core Features (MVP)

1. **News text input:** Accept a headline or article body.
2. **Text validation:** Reject empty input and handle overly long text safely.
3. **NLP preprocessing:** Apply consistent, documented text preparation.
4. **TF-IDF vectorization:** Convert text into numerical features.
5. **Model comparison:** Train and evaluate multiple baseline classifiers.
6. **Prediction interface:** Show the model's predicted class.
7. **Evaluation dashboard:** Display precision, recall, F1-score, and a confusion matrix.
8. **Model information:** Identify the selected model and the dataset used.
9. **Limitations notice:** Explain that a prediction is not a factual verdict.

### 2.2 Recommended Enhancements

- **Explainable AI:** Use LIME or another suitable method to highlight influential words or phrases.
- **Headline/body comparison:** Estimate semantic consistency between a headline and the article body.
- **Error analysis:** Review false positives and false negatives.
- **Model comparison charts:** Compare models using the same test set.
- **Calibration:** If probability estimates are shown, evaluate and calibrate them using validation data.
- **Independent dataset evaluation:** Test whether the model generalizes beyond its training dataset.
- **Analysis history:** Optionally store recent analyses with user consent and a clear retention policy.

### 2.3 Advanced / Optional Features

- **URL article extraction:** Extract publicly accessible article text and metadata from a supplied URL.
- **Source information:** Display the publisher, URL, and available publication date.
- **Related-source retrieval:** Retrieve relevant reporting or reference material from permitted sources.
- **Claim-evidence comparison:** Compare a claim with retrieved evidence and label the result as supported, contradicted, or insufficient evidence.
- **Multilingual support:** Evaluate Hindi, English, and/or Hinglish separately with suitable labeled datasets.
- **Transformer models:** Fine-tune or evaluate a pretrained model such as DistilBERT or a multilingual transformer.
- **React frontend:** Replace or supplement Streamlit with a custom frontend backed by FastAPI.

These advanced features should only be presented as available after they have actually been implemented and tested. Retrieved links alone do not establish that a claim is true.

---

## 3. Recommended Technology Stack

| Component | Recommended technology | Purpose |
|---|---|---|
| Programming language | Python | Main application and ML code |
| Data processing | Pandas, NumPy | Loading, cleaning, and analyzing datasets |
| NLP utilities | NLTK or spaCy (as needed) | Text processing |
| Feature extraction | Scikit-learn TF-IDF | Convert text into sparse numerical features |
| Baseline ML | Logistic Regression | Interpretable baseline classifier |
| Baseline ML | Multinomial Naive Bayes | Fast text-classification baseline |
| Baseline ML | Linear SVM | Strong classical text-classification candidate |
| Advanced NLP (optional) | Hugging Face Transformers | Transformer-based text classification |
| Explainability | LIME or SHAP | Investigate which features influence predictions |
| Interface | Streamlit | Build an interactive application quickly |
| API (optional) | FastAPI | Expose prediction functionality through endpoints |
| Frontend (optional) | React | Custom full-stack interface |
| Visualization | Plotly, Matplotlib | Metrics and charts |
| Model persistence | Joblib | Save a trained scikit-learn pipeline |
| Database (optional) | SQLite | Store opt-in analysis history or metadata |
| Version control | Git and GitHub | Track code and showcase the project |
| Deployment (optional) | Streamlit Community Cloud or Hugging Face Spaces | Publish a demo |

### Recommended starting point

Begin with **Python + Pandas + Scikit-learn + TF-IDF + Streamlit**. Add explainability and API separation after the baseline system works. A React frontend and transformer model are optional extensions, not prerequisites for a strong project.

### Official learning resources

- Python: https://docs.python.org/3/
- Scikit-learn: https://scikit-learn.org/stable/user_guide.html
- NLTK: https://www.nltk.org/
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- LIME: https://github.com/marcotcr/lime
- SHAP: https://shap.readthedocs.io/

---

## 4. Dataset Strategy

A model's output is only as reliable as the data, labels, and evaluation design behind it. Record the source, license, label definitions, and limitations of every dataset used.

### 4.1 Starter Dataset: Kaggle Fake and Real News Dataset

Resource: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

This dataset is commonly used for introductory binary news classification. The downloadable files typically contain fake and real news examples, with fields such as title and text. Check the actual files before writing code because schemas can vary.

**Potential uses**
- Initial data exploration.
- TF-IDF experiments.
- Baseline model comparison.
- A first end-to-end demo.

**Limitations**
- Labels are dataset annotations, not universal ground truth.
- The model may learn publisher, topic, or writing-style patterns.
- Performance on a random test split may overstate real-world performance.
- Dataset license and permitted use must be checked before redistribution.

### 4.2 Research-Oriented Dataset: LIAR

Paper and dataset information: https://aclanthology.org/P17-2067/

LIAR contains short statements annotated with multiple truthfulness labels. It is not simply interchangeable with a binary fake/real article dataset.

**Potential uses**
- Multi-class classification experiments.
- Independent evaluation, where a carefully justified label mapping is possible.
- Studying differences between statement verification and article classification.

### 4.3 Dataset Recommendations

- Start with one dataset and understand its columns and label meanings.
- Check missing values, duplicates, near-duplicates, and class balance.
- Keep a held-out test set separate until final evaluation.
- Fit vectorizers and preprocessing transformations on training data only.
- If possible, group duplicates or closely related stories in the same split.
- Consider a source-based or time-based evaluation to test generalization.
- Do not combine datasets blindly; document any label mapping and its assumptions.
- For Hindi or Hinglish, use suitable labeled data and evaluate each language separately.

---

## 5. System Architecture

### 5.1 Core Classification Pipeline

```text
User enters headline/article
            |
            v
Input validation
            |
            v
Text preprocessing
            |
            v
TF-IDF vectorization
            |
            v
Trained ML classifier
            |
            v
Predicted training-data class
            |
            v
Explanation + limitations notice
```

### 5.2 Development and Evaluation Pipeline

```text
Labeled dataset
      |
      v
Data validation and exploration
      |
      v
Duplicate / leakage checks
      |
      v
Train-validation-test split
      |
      v
TF-IDF + candidate model pipelines
      |
      v
Model selection on validation data
      |
      v
Final evaluation on held-out test data
      |
      v
Save final pipeline and metrics
      |
      v
Streamlit application
```

### 5.3 Optional Evidence-Verification Pipeline

```text
News URL or factual claim
          |
          v
Safe article extraction / claim extraction
          |
          v
Retrieve relevant independent sources
          |
          v
Rank and display evidence passages
          |
          v
Compare claim with retrieved evidence
          |
          v
Supported / contradicted / insufficient evidence
          |
          v
Show source links, dates, and limitations
```

The evidence module must not invent sources or citations. It should handle missing, outdated, conflicting, or low-quality evidence explicitly.

---

## 6. NLP Preprocessing

Text preprocessing should be consistent between training and prediction. Preserve the original input for display and evidence retrieval; use a separate processed version for the classifier.

### 6.1 Potential preprocessing steps

- Convert missing text to a safe representation and handle it explicitly.
- Normalize whitespace.
- Remove HTML markup where appropriate.
- Decide whether to remove, preserve, or separately extract URLs.
- Consider lowercasing for a case-insensitive baseline.
- Preserve meaningful negation, numbers, dates, and other potentially useful information.
- Compare punctuation/capitalization handling rather than assuming these signals are useless.
- Validate language and input length.

Avoid aggressive cleaning by default. Removing all punctuation, numbers, or negation can discard useful information. Preprocessing choices should be tested experimentally.

### 6.2 Illustrative cleaning function

```python
import re

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

This is a simple example, not a universal best practice. If URLs or capitalization may matter for a particular experiment, preserve them in separate features or compare alternate preprocessing pipelines.

---

## 7. Feature Extraction with TF-IDF

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts text into numerical features that classical ML algorithms can process. It gives more weight to terms that are informative within a document relative to how common they are across the corpus.

Example configuration:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
```

These are starting parameters only. Tune them using training/validation data.

**Leakage prevention:** Do not fit TF-IDF on the complete dataset before splitting. Use a scikit-learn `Pipeline` so the vectorizer is fit only on the training fold and then applied to validation/test data.

---

## 8. Machine-Learning Models

Train and compare at least three classical models:

### 8.1 Logistic Regression
- Useful baseline for sparse TF-IDF text features.
- Relatively fast to train.
- Coefficients can help inspect influential features.

### 8.2 Multinomial Naive Bayes
- Common text-classification baseline.
- Fast and simple.
- Makes strong feature-independence assumptions, so it should be evaluated rather than assumed to be best.

### 8.3 Linear SVM
- Often a strong candidate for high-dimensional sparse text data.
- Suitable for comparison against simpler baselines.
- `LinearSVC` does not directly provide calibrated class probabilities.

### 8.4 Optional Transformer Model
A model such as DistilBERT or a multilingual transformer may capture contextual information, but it requires more computation and careful fine-tuning. It is not automatically better than a classical model.

### Example scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    ("classifier", LinearSVC())
])
```

This creates an untrained pipeline. It must be fitted on training data before it can make useful predictions. Keep the vectorizer and classifier together when saving the model.

---

## 9. Training and Evaluation

### 9.1 Data splitting

A stratified 80/20 train/test split is a reasonable initial baseline when the dataset is sufficiently large and independent examples can be assumed. A validation set or cross-validation on training data can be used for model selection.

Example:

```python
from sklearn.model_selection import train_test_split

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Before using this split, verify that `df["text"]` and `df["label"]` exist and contain the expected data. If the same article, syndicated story, or near-duplicate appears in both splits, the test result may be misleading. Group related examples where feasible.

### 9.2 Metrics to report

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Of items predicted as a class, the fraction that truly have that dataset label |
| Recall | Of items with a dataset label, the fraction the model identifies |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Counts of correct and incorrect predictions by class |
| Calibration (if applicable) | Whether predicted probabilities align with observed frequencies |

Report class-wise precision, recall, and F1, not just a single overall score. The appropriate trade-off depends on the intended use.

### 9.3 Required experiments

1. Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
2. Compare unigram TF-IDF against unigram + bigram TF-IDF.
3. Compare headline-only, body-only, and headline-plus-body inputs.
4. Inspect false positives and false negatives.
5. Evaluate on independent data when compatible labels are available.
6. If probabilities are displayed, assess calibration on validation data.
7. Document the dataset, split strategy, random seed, and model settings.

A high score on one dataset does not prove reliable real-world fact-checking.

---

## 10. Explainability

Use LIME or another appropriate method to investigate which text features influenced a prediction.

Possible UI output:
- Predicted dataset class.
- Important words/phrases that influenced the model.
- A short explanation of how to interpret the explanation.
- A notice that influential wording is not proof of falsity.

Example:

```text
Model prediction: Fake-labeled class
Influential features: [computed from the trained model]
Interpretation: These features influenced the model's output.
They are not, by themselves, evidence that the claim is false.
```

Do not hard-code sample words as if they were actual model explanations. Generate explanations from the trained model and the user's input.

---

## 11. Headline and Article Consistency (Optional)

A headline may not accurately summarize the article body. A separate module can compare them using sentence embeddings and cosine similarity, or use a suitable Natural Language Inference (NLI) model.

Possible output:
- High, medium, or low semantic similarity.
- Relevant headline/body passages.
- A note that low similarity is a review signal, not proof of misinformation.

This feature should be evaluated independently and should not be treated as a replacement for fact-checking.

---

## 12. Source Retrieval and Evidence Verification (Advanced)

### 12.1 URL article extraction
If the user supplies a URL, the application may attempt to extract:
- Article title.
- Main text.
- Publisher/domain.
- Publication date, if available.

Respect site terms, access rules, and rate limits. Article extraction may fail because of paywalls, JavaScript rendering, or unavailable metadata.

### 12.2 Evidence retrieval
For a factual claim, retrieve relevant independent sources and display:
- Source title and link.
- Publisher and publication date where available.
- Relevant passage.
- Whether the passage appears to support, contradict, or not resolve the claim.

### 12.3 Responsible interpretation
- “Supported” means supported by the retrieved evidence, not guaranteed universally true.
- “Contradicted” requires clear evidence that conflicts with the claim.
- “Insufficient evidence” is appropriate when sources are missing, weak, or conflicting.
- Do not infer truth from a domain name alone.
- Do not fabricate sources, quotes, dates, or citations.
- Display source provenance so users can inspect evidence themselves.

---

## 13. Hindi and Hinglish Support (Optional)

English-only training does not establish performance on Hindi or Hinglish.

To add multilingual support:
1. Obtain a suitable labeled dataset for the target language(s).
2. Inspect scripts, spelling variation, transliteration, and label quality.
3. Compare a multilingual transformer with appropriate baselines.
4. Keep separate evaluation results for English, Hindi, and Romanized Hindi/Hinglish.
5. Document unsupported languages and failure cases.

A multilingual model is a candidate, not a guarantee of accurate language support.

---

## 14. User Interface Design

A Streamlit MVP can have the following pages or sections:

### Home
- Project title and short explanation.
- Input mode: headline or article text.
- Optional URL input (only if implemented).
- Analyze button and input validation.

### Analysis Results
- Predicted dataset class.
- Score or probability only if its meaning is clearly defined and appropriately calibrated.
- Explainability output, if implemented.
- Clear statement that model output is not independent verification.

### Model Performance
- Precision, recall, F1-score, and accuracy.
- Confusion matrix.
- Model comparison charts.
- Dataset and test-split information.

### Evidence (advanced)
- Retrieved sources and passages.
- Publication dates where available.
- Support/contradiction/insufficient-evidence assessment.
- Retrieval limitations.

### About
- Dataset information.
- Model details.
- Known limitations.
- Privacy policy and supported languages.

Suggested visual style: a clean, readable interface with clear hierarchy, accessible contrast, and distinct but non-alarming status indicators. Avoid presenting a red “FAKE” badge as a definitive factual judgment.

---

## 15. Suggested Project Structure

```text
TruthLens-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   └── config.py
├── models/
│   └── .gitkeep
├── reports/
│   ├── model_comparison.csv
│   └── evaluation_summary.json
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
└── api/
    └── main.py
```

The `api/` directory is optional if FastAPI is introduced. Do not commit large datasets, private user inputs, or model artifacts without checking size and license requirements.

---

## 16. Security, Privacy, and Reliability

- Validate empty, malformed, and oversized input.
- Handle missing values and unexpected file formats gracefully.
- Preserve original input separately from the classifier's cleaned text.
- Do not store user-submitted article text by default; make history opt-in and disclose retention.
- If fetching URLs, protect against Server-Side Request Forgery (SSRF): block localhost/private IP destinations, validate redirects, use timeouts, and restrict protocols.
- Respect website terms, robots policies where applicable, and rate limits.
- Record model version, dataset version, training date, and evaluation metrics.
- Keep dependencies pinned for reproducibility.
- Test the inference pipeline and UI failure cases.
- Avoid claims of guaranteed detection or universal language coverage.
- Explain that a model score measures its learned prediction, not the objective probability that a news item is false unless calibration and interpretation support that claim.

---

## 17. Development Roadmap

| Phase | Tasks | Deliverable |
|---|---|---|
| Week 1 | Select dataset, inspect labels, clean data, explore class balance and duplicates | Validated dataset and EDA |
| Week 2 | Build TF-IDF pipelines and train baseline models | Initial trained models |
| Week 3 | Evaluate, compare models, investigate errors and leakage | Evaluation report |
| Week 4 | Build Streamlit interface and connect saved model | Working MVP |
| Week 5 | Add explainability and improved charts | More interpretable application |
| Week 6 | Add tests, documentation, deployment, and independent evaluation where feasible | Portfolio-ready version |
| Later | Add URL extraction, evidence retrieval, multilingual support, or transformer experiments | Advanced extension |

The schedule is an estimate and depends on experience, dataset quality, and feature scope. Complete the MVP before starting all advanced features.

---

## 18. Testing Checklist

### Data and model
- [ ] Dataset files and columns are validated.
- [ ] Missing values and duplicates are checked.
- [ ] Label mapping is documented.
- [ ] Train/test leakage is checked.
- [ ] TF-IDF is fitted only on training data.
- [ ] Candidate models are compared consistently.
- [ ] Class-wise metrics and confusion matrix are generated.
- [ ] False positives and false negatives are reviewed.
- [ ] Independent evaluation is attempted where feasible.

### Application
- [ ] Empty input is rejected.
- [ ] Very long input is handled safely.
- [ ] Prediction uses the saved pipeline.
- [ ] The model version and dataset are disclosed.
- [ ] No fake metrics or hard-coded explanations are shown.
- [ ] The limitation notice is visible.
- [ ] Errors are shown in understandable language.

### Optional URL/evidence module
- [ ] Unsafe destinations and redirects are blocked.
- [ ] Timeouts and response-size limits are enforced.
- [ ] Retrieval failures are handled.
- [ ] Sources and publication dates are displayed only when actually retrieved.
- [ ] Conflicting or insufficient evidence is represented honestly.
- [ ] No claim is labeled verified solely because of the publisher domain.

---

## 19. Resume Description (Use Only After Implementation)

**TruthLens AI — Explainable Fake News Detection System**

- Developed an NLP-based news-text classification pipeline using TF-IDF and multiple machine-learning algorithms.
- Compared Logistic Regression, Multinomial Naive Bayes, and Linear SVM using precision, recall, F1-score, and confusion matrices.
- Built an interactive Streamlit interface for analyzing headlines and article text.
- Added explainability and error-analysis tools to investigate model behavior.
- Evaluated limitations and generalization using held-out and, where feasible, independent data.

Only retain bullets for features actually implemented. Add measured metrics only after running the evaluation; never invent accuracy or claim that the system reliably fact-checks all news.

---

## 20. Final Recommended Scope

### Build first
- One documented binary dataset.
- Data validation and duplicate checks.
- TF-IDF + Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- Leakage-aware evaluation.
- Saved scikit-learn pipeline.
- Streamlit app.
- Performance dashboard and limitations notice.

### Add next
- LIME-based explanations.
- Headline/body consistency analysis.
- Independent dataset evaluation.
- Model calibration if probabilities are shown.
- Unit tests and reproducible setup instructions.

### Add only after the core works
- URL extraction.
- Independent source retrieval.
- Evidence-based claim comparison.
- Hindi/Hinglish support.
- Transformer fine-tuning.
- FastAPI + React architecture.

This staged approach keeps the project achievable while leaving room for technically meaningful improvements.

---

## 21. Quick Start Plan

1. Create a Python virtual environment.
2. Install Pandas, NumPy, Scikit-learn, Joblib, and Streamlit.
3. Download and inspect the selected dataset.
4. Validate columns, labels, missing values, and duplicates.
5. Split data before fitting the TF-IDF vectorizer.
6. Train and compare the baseline models.
7. Save the complete fitted pipeline with Joblib.
8. Build a Streamlit interface that loads the saved pipeline.
9. Test on unseen examples and inspect errors.
10. Document metrics, limitations, and reproducible run instructions.

**Important:** The code snippets in this document are examples, not a complete ready-to-run application. The project is complete only after the training, evaluation, inference, interface, and tests are implemented and executed.
