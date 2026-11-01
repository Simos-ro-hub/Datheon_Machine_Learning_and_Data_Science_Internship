# Task 4: NLP Sentiment Classification Pipeline

**Datheon Machine Learning & Data Science Internship**
- **Applicant:** Azmain Haq Siam
- **Application ID:** DT-2026-YN1B7
- **Deliverable:** NLP classification pipeline notebook (.ipynb) with accuracy benchmarks and interactive test helper.

## Overview
This project builds a Natural Language Processing (NLP) pipeline to classify customer product or movie reviews as positive or negative. The process involves raw text preprocessing, numerical feature extraction using TF-IDF, and training robust classification models.

## Pipeline Architecture
1. **Data Ingestion:** Loads review data.
2. **Text Preprocessing:** Utilizes `NLTK` and regular expressions to normalize the text by converting to lowercase, removing HTML tags, URLs, and punctuation, filtering out stopwords, and performing lemmatization.
3. **Feature Extraction:** Converts cleaned text to a numerical matrix using `TfidfVectorizer` (with `max_features=5000` and unigrams + bigrams).
4. **Model Training:** Trains and benchmarks `MultinomialNB` (Naive Bayes) and `LogisticRegression`.
5. **Evaluation:** Provides accuracy, precision, recall, F1-scores, and confusion matrices.
6. **Inference Function:** Includes a custom `predict_sentiment()` function to test live phrases.

## Execution
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `Datheon_Task4_NLP_Sentiment_Classifier.ipynb` and execute all cells.
