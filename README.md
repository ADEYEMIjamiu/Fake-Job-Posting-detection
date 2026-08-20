# Fake Job Posting Detection

AI-based fraud detection system for online job postings, built for the B198c7 AI Applications for Digital Business module at Gisma University of Applied Sciences.

## Overview

This project uses NLP and machine learning (TF-IDF + Linear SVM) to identify potentially fraudulent job postings from their text content, addressing the significant class imbalance in the dataset (~4.8% fraudulent).

## Dataset

Real / Fake Job Posting Prediction (EMSCAD), available on Kaggle:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Not included in this repo — download separately and place in a local `data/` folder to reproduce.

## Repository Structure

```
notebooks/
  fake_job_posting_analysis.ipynb   # Full analysis: EDA, preprocessing, modelling, evaluation
requirements.txt
README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Open `notebooks/fake_job_posting_analysis.ipynb` in either Jupyter or Google Colab. Download the dataset CSV from the Kaggle link above and update the file path in the notebook's data-loading cell accordingly.

## Approach

1. Exploratory data analysis on class balance, text length, and field missingness patterns
2. Text preprocessing: cleaning, normalisation (stopword/lemmatisation handling tested empirically)
3. TF-IDF vectorisation with train/test split performed prior to vectorisation to prevent data leakage
4. Comparison of three classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM) with class weighting for imbalance handling
5. Threshold tuning based on precision-recall tradeoff analysis
6. Feature importance analysis for model interpretability

## Author

Adeyemi Jamiu Adegbenro (GH1030456)
