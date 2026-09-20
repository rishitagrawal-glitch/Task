# SMS Spam Detection

An end-to-end machine learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate), covering preprocessing, model comparison, evaluation and a saved, reusable model.

## Problem

Spam messages waste time and are often scams. The goal is to flag spam while rarely blocking real messages, so both **precision** (avoid false alarms) and **recall** (catch spam) matter. Accuracy alone would be misleading because only about 12.6% of messages are spam.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI): 5,572 labelled messages, bundled in `data/sms.tsv` (the script re-downloads it if missing). After removing 403 duplicate messages, 5,169 remain: 4,516 ham and 653 spam.

## Approach

1. **Preprocessing:** drop duplicates, lowercase, and replace URLs, currency amounts and long phone numbers with placeholder tokens (`urltoken`, `moneytoken`, `longnumbertoken`). Spam is full of these, so the model learns from them instead of losing them.
2. **Features:** TF-IDF on unigrams and bigrams, English stop words removed, words appearing fewer than 2 times ignored.
3. **Split:** stratified 80/20 train/test split. The test set is used only once, at the end.
4. **Model selection:** Naive Bayes, Logistic Regression and Linear SVM (class-weight balanced), compared with 5-fold stratified cross-validation on the training set using F1 for the spam class.
5. **Leakage prevention:** the vectorizer and classifier live in a single scikit-learn `Pipeline`, so the vocabulary is learned from training data only.
6. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC and a confusion matrix on the held-out test set.

## Results

Cross-validation (F1 on spam): Naive Bayes 0.935, Logistic Regression 0.948, **Linear SVM 0.953** (selected).

Test set (1,034 messages, 131 of them spam):

| Metric | Score |
|---|---|
| Accuracy | 98.7% |
| Precision (spam) | 96.1% |
| Recall (spam) | 93.1% |
| F1 (spam) | 94.6% |
| ROC-AUC | 0.997 |

The model made 5 false positives (ham flagged as spam) and 9 false negatives (spam missed). The strongest spam indicators were the placeholder tokens for long numbers, URLs and money, plus words like `txt`, `freephone`, `reply` and `won`.

## Limitations

- The data is a small collection of older, mostly UK and Singapore messages, so it may not generalize to modern spam. Some quirks appear, such as `arsenal` being a spam indicator because of football-prize scams in the dataset.
- Only 131 spam messages are in the test set, so the metrics carry some uncertainty.
- The decision threshold could be tuned if false positives are costlier than missed spam.

## Setup and usage

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
python spam_detection.py
```

The script prints all results and writes these files to `outputs/`: `eda.png`, `confusion_matrix.png` and `spam_model.joblib`.

Use the saved model on new messages:

```python
import joblib
model = joblib.load("outputs/spam_model.joblib")
print(model.predict(["You won a free prize! Call 09061701461 now"]))  # [1] = spam
```

## Files

```
├── spam_detection.py    # Full pipeline
├── data/sms.tsv         # Dataset
├── outputs/             # Charts and saved model (generated)
├── requirements.txt
└── README.md
```
