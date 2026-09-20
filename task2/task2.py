
import re
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sms.tsv"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/"
    "pycon-2016-tutorial/master/data/sms.tsv"
)

def load_data() -> pd.DataFrame:
    """Load the SMS Spam Collection (tab-separated: label, message)."""
    if not DATA_PATH.exists():
        print(f"Downloading dataset to {DATA_PATH} ...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])

def clean_text(text: str) -> str:
    """Normalise a message so the model sees patterns, not raw tokens.

    Spam often contains URLs, phone numbers and prices, so instead of deleting
    them we replace them with placeholder tokens the model can learn from.
    """
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"[£$€]\s?\d+", " moneytoken ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumbertoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Preprocessing ---")
    print(f"Raw shape: {df.shape}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate messages removed: {n_dupes}")

    df["target"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    df["length"] = df["message"].str.len()
    print(f"Clean shape: {df.shape}")
    return df


def explore(df: pd.DataFrame) -> None:
    print("\n--- Exploration ---")
    counts = df["label"].value_counts()
    print(counts.to_string())
    print(f"Spam share: {counts['spam'] / len(df):.1%} (imbalanced dataset)")
    print("\nMean message length (characters):")
    print(df.groupby("label")["length"].mean().round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="label", ax=axes[0], hue="label", legend=False)
    axes[0].set_title("Class distribution")
    sns.histplot(data=df, x="length", hue="label", bins=40, ax=axes[1])
    axes[1].set_title("Message length by class")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda.png", dpi=120)
    plt.close(fig)


def build_pipeline(classifier) -> Pipeline:
    """TF-IDF features feeding a classifier. Bundling both in one Pipeline means
    the vectorizer is fit only on training data, which prevents data leakage."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("clf", classifier),
        ]
    )


def get_candidates() -> dict:
    return {
        "Naive Bayes": build_pipeline(MultinomialNB(alpha=0.1)),
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "Linear SVM": build_pipeline(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
        ),
    }


def compare_models(X_train, y_train) -> str:
    print(f"\n--- Model comparison ({CV_FOLDS}-fold stratified CV, F1 on spam) ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in get_candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        results[name] = scores.mean()
        print(f"{name:20s} F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    best = max(results, key=results.get)
    print(f"\nBest model by CV F1: {best}")
    return best

def spam_scores(model: Pipeline, X) -> np.ndarray:
    """Continuous spam score, used for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model: Pipeline, name: str, X_test, y_test) -> None:
    print(f"\n--- Test set evaluation: {name} ---")
    preds = model.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1 score : {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, spam_scores(model, X_test)):.4f}")
    print("\n", classification_report(y_test, preds, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"False positives (ham flagged as spam): {fp}")
    print(f"False negatives (spam that slipped through): {fn}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {name}")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def show_top_words(model: Pipeline, n: int = 10) -> None:
    """Print the words the model associates most strongly with spam."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    words = np.array(vec.get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    else:  # Naive Bayes: difference in log-probabilities between classes
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    print(f"\nTop {n} spam indicators:", ", ".join(words[np.argsort(weights)[-n:][::-1]]))
    print(f"Top {n} ham indicators: ", ", ".join(words[np.argsort(weights)[:n]]))

def demo(model: Pipeline) -> None:
    samples = [
        "Congratulations! You've won a free $1000 gift card. Click www.claim-now.com",
        "URGENT! Call 09061701461 now to claim your prize",
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you send me the notes from today's class?",
    ]
    print("\n--- Sample predictions ---")
    for msg, pred in zip(samples, model.predict(samples)):
        print(f"[{'SPAM' if pred else 'HAM ':4s}] {msg}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = preprocess(load_data())
    explore(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    print(f"\nTrain: {len(X_train)} messages | Test: {len(X_test)} messages")

    best_name = compare_models(X_train, y_train)
    best_model = get_candidates()[best_name].fit(X_train, y_train)

    evaluate(best_model, best_name, X_test, y_test)
    show_top_words(best_model)

    joblib.dump(best_model, OUTPUT_DIR / "spam_model.joblib")
    print(f"\nSaved model to {OUTPUT_DIR / 'spam_model.joblib'}")
    demo(best_model)


if __name__ == "__main__":
    main()