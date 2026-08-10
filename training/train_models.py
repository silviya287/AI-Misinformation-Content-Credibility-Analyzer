"""
training/train_models.py

Module B - Model Training (Milestone 4).

This is a ONE-TIME SCRIPT you run manually to train the models -
it is NOT part of the running desktop app. You run it, it produces
.pkl files in models/, and the app (Milestone 5 onward) just loads
those files.

For each of the 5 categories, this script:
    1. Loads that category's dataset (training/dataset_loader.py)
    2. Cleans the text the SAME way the running app will
       (core/preprocessing.py) - this consistency is critical
    3. Splits into train/test sets
    4. Converts text to TF-IDF feature vectors
    5. Trains 3 classical models: Naive Bayes, Logistic Regression, SVM
    6. Evaluates all 3 on the test set (accuracy, precision, recall, F1,
       confusion matrix) and prints a comparison table
    7. Saves ONLY the best-performing model (by F1 score) + its
       TF-IDF vectorizer to models/<category>_model.pkl /
       models/<category>_vectorizer.pkl

Run it with:
    python training/train_models.py
"""

import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Let this script be run directly (python training/train_models.py)
# by making sure the project root is on the import path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.dataset_loader import load_dataset, VALID_CATEGORIES, DatasetLoadError
from core.preprocessing import clean_and_tokenize, tokens_to_text

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_models():
    """
    Returns a fresh dict of the 3 untrained models we compare.
    A fresh dict is built each time this is called (once per category)
    so training on one category never affects another category's model.

    Note: LinearSVC doesn't give probability scores out of the box, but
    Milestone 5 needs a confidence % for every model type. Wrapping it
    in CalibratedClassifierCV adds that probability support while still
    being "an SVM" underneath - this keeps all 3 models usable the same
    way later (model.predict() and model.predict_proba()).
    """
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": CalibratedClassifierCV(LinearSVC(), cv=3),
    }


def prepare_features(texts: list[str]) -> list[str]:
    """
    Runs every text sample through the SAME cleaning pipeline the app
    uses at prediction time, then rejoins tokens into a string (what
    TfidfVectorizer expects as input).
    """
    return [tokens_to_text(clean_and_tokenize(text)) for text in texts]


def evaluate(y_true, y_pred) -> dict:
    """
    Computes the 4 required comparison metrics + confusion matrix for
    one model's predictions on the test set.

    zero_division=0 avoids crashing/warning if a model happens to
    predict only one class on a very small dataset.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def select_best_model_name(results: dict) -> str:
    """
    Given {"naive_bayes": {...metrics...}, "logistic_regression": {...}, "svm": {...}},
    returns the name of the model with the highest F1 score.
    F1 is used (rather than accuracy) because it balances precision and
    recall, which matters more than raw accuracy for flagging tasks
    like spam/toxic detection where classes are often imbalanced.
    """
    return max(results, key=lambda model_name: results[model_name]["f1"])


def print_comparison_table(category: str, results: dict, best_model_name: str):
    """Prints a simple, readable comparison table to the console."""
    print(f"\n=== {category.upper()} — Model Comparison ===")
    print(f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    for model_name, metrics in results.items():
        marker = "  <-- BEST" if model_name == best_model_name else ""
        print(
            f"{model_name:<22}"
            f"{metrics['accuracy']:<12.3f}"
            f"{metrics['precision']:<12.3f}"
            f"{metrics['recall']:<12.3f}"
            f"{metrics['f1']:<10.3f}{marker}"
        )
    print(f"Confusion matrix ({best_model_name}): {results[best_model_name]['confusion_matrix']}")


def train_category(category: str):
    """
    Runs the full training + comparison + save pipeline for ONE
    category. Returns True if a model was successfully trained and
    saved, False if this category had to be skipped (e.g. no dataset
    file yet).
    """
    try:
        texts, labels = load_dataset(category)
    except DatasetLoadError as error:
        print(f"\n[train_models] Skipping '{category}': {error}")
        return False

    # Clean every text sample the same way the app will at prediction time.
    cleaned_texts = prepare_features(texts)

    # Split into training and testing sets. stratify=labels keeps the
    # 0/1 ratio consistent between train and test, which matters a lot
    # on small datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Convert text into TF-IDF feature vectors. Fit ONLY on training
    # data (never on test data) to avoid leaking test information into
    # the vectorizer's vocabulary/weights.
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train and evaluate all 3 models.
    models = build_models()
    results = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train_vectors, y_train)
        predictions = model.predict(X_test_vectors)
        results[model_name] = evaluate(y_test, predictions)
        trained_models[model_name] = model

    best_model_name = select_best_model_name(results)
    print_comparison_table(category, results, best_model_name)

    # Save only the winning model + the vectorizer it was trained with.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{category}_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, f"{category}_vectorizer.pkl")

    joblib.dump(trained_models[best_model_name], model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved best model ('{best_model_name}') to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    return True


def main():
    print("Starting training for all categories...")
    trained_count = 0

    for category in VALID_CATEGORIES:
        if train_category(category):
            trained_count += 1

    print(f"\nDone. Trained and saved models for {trained_count}/{len(VALID_CATEGORIES)} categories.")
    if trained_count < len(VALID_CATEGORIES):
        print("See training/datasets/DATASETS.md to add the missing dataset(s).")


if __name__ == "__main__":
    main()
