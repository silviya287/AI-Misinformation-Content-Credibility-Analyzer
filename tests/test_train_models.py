"""
tests/test_train_models.py

Tests for the reusable helper functions in training/train_models.py
(Milestone 4). We test evaluate() and select_best_model_name() with
small, hand-crafted, known inputs rather than running full training -
that keeps these tests fast and makes it obvious exactly what's being
checked.

Run with:
    pytest tests/test_train_models.py -v
"""

from training.train_models import evaluate, select_best_model_name


def test_evaluate_perfect_predictions():
    """If predictions exactly match the truth, all metrics should be 1.0."""
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_known_mistakes():
    """
    y_true: [1, 0, 1, 0]
    y_pred: [1, 0, 0, 0]  (missed one positive case - a false negative)

    Accuracy should be 3/4 = 0.75.
    Precision should be 1.0 (every predicted "1" was correct).
    Recall should be 0.5 (only caught 1 of the 2 real positives).
    """
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_select_best_model_name_picks_highest_f1():
    """select_best_model_name should return the model with the highest F1, regardless of other metrics."""
    fake_results = {
        "naive_bayes": {"accuracy": 0.9, "precision": 0.9, "recall": 0.5, "f1": 0.64},
        "logistic_regression": {"accuracy": 0.85, "precision": 0.8, "recall": 0.9, "f1": 0.85},
        "svm": {"accuracy": 0.88, "precision": 0.7, "recall": 0.7, "f1": 0.70},
    }

    assert select_best_model_name(fake_results) == "logistic_regression"
