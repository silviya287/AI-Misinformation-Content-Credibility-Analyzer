"""
tests/test_ml_engine.py

Tests for core/ml_engine.py (Milestone 5).

Important: these tests do NOT rely on the real trained .pkl files or
their (currently tiny, synthetic) accuracy. They test the ENGINE'S
OWN LOGIC - risk level thresholds, credibility score math, and
explanation generation - using small, controlled, hand-built models so
we know exactly what the "correct" answer should be. Whether the real
trained models are accurate is a separate concern (see Milestone 4's
notes about needing real datasets).

Run with:
    pytest tests/test_ml_engine.py -v
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from core.ml_engine import (
    MLEngine,
    _risk_level_from_confidence,
)


# ---------- Pure function tests (no model needed) ----------

def test_risk_level_thresholds():
    """Confidence -> risk level mapping should match our finalized thresholds."""
    assert _risk_level_from_confidence(0) == "Low"
    assert _risk_level_from_confidence(49.9) == "Low"
    assert _risk_level_from_confidence(50) == "Medium"
    assert _risk_level_from_confidence(80) == "Medium"
    assert _risk_level_from_confidence(80.1) == "High"
    assert _risk_level_from_confidence(100) == "High"


def test_credibility_score_no_flags_is_100():
    """If nothing was flagged, credibility should be a perfect 100."""
    engine = MLEngine.__new__(MLEngine)  # build without loading any real models
    assert engine._calculate_credibility_score([]) == 100.0


def test_credibility_score_averages_flagged_confidences():
    """Credibility score should be 100 minus the AVERAGE of flagged confidences."""
    engine = MLEngine.__new__(MLEngine)
    # Two categories flagged at 90% and 70% confidence -> average 80 -> 100-80=20
    assert engine._calculate_credibility_score([90.0, 70.0]) == 20.0


def test_overall_risk_uses_highest_flagged_confidence():
    """Overall risk should reflect the single MOST confident flag, not the average."""
    engine = MLEngine.__new__(MLEngine)
    assert engine._calculate_overall_risk([]) == "Low"
    assert engine._calculate_overall_risk([55.0, 30.0]) == "Medium"  # max is 55
    assert engine._calculate_overall_risk([55.0, 90.0]) == "High"    # max is 90


# ---------- Tests using a small, controlled, hand-trained model ----------

def _build_tiny_engine():
    """
    Builds an MLEngine with ONE category ("spam") backed by a small,
    clearly-separable, hand-built model - not loaded from disk. This
    lets us assert exact expected behavior instead of depending on
    whatever the real (currently tiny/synthetic) trained models do.
    """
    texts = [
        "free prize click now", "win a free prize", "click here to win",
        "hello how are you", "lets meet for lunch", "see you tomorrow",
    ]
    labels = [1, 1, 1, 0, 0, 0]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(vectors, labels)

    engine = MLEngine.__new__(MLEngine)  # skip __init__'s disk loading
    engine.models = {"spam": model}
    engine.vectorizers = {"spam": vectorizer}
    return engine


def test_predict_flags_obvious_spam_text():
    """A clearly spam-like sentence should be flagged with high confidence."""
    engine = _build_tiny_engine()
    result = engine.predict("free prize click now")

    assert "spam" in result["flagged_categories"]
    assert result["categories"]["spam"]["is_flagged"] is True
    assert result["categories"]["spam"]["confidence"] > 50
    assert result["credibility_score"] < 100


def test_predict_does_not_flag_obvious_safe_text():
    """A clearly safe sentence should not be flagged."""
    engine = _build_tiny_engine()
    result = engine.predict("lets meet for lunch tomorrow")

    assert result["flagged_categories"] == []
    assert result["categories"]["spam"]["is_flagged"] is False
    assert result["credibility_score"] == 100.0


def test_explanation_includes_top_words_for_flagged_content():
    """Explanation text and top_words should reference the input's own words."""
    engine = _build_tiny_engine()
    result = engine.predict("free prize click now")

    top_words = result["categories"]["spam"]["top_words"]
    explanation = result["categories"]["spam"]["explanation"]

    assert len(top_words) > 0
    # Every word we claim "influenced" the result should be a real word
    # from the cleaned input text.
    for word in top_words:
        assert word in result["cleaned_text"]
    assert "Spam" in explanation
