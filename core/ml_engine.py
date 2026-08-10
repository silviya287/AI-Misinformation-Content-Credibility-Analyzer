"""
core/ml_engine.py

Module B - ML Engine & Explainability (Milestone 5).

This file is the bridge between the trained .pkl files (produced once
by training/train_models.py) and the running app. It has ONE class,
MLEngine, which:
    - loads all 5 category models + vectorizers ONCE when the app starts
    - for any piece of text, predicts whether each category applies
    - turns raw model output into something a human can read:
      confidence %, risk level (Low/Medium/High), a credibility score,
      the specific words that influenced the decision, and a plain-
      English explanation sentence

Public interface every other module should use:
    engine = MLEngine()
    result = engine.predict(text)

`result` is a plain dictionary - see predict()'s docstring below for
its exact shape. This shape is the "contract" the UI (Milestone 6)
and database (already built) will both rely on.
"""

import os
import sys

import joblib

# Let this module be run directly (python core/ml_engine.py) as well as
# imported normally by other modules - make sure the project root is on
# the import path either way.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.preprocessing import clean_and_tokenize, tokens_to_text

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORIES = ["spam", "phishing", "clickbait", "toxic", "fake_review"]

# Human-friendly display names, used in explanation sentences.
CATEGORY_DISPLAY_NAMES = {
    "spam": "Spam",
    "phishing": "Phishing",
    "clickbait": "Clickbait",
    "toxic": "Toxic Content",
    "fake_review": "Fake Review",
}

# How many top influencing words to show in the explanation.
TOP_WORDS_COUNT = 5


def _risk_level_from_confidence(confidence_percent: float) -> str:
    """
    Maps a confidence percentage to a simple Low/Medium/High risk label.
    Thresholds (per our finalized design):
        < 50%        -> Low
        50% - 80%    -> Medium
        > 80%        -> High
    """
    if confidence_percent > 80:
        return "High"
    elif confidence_percent >= 50:
        return "Medium"
    else:
        return "Low"


class MLEngine:
    """
    Loads every category's trained model + vectorizer once, then
    answers predict() calls using those already-loaded objects (fast -
    no disk access per prediction).
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models = {}
        self.vectorizers = {}

        for category in CATEGORIES:
            model_path = os.path.join(models_dir, f"{category}_model.pkl")
            vectorizer_path = os.path.join(models_dir, f"{category}_vectorizer.pkl")

            if os.path.isfile(model_path) and os.path.isfile(vectorizer_path):
                self.models[category] = joblib.load(model_path)
                self.vectorizers[category] = joblib.load(vectorizer_path)
            else:
                # Don't crash the whole app just because one category
                # hasn't been trained yet - skip it and let the rest
                # of the app work. predict() will simply omit this
                # category from its results.
                print(
                    f"[ml_engine] Warning: no trained model found for "
                    f"'{category}' - run training/train_models.py first. "
                    f"Skipping this category for now."
                )

    def predict(self, text: str) -> dict:
        """
        Runs the full ML Engine pipeline on a piece of text and returns
        a single result dictionary shaped like this:

        {
            "cleaned_text": "click free prize",
            "credibility_score": 62.5,
            "overall_risk_level": "Medium",
            "flagged_categories": ["spam", "clickbait"],
            "categories": {
                "spam": {
                    "is_flagged": True,
                    "confidence": 91.2,
                    "risk_level": "High",
                    "top_words": ["free", "prize", "click"],
                    "explanation": "This content was flagged as Spam ..."
                },
                "phishing": { ... },
                ...
            }
        }

        Every category that has a loaded model appears under
        "categories". Categories with no trained model yet are simply
        left out (see __init__).
        """
        tokens = clean_and_tokenize(text)
        cleaned_text = tokens_to_text(tokens)

        category_results = {}
        flagged_categories = []
        flagged_confidences = []

        for category in CATEGORIES:
            if category not in self.models:
                continue  # no trained model for this category yet

            result = self._predict_single_category(category, cleaned_text, tokens)
            category_results[category] = result

            if result["is_flagged"]:
                flagged_categories.append(category)
                flagged_confidences.append(result["confidence"])

        credibility_score = self._calculate_credibility_score(flagged_confidences)
        overall_risk_level = self._calculate_overall_risk(flagged_confidences)

        return {
            "cleaned_text": cleaned_text,
            "credibility_score": credibility_score,
            "overall_risk_level": overall_risk_level,
            "flagged_categories": flagged_categories,
            "categories": category_results,
        }

    def _predict_single_category(self, category: str, cleaned_text: str, tokens: list[str]) -> dict:
        """
        Runs one category's model on already-cleaned text and builds
        that category's result dictionary (confidence, risk level,
        top words, explanation).
        """
        model = self.models[category]
        vectorizer = self.vectorizers[category]

        vector = vectorizer.transform([cleaned_text])
        predicted_label = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]  # [P(class 0), P(class 1)]

        is_flagged = bool(predicted_label == 1)
        confidence = round(float(probabilities[1]) * 100, 1)  # confidence of being flagged
        risk_level = _risk_level_from_confidence(confidence) if is_flagged else "Low"

        top_words, explanation = self.explain_prediction(category, tokens, is_flagged, confidence)

        return {
            "is_flagged": is_flagged,
            "confidence": confidence,
            "risk_level": risk_level,
            "top_words": top_words,
            "explanation": explanation,
        }

    def explain_prediction(
        self, category: str, tokens: list[str], is_flagged: bool, confidence: float
    ) -> tuple[list[str], str]:
        """
        Builds a simple explanation for one category's prediction:
            - the words from the input that most strongly pushed the
              model toward the "flagged" class
            - a one-sentence, plain-English explanation string

        This is intentionally simple (per our finalized architecture -
        no SHAP/LIME): we just look at which of the input's own words
        have the highest learned weight for the "flagged" class.
        """
        display_name = CATEGORY_DISPLAY_NAMES[category]

        if not is_flagged:
            return [], f"No strong indicators of {display_name.lower()} were found in this content."

        feature_weights = self._get_feature_weight_lookup(category)
        if feature_weights is None:
            # Some model types might not expose usable weights - fail
            # gracefully with a generic explanation rather than crashing.
            return [], f"This content was flagged as {display_name} with {confidence}% confidence."

        # Only consider each unique word once, and only words that are
        # actually in this model's vocabulary.
        scored_words = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            if token in feature_weights:
                scored_words.append((token, feature_weights[token]))

        # Highest weight = pushed the model most strongly toward "flagged".
        scored_words.sort(key=lambda pair: pair[1], reverse=True)
        top_words = [word for word, weight in scored_words[:TOP_WORDS_COUNT] if weight > 0]

        if top_words:
            word_list = ", ".join(f"'{w}'" for w in top_words)
            explanation = (
                f"This content was flagged as {display_name} with {confidence}% confidence. "
                f"Words like {word_list} strongly influenced this result."
            )
        else:
            explanation = (
                f"This content was flagged as {display_name} with {confidence}% confidence, "
                f"based on its overall wording pattern."
            )

        return top_words, explanation

    def _get_feature_weight_lookup(self, category: str) -> dict | None:
        """
        Returns a {word: weight} dictionary for this category's model,
        where a higher weight means that word pushes the prediction
        more strongly toward the "flagged" (label 1) class.

        Different model types store their learned weights differently,
        so this function knows how to read all 3 types we might have
        deployed for any given category:
            - Naive Bayes:            feature_log_prob_
            - Logistic Regression:    coef_
            - SVM (Calibrated SVC):   coef_ on the wrapped estimator
        """
        model = self.models[category]
        vectorizer = self.vectorizers[category]
        feature_names = vectorizer.get_feature_names_out()

        weights = None

        if hasattr(model, "coef_"):
            # Logistic Regression (and plain LinearSVC, if ever used
            # without calibration) both expose coef_ directly.
            weights = model.coef_[0]

        elif hasattr(model, "feature_log_prob_"):
            # Naive Bayes: compare how strongly each word is associated
            # with class 1 vs class 0. classes_ tells us which row is
            # which (should be [0, 1] for our binary setup).
            class_1_index = list(model.classes_).index(1)
            class_0_index = list(model.classes_).index(0)
            weights = model.feature_log_prob_[class_1_index] - model.feature_log_prob_[class_0_index]

        elif hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV-wrapped SVM: the underlying
            # LinearSVC (with its coef_) is stored on the first
            # calibrated fold's `.estimator`.
            base_estimator = model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "coef_"):
                weights = base_estimator.coef_[0]

        if weights is None:
            return None

        return dict(zip(feature_names, weights))

    def _calculate_credibility_score(self, flagged_confidences: list[float]) -> float:
        """
        Credibility score: 100 means "no red flags at all". Every
        flagged category pulls it down, proportional to how confident
        the model was. If nothing was flagged, credibility is 100.
        If multiple categories were flagged, we average their impact
        rather than stacking penalties, since a wildly low score for a
        borderline multi-category flag would be misleading.
        """
        if not flagged_confidences:
            return 100.0

        average_flagged_confidence = sum(flagged_confidences) / len(flagged_confidences)
        return round(100 - average_flagged_confidence, 1)

    def _calculate_overall_risk(self, flagged_confidences: list[float]) -> str:
        """
        Overall risk level is based on the single highest confidence
        among all flagged categories - i.e. "how sure are we about the
        worst thing this content was flagged for".
        """
        if not flagged_confidences:
            return "Low"
        return _risk_level_from_confidence(max(flagged_confidences))


# Lets a teammate quickly test this file directly, e.g.:
#   python core/ml_engine.py "Congratulations! You won a free prize, click here now"
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print('Usage: python core/ml_engine.py "some text to analyze"')
    else:
        engine = MLEngine()
        output = engine.predict(sys.argv[1])
        print(json.dumps(output, indent=2))
