"""
core/preprocessing.py

Module A - NLP Preprocessing (Milestone 3).

This file has ONE job: take a plain text string (from
core.input_handler.extract_text) and turn it into a clean list of
tokens (words) that a TF-IDF vectorizer and ML model can understand.

This exact same function will be used in TWO places later:
    1. training/train_models.py  - to clean the training datasets
    2. core/ml_engine.py         - to clean text at prediction time
Using the same function in both places is important - if training and
prediction clean text differently, the model's predictions become
unreliable.

Public function every other module should use:
    clean_and_tokenize(text: str) -> list[str]
"""

import re
import string

import os
import sys

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# When this app is packaged into a standalone executable (PyInstaller),
# the NLTK data files are bundled alongside it rather than downloaded
# at runtime - this keeps the packaged app fully offline. PyInstaller
# extracts bundled files to a temporary folder pointed to by
# sys._MEIPASS; we tell NLTK to look there first if it exists.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    nltk.data.path.insert(0, os.path.join(sys._MEIPASS, "nltk_data"))


def _ensure_nltk_data():
    """
    Makes sure the NLTK data files we need (tokenizer, stopword list,
    lemmatizer dictionary, POS tagger) are downloaded. This only
    downloads once - if the data is already there, it does nothing.

    We call this once, automatically, the first time this module is
    imported, so teammates don't have to remember to run a separate
    setup script.
    """
    required_packages = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for lookup_path, package_name in required_packages:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            print(f"[preprocessing] Downloading NLTK data: {package_name} ...")
            nltk.download(package_name, quiet=True)


# Run the check once when this file is first imported.
_ensure_nltk_data()

# Build these ONCE at import time (not inside the function) since
# creating them repeatedly for every call would be slow.
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Matches http://... and https://... links so we can strip them out.
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def _get_wordnet_pos(nltk_pos_tag: str):
    """
    WordNetLemmatizer needs to know whether a word is a noun, verb,
    adjective, or adverb to lemmatize it correctly (e.g. "running" as
    a VERB -> "run", but as a NOUN it would stay "running").

    NLTK's pos_tag() gives tags like 'VBG', 'NN', 'JJ' etc. This
    function maps those to the simpler noun/verb/adjective/adverb
    categories that WordNetLemmatizer expects. Defaults to noun if we
    don't recognize the tag, since noun is WordNet's default anyway.
    """
    if nltk_pos_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_pos_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_and_tokenize(text: str) -> list[str]:
    """
    Cleans raw text and returns a list of clean, lemmatized tokens.

    Steps (in order):
        1. Lowercase everything
        2. Remove URLs
        3. Remove punctuation
        4. Tokenize (split into individual words)
        5. Remove stopwords (common words like "the", "is", "and")
        6. Lemmatize (reduce words to their base/dictionary form,
           e.g. "running" -> "run", "better" -> "good")

    Example:
        >>> clean_and_tokenize("Click HERE now!! http://scam.com WON A PRIZE")
        ['click', '0won', 'prize']   # (illustrative - stopwords/URLs gone)
    """
    if not text:
        return []

    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove URLs
    text = _URL_PATTERN.sub(" ", text)

    # Step 3: remove punctuation (replace with a space so words don't
    # get glued together, e.g. "prize.click" -> "prize click")
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Step 4: tokenize into individual words
    tokens = word_tokenize(text)

    # Step 5: drop stopwords and anything that isn't a real alphabetic
    # word (numbers, leftover symbols, single letters from splitting)
    # BEFORE lemmatizing - no point tagging/lemmatizing words we're
    # going to throw away anyway.
    candidate_words = [
        token for token in tokens
        if token.isalpha() and token not in _STOPWORDS and len(token) > 1
    ]

    if not candidate_words:
        return []

    # Step 6: lemmatize using each word's part of speech for accuracy
    # (e.g. "running" -> "run" instead of staying "running").
    pos_tags = nltk.pos_tag(candidate_words)
    clean_tokens = [
        _LEMMATIZER.lemmatize(word, _get_wordnet_pos(pos_tag))
        for word, pos_tag in pos_tags
    ]

    return clean_tokens


def tokens_to_text(tokens: list[str]) -> str:
    """
    Small helper: joins a token list back into a single space-separated
    string. TF-IDF's TfidfVectorizer expects strings, not token lists,
    so this is what training/prediction code will call right after
    clean_and_tokenize().
    """
    return " ".join(tokens)


# Lets a teammate quickly test this file directly, e.g.:
#   python core/preprocessing.py "Click HERE now!! http://scam.com WON A PRIZE"
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python core/preprocessing.py "some text to clean"')
    else:
        sample_text = sys.argv[1]
        result_tokens = clean_and_tokenize(sample_text)
        print("--- Original Text ---")
        print(sample_text)
        print("--- Clean Tokens ---")
        print(result_tokens)
        print("--- Rejoined for TF-IDF ---")
        print(tokens_to_text(result_tokens))
