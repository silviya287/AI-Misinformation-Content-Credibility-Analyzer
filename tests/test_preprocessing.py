"""
tests/test_preprocessing.py

Basic tests for core/preprocessing.py (Milestone 3).

Run with:
    pytest tests/test_preprocessing.py -v
"""

from core.preprocessing import clean_and_tokenize, tokens_to_text


def test_empty_text_returns_empty_list():
    """No text in, no tokens out - shouldn't crash."""
    assert clean_and_tokenize("") == []
    assert clean_and_tokenize(None) == []


def test_lowercase_conversion():
    """Mixed-case words should end up lowercase."""
    tokens = clean_and_tokenize("HELLO World")
    assert "hello" in tokens
    assert "world" in tokens
    assert "HELLO" not in tokens


def test_url_removal():
    """URLs should be stripped out entirely, not left as tokens."""
    tokens = clean_and_tokenize("Visit http://scam-site.com right now")
    joined = tokens_to_text(tokens)
    assert "http" not in joined
    assert "scam" not in joined  # the whole URL should be gone


def test_punctuation_removal():
    """Punctuation should not survive as its own token."""
    tokens = clean_and_tokenize("Wow!!! Amazing... right?")
    for token in tokens:
        assert token.isalpha()


def test_stopword_removal():
    """Common stopwords like 'the', 'is', 'and' should be filtered out."""
    tokens = clean_and_tokenize("This is a message and the sender wants money")
    for stopword in ["is", "a", "and", "the"]:
        assert stopword not in tokens


def test_lemmatization_reduces_words_to_base_form():
    """Words should be reduced to their dictionary/base form."""
    tokens = clean_and_tokenize("He is running and jumping quickly")
    assert "run" in tokens
    assert "jump" in tokens


def test_tokens_to_text_rejoins_correctly():
    """tokens_to_text should just space-join the list back together."""
    result = tokens_to_text(["click", "free", "prize"])
    assert result == "click free prize"
