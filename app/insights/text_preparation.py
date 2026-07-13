"""Shared text pipeline for keyword/theme mining: normalize -> letters-only tokens
-> stopwords -> light stemming, kept identical between this module and sklearn."""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Letters-only tokens keep digits (e.g. the "99" in "$4.99") out of the vocabulary.
TOKEN_PATTERN = r"[a-z]{2,}"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

# Normalization strips apostrophes, so contractions need apostrophe-less stopword entries.
_CONTRACTION_STOP_WORDS = {
    "aint", "arent", "cant", "cannot", "couldnt", "couldve", "didnt", "doesnt",
    "dont", "hadnt", "hasnt", "havent", "hes", "id", "ill", "im", "isnt",
    "its", "ive", "shes", "shouldnt", "shouldve", "thats", "theres", "theyd",
    "theyll", "theyre", "theyve", "wasnt", "werent", "whats", "wont",
    "wouldnt", "wouldve", "youd", "youll", "youre", "youve",
}

_CORPUS_GENERIC_STOP_WORDS = {"app", "apps", "application"}

STOP_WORDS = list(ENGLISH_STOP_WORDS | _CONTRACTION_STOP_WORDS | _CORPUS_GENERIC_STOP_WORDS)


def normalize_text(text: str) -> str:
    """Lowercase and drop apostrophes so "don't" becomes "dont", not "don"/"t"."""
    return text.lower().replace("'", "").replace("’", "")


def tokenize(text: str) -> list[str]:
    """Normalize, then extract letters-only tokens."""
    return _TOKEN_RE.findall(normalize_text(text))


def stem(token: str) -> str:
    """Light suffix-stripping stem, used only as a grouping key, never displayed."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) >= 4 and token[-1] == token[-2]:
        token = token[:-1]
    if len(token) >= 4 and token.endswith("e"):
        token = token[:-1]
    return token


def stem_phrase(phrase: str) -> str:
    return " ".join(stem(token) for token in phrase.split())


def stemmed_tokens(text: str) -> set[str]:
    """The full pipeline (normalize, tokenize, stem) applied to raw review text."""
    return {stem(token) for token in tokenize(text)}
