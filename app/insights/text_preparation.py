"""Data preparation for keyword and theme mining.

Every piece of review text goes through the same explicit pipeline before any
counting or scoring happens:

1. **Normalize** (`normalize_text`) — lowercase and strip apostrophes so
   contractions arrive as single tokens ("Don't" -> "dont") instead of the
   junk fragments a default tokenizer would produce ("don", "t").
2. **Tokenize** (`TOKEN_PATTERN` / `tokenize`) — alphabetic words of two or
   more letters only. Digits never become tokens, so price fragments such as
   the "99" in "$4.99" cannot surface as keywords. The pattern is shared
   verbatim with ``TfidfVectorizer(token_pattern=...)`` so sklearn tokenizes
   exactly the same way this module does.
3. **Drop stopwords** (`STOP_WORDS`) — sklearn's English list, plus the
   apostrophe-less contraction forms produced by step 1, plus corpus-generic
   terms ("app", "apps", "application") that carry no discriminative signal
   in a corpus that is, by definition, entirely reviews of one app. The app's
   own name is deliberately left in, since it can still co-occur with
   genuinely distinctive complaints.
4. **Stem** (`stem` / `stem_phrase` / `stemmed_tokens`) — light suffix
   stripping used purely as a grouping key, so inflections of one complaint
   ("charge"/"charged"/"charging"/"charges", "canceled"/"cancelled") share a
   single key. Stems are never shown to users; display always uses a real
   surface form.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Step 2 — letters-only tokens, two characters minimum (see module docstring).
TOKEN_PATTERN = r"[a-z]{2,}"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

# Step 3 — apostrophes are stripped during normalization (step 1), so
# contractions arrive as single tokens ("dont", "cant"); these apostrophe-less
# forms need their own stopword entries on top of sklearn's English list.
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
    """Step 1 — lowercase and drop apostrophes so "don't"/"don’t" become
    "dont" rather than splitting into meaningless fragments at tokenization
    time."""
    return text.lower().replace("'", "").replace("’", "")


def tokenize(text: str) -> list[str]:
    """Steps 1–2 — normalize, then extract letters-only tokens."""
    return _TOKEN_RE.findall(normalize_text(text))


def stem(token: str) -> str:
    """Step 4 — light suffix-stripping stem (see module docstring)."""
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
    """The full pipeline (steps 1, 2, 4) applied to raw review text."""
    return {stem(token) for token in tokenize(text)}
