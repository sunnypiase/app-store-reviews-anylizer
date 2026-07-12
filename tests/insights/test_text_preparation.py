from app.insights.text_preparation import (
    STOP_WORDS,
    normalize_text,
    stem,
    stem_phrase,
    stemmed_tokens,
    tokenize,
)


def test_normalize_lowercases_and_strips_both_apostrophe_styles():
    assert normalize_text("Don't and Don’t") == "dont and dont"


def test_tokenize_drops_numbers_and_price_fragments():
    # "4.99" must not leak a "99" token — prices are noise, not keywords.
    assert tokenize("They charged me $4.99, then 9.99 more") == ["they", "charged", "me", "then", "more"]


def test_tokenize_drops_single_letter_fragments():
    assert tokenize("I paid a lot") == ["paid", "lot"]


def test_stem_groups_inflectional_variants():
    assert len({stem(t) for t in ("charge", "charged", "charging", "charges")}) == 1
    assert stem("canceled") == stem("cancelled")


def test_stem_phrase_stems_each_word():
    assert stem_phrase("charged twice") == stem_phrase("charging twice")


def test_stemmed_tokens_applies_full_pipeline():
    assert stemmed_tokens("Charging me $4.99!") == {stem("charging"), "me"}


def test_stop_words_cover_apostrophe_less_contractions_and_generic_terms():
    assert {"dont", "cant", "app", "apps"} <= set(STOP_WORDS)
