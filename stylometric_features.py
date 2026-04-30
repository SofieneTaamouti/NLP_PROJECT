import re
import numpy as np
import pandas as pd
import textstat
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm", disable=["ner"])

ENGLISH_STOPWORDS = {
    "a","an","the","and","or","but","if","while","of","at","by","for","with","about",
    "against","between","into","through","during","before","after","above","below",
    "to","from","up","down","in","out","on","off","over","under","again","further",
    "then","once","here","there","when","where","why","how","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own","same",
    "so","than","too","very","can","will","just","don","should","now","is","am","are",
    "was","were","be","been","being","have","has","had","do","does","did","i","you",
    "he","she","it","we","they","me","him","her","us","them","my","your","his","its",
    "our","their","this","that","these","those"
}

PUNCTUATION_MARKS = {
    "period": ".",
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "exclamation": "!",
    "question": "?",
    "double_quote": '"',
    "single_quote": "'",
    "open_paren": "(",
    "close_paren": ")",
    "hyphen": "-",
}

def compute_stylometric_features(text):
    if pd.isna(text):
        text = ""
    text = str(text)

    text_stripped = text.strip()
    n_characters = len(text)
    n_characters_no_spaces = len(re.sub(r"\s+", "", text))

    words = re.findall(r"\b\w+\b", text)
    words_lower = [w.lower() for w in words]
    n_words = len(words)

    sentence_candidates = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentence_candidates if s.strip()]
    sentence_word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    n_sentences = len(sentences)

    word_lengths = [len(w) for w in words]

    mean_word_length = np.mean(word_lengths) if word_lengths else 0
    std_word_length = np.std(word_lengths) if word_lengths else 0

    mean_sentence_length = np.mean(sentence_word_counts) if sentence_word_counts else 0
    std_sentence_length = np.std(sentence_word_counts) if sentence_word_counts else 0

    lexical_diversity = len(set(words_lower)) / n_words if n_words > 0 else 0

    word_counts = Counter(words_lower)
    repeated_word_tokens = sum(count for count in word_counts.values() if count > 1)
    repetition_rate = repeated_word_tokens / n_words if n_words > 0 else 0

    uppercase_share = (
        sum(1 for char in text if char.isupper()) / n_characters_no_spaces
        if n_characters_no_spaces > 0 else 0
    )
    digit_share = (
        sum(1 for char in text if char.isdigit()) / n_characters_no_spaces
        if n_characters_no_spaces > 0 else 0
    )

    stopword_count = sum(1 for w in words_lower if w in ENGLISH_STOPWORDS)
    stopword_share = stopword_count / n_words if n_words > 0 else 0

    punctuation_counts = {
    f"count_{name}": text.count(mark)
    for name, mark in PUNCTUATION_MARKS.items()
    }

    punctuation_rates = {
    f"rate_{name}": text.count(mark) / n_characters if n_characters > 0 else 0
    for name, mark in PUNCTUATION_MARKS.items()
    }

    n_em_dash = text.count("—")
    n_ellipsis = len(re.findall(r"\.\.\.", text))

    doc = nlp(text)

    pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
    n_tokens_spacy = sum(pos_counts.values())

    pos_features = {}
    pos_tags_to_keep = [
        "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
        "ADP", "AUX", "CCONJ", "SCONJ", "PROPN", "NUM"
    ]

    for pos_tag in pos_tags_to_keep:
        pos_features[f"pos_share_{pos_tag.lower()}"] = (
            pos_counts[pos_tag] / n_tokens_spacy if n_tokens_spacy > 0 else 0
        )

    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
    except:
        flesch_reading_ease = np.nan

    try:
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    except:
        flesch_kincaid_grade = np.nan

    try:
        gunning_fog = textstat.gunning_fog(text)
    except:
        gunning_fog = np.nan

    return {
        "n_characters": n_characters,
        "n_characters_no_spaces": n_characters_no_spaces,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "mean_word_length": mean_word_length,
        "std_word_length": std_word_length,
        "mean_sentence_length": mean_sentence_length,
        "std_sentence_length": std_sentence_length,
        "lexical_diversity": lexical_diversity,
        "repetition_rate": repetition_rate,
        "uppercase_share": uppercase_share,
        "digit_share": digit_share,
        "stopword_share": stopword_share,
        "n_em_dash": n_em_dash,
        "has_em_dash": int(n_em_dash > 0),
        "n_ellipsis": n_ellipsis,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog": gunning_fog,
        **punctuation_counts,
        **punctuation_rates,
        **pos_features
    }