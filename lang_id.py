"""
lang_id.py
----------
Token-level Language Identification (LID) for Kannada-Hindi (-English)
code-switched text.

Compared to the Hindi-English case, this is easier for native-script
text: Kannada script (U+0C80-U+0CFF) and Devanagari script
(U+0900-U+097F) occupy different Unicode blocks, so a token written in
its native script can be identified with near-perfect precision by
just checking which block its characters fall in -- no dictionary or
classifier needed.

Latin-script (romanized) tokens are harder, exactly as in the
Hindi-English case, and go through the same fallback strategy:
  1. dictionary lookup (KANNADA_ROMAN_WORDS / HINDI_ROMAN_WORDS / EN)
  2. capitalization heuristic -> named entity
  3. character n-gram classifier trained on the seed lexicons
"""

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from lexicon import KANNADA_ROMAN_WORDS, HINDI_ROMAN_WORDS, ENGLISH_SEED_WORDS

KN, HI, EN, NUM, PUNCT, OTHER = "KN", "HI", "EN", "NUM", "PUNCT", "OTHER"

KANNADA_CHAR = r"\u0C80-\u0CFF"
DEVANAGARI_CHAR = r"\u0900-\u097F"

TOKEN_RE = re.compile(
    rf"[A-Za-z']+|[{KANNADA_CHAR}]+|[{DEVANAGARI_CHAR}]+|[0-9]+|"
    rf"[^\sA-Za-z0-9{KANNADA_CHAR}{DEVANAGARI_CHAR}]"
)
_KANNADA_RE = re.compile(rf"^[{KANNADA_CHAR}]+$")
_DEVANAGARI_RE = re.compile(rf"^[{DEVANAGARI_CHAR}]+$")
_LATIN_RE = re.compile(r"^[A-Za-z']+$")


def tokenize(text: str):
    return TOKEN_RE.findall(text)


class LanguageIdentifier:
    def __init__(self, model_path=None):
        """model_path: optional joblib file from train_models.py (e.g.
        results/model_linear_svm.joblib). If given, it replaces the small
        built-in char n-gram classifier for romanized tokens not found in
        the dictionaries."""
        self._external = None
        if model_path:
            import joblib
            self._external = joblib.load(model_path)
        self._dict_kn = set(KANNADA_ROMAN_WORDS.keys())
        self._dict_hi = set(HINDI_ROMAN_WORDS)
        self._dict_en = set(w.lower() for w in ENGLISH_SEED_WORDS)
        self._clf, self._vec = self._train_fallback_classifier()

    def _train_fallback_classifier(self):
        X_text = list(self._dict_kn) + list(self._dict_hi) + list(self._dict_en)
        y = (
            [KN] * len(self._dict_kn)
            + [HI] * len(self._dict_hi)
            + [EN] * len(self._dict_en)
        )
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        X = vec.fit_transform(X_text)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return clf, vec

    def tag_token(self, token: str) -> str:
        if token.isdigit():
            return NUM

        # Native script: resolved for free via Unicode block, no lookup needed.
        if _KANNADA_RE.match(token):
            return KN
        if _DEVANAGARI_RE.match(token):
            return HI
        if not _LATIN_RE.match(token):
            return PUNCT

        low = token.lower()
        if low in self._dict_kn:
            return KN
        if low in self._dict_hi:
            return HI
        if low in self._dict_en:
            return EN

        if token[0].isupper():
            return OTHER  # heuristic named-entity guess

        if self._external is not None:
            return self._external.predict([low])[0]
        X = self._vec.transform([low])
        return self._clf.predict(X)[0]

    def tag_sentence(self, text: str):
        tokens = tokenize(text)
        return [(tok, self.tag_token(tok)) for tok in tokens]
