"""
pipeline.py
-----------
End-to-end pipeline: Kannada-Hindi(-English) code-switched text -> Hindi.

    raw text
        -> tokenize
        -> token-level language ID     (lang_id.py: KN / HI / EN / NUM / PUNCT / OTHER)
        -> group consecutive same-tag tokens into spans
        -> KN spans:  translate -> Hindi          (translator.py)
        -> HI spans:  transliterate romanized -> Devanagari, else pass through
        -> EN/NUM/OTHER spans: pass through unchanged
        -> reassemble into a single Hindi(-ish) sentence

Unlike the Hindi-English system, here the SOURCE has two languages that
both need different handling: Kannada must be *translated* (different
language family), while Hindi only needs *script normalization*
(romanized -> Devanagari) since it's already the target language.
"""

from dataclasses import dataclass
from typing import List

from lang_id import LanguageIdentifier, KN, HI, EN, NUM, PUNCT, OTHER, _LATIN_RE
from transliterator import to_devanagari
from translator import load_best_available_backend


@dataclass
class Span:
    tokens: List[str]
    tag: str


def group_into_spans(tagged_tokens):
    spans = []
    for tok, tag in tagged_tokens:
        if spans and spans[-1].tag == tag:
            spans[-1].tokens.append(tok)
        else:
            spans.append(Span(tokens=[tok], tag=tag))
    return spans


class CodeSwitchTranslator:
    def __init__(self, backend=None, verbose=False):
        self.lid = LanguageIdentifier()
        self.backend = backend or load_best_available_backend()
        self.verbose = verbose

    def translate(self, text: str) -> str:
        tagged = self.lid.tag_sentence(text)
        spans = group_into_spans(tagged)

        if self.verbose:
            print("LID spans:", [(s.tokens, s.tag) for s in spans])

        out_pieces = []
        for span in spans:
            if span.tag == KN:
                out_pieces.append(self.backend.translate_span(span.tokens))
            elif span.tag == HI:
                # Already the target language: just normalize script.
                normalized = [
                    to_devanagari(w) if _LATIN_RE.match(w) else w
                    for w in span.tokens
                ]
                out_pieces.append(" ".join(normalized))
            elif span.tag == PUNCT:
                if out_pieces:
                    out_pieces[-1] = out_pieces[-1] + "".join(span.tokens)
                else:
                    out_pieces.append("".join(span.tokens))
            else:  # EN, NUM, OTHER pass through unchanged
                out_pieces.append(" ".join(span.tokens))

        result = " ".join(p for p in out_pieces if p)
        return self._cleanup(result)

    @staticmethod
    def _cleanup(text: str) -> str:
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        text = text.replace(" !", "!")
        return text

    def explain(self, text: str):
        tagged = self.lid.tag_sentence(text)
        width = max((len(t) for t, _ in tagged), default=1)
        for tok, tag in tagged:
            print(f"  {tok:<{width}}  ->  {tag}")
