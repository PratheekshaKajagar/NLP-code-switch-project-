"""
translator.py
-------------
Translates the Kannada spans of a Kannada-Hindi code-switched sentence
into Hindi. (Hindi spans are already the target language and just pass
through the pipeline, transliterated to Devanagari if romanized -- see
pipeline.py.)

Two backends, matching the Hindi-English system's design:

  1. NMTBackend (recommended, disabled by default here)
     Wraps a pretrained HuggingFace MT model that supports Kannada -> Hindi
     directly, e.g.:
         ai4bharat/indictrans2-indic-indic-1B   (best quality, all 22 Indian
                                                  languages incl. kn<->hi)
         facebook/nllb-200-distilled-600M        (src_lang="kan_Knda",
                                                  tgt_lang="hin_Deva")
     Needs `pip install transformers torch` and internet access to
     download weights on first use.

  2. DictionaryBackend (automatic fallback / offline demo)
     Word-for-word gloss lookup using the curated Kannada->Hindi
     lexicon. No downloads required, but no grammar/reordering and
     limited vocabulary -- exists so the pipeline runs end-to-end with
     zero setup.
"""

from lexicon import build_kannada_script_dictionary, KANNADA_NATIVE_TO_HINDI
from transliterator import to_kannada
from lang_id import _KANNADA_RE


def normalize_to_kannada_script(word: str) -> str:
    """Return the Kannada-script form of a token, transliterating it
    first if it was typed in romanized Latin script."""
    if _KANNADA_RE.match(word):
        return word
    return to_kannada(word)


class DictionaryBackend:
    name = "dictionary-fallback"

    def __init__(self):
        # Direct native-script entries take priority (they're hand-typed,
        # so always spelled the way a Kannada speaker would write them).
        # ITRANS-derived entries fill in additional coverage for
        # romanized input, but can't override a native match.
        self.kn_dict = build_kannada_script_dictionary(to_kannada)
        self.kn_dict.update(KANNADA_NATIVE_TO_HINDI)

    def translate_span(self, kn_words):
        glosses = []
        for w in kn_words:
            kn_script = normalize_to_kannada_script(w)
            glosses.append(self.kn_dict.get(kn_script, kn_script))
        return " ".join(glosses)


class NMTBackend:
    """Wraps a pretrained HuggingFace seq2seq MT model for Kannada ->
    Hindi translation. Requires:
        pip install transformers torch sentencepiece
    plus internet access to download model weights on first use.
    """

    name = "nmt (ai4bharat/indictrans2-indic-indic-1B)"

    def __init__(self, model_name="ai4bharat/indictrans2-indic-indic-1B"):
        from transformers import pipeline  # imported lazily
        self._pipe = pipeline(
            "translation",
            model=model_name,
            src_lang="kan_Knda",
            tgt_lang="hin_Deva",
            trust_remote_code=True,
        )

    def translate_span(self, kn_words):
        kn_sentence = " ".join(normalize_to_kannada_script(w) for w in kn_words)
        out = self._pipe(kn_sentence)
        return out[0]["translation_text"]


def load_best_available_backend():
    try:
        return NMTBackend()
    except Exception:
        return DictionaryBackend()
