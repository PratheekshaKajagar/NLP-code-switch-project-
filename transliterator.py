"""
transliterator.py
------------------
Script conversion helpers, using the `indic-transliteration` package
(ITRANS phonetic romanization scheme; pure Python, offline, no model
download).

Two conversions are needed for the Kannada<->Hindi pipeline:
  - Romanized Kannada  -> Kannada script   (to_kannada)
  - Romanized Hindi     -> Devanagari       (to_devanagari)

Note: converting KANNADA SCRIPT directly to DEVANAGARI SCRIPT (a
script-to-script transliteration) is also possible with this library,
but it only maps sounds/letters -- it does NOT translate meaning, since
Kannada and Hindi are different languages with different vocabularies.
That's why actual Kannada->Hindi *translation* (translator.py) is a
separate, dictionary/NMT-based step, not just a transliteration call.
"""

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _translit


def to_kannada(word: str) -> str:
    """Transliterate a romanized Kannada word/phrase to Kannada script."""
    return _translit(word.lower(), sanscript.ITRANS, sanscript.KANNADA)


def to_devanagari(word: str) -> str:
    """Transliterate a romanized Hindi word/phrase to Devanagari."""
    return _translit(word.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
