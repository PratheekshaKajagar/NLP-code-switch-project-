"""
data_builder.py
---------------
Builds the labelled dataset used for MODEL BUILDING & TRAINING of the
token-level language identifier (KN / HI / EN, romanized text).

Why a builder at all?  The original project only shipped small seed
lexicons (no labelled corpus).  To train and compare real classifiers we
need labelled tokens, so this script assembles them from:

  KN : your lexicon.py  (romanized keys + native-script keys romanized)
       + Kannada case-suffix variants (-annu, -alli, -inda, -ge, -ina)
       + noisy-spelling variants (aa->a, ee->e, th->t, ...)
  HI : your lexicon.py  HINDI_ROMAN_WORDS
       + top-N Hindi words from the `wordfreq` package, romanized
       + noisy-spelling variants
  EN : your lexicon.py  ENGLISH_SEED_WORDS
       + top-N English words from `wordfreq`

Important honesty note: this is a *synthetic / semi-synthetic* dataset.
Kannada is the weak spot (only your ~280 curated words + variants).
If you have real annotated Kanglish/Hinglish tokens, drop them in a CSV
with columns `token,label,group` and pass --data to train_models.py.

Train/test split is done by `group` (the base word), so a word and its
spelling/suffix variants are never split across train and test.
"""

import argparse
import os
import random
import unicodedata

import pandas as pd

from lexicon import (
    KANNADA_ROMAN_WORDS,
    KANNADA_NATIVE_TO_HINDI,
    HINDI_ROMAN_WORDS,
    ENGLISH_SEED_WORDS,
)

SEED = 42
KN, HI, EN = "KN", "HI", "EN"

KN_SUFFIXES = ["annu", "alli", "inda", "ge"]


# ----------------------------------------------------------------------
# romanization helpers
# ----------------------------------------------------------------------
def _strip_diacritics(s: str) -> str:
    s = s.replace("ṁ", "n").replace("ṃ", "n").replace("ṅ", "n").replace("ñ", "n")
    s = s.replace("ś", "sh").replace("ṣ", "sh").replace("ḥ", "h")
    s = s.replace("ṭ", "t").replace("ḍ", "d").replace("ṇ", "n").replace("ḷ", "l")
    s = s.replace("ṛ", "ri").replace("ṝ", "ri")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _romanize(native: str, source_script: str, drop_schwa: bool) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    iast = transliterate(native, source_script, sanscript.IAST)
    # IAST keeps long vowels as macron letters -> make them 'aa','ee','oo'
    iast = (iast.replace("ā", "aa").replace("ī", "ee").replace("ū", "oo")
                .replace("ē", "e").replace("ō", "o"))
    r = _strip_diacritics(iast)
    # approximate Hindi schwa deletion: "ghara" -> "ghar"
    if drop_schwa and len(r) > 3 and r.endswith("a") and not r.endswith("aa"):
        r = r[:-1]
    return r


def romanize_hindi(devanagari: str) -> str:
    from indic_transliteration import sanscript
    return _romanize(devanagari, sanscript.DEVANAGARI, drop_schwa=True)


def romanize_kannada(kannada: str) -> str:
    from indic_transliteration import sanscript
    return _romanize(kannada, sanscript.KANNADA, drop_schwa=False)


def add_kn_suffix(word: str, suffix: str) -> str:
    """Attach a Kannada case suffix with a plausible sandhi:
    final -u is dropped, other final vowels take a glide (mane+alli -> maneyalli)."""
    if word.endswith("u"):
        return word[:-1] + suffix
    if word[-1] in "aeio":
        return word + "y" + suffix
    return word + suffix


def spelling_variants(word: str, rng: random.Random, k: int = 2):
    """Mimic the non-standard romanization people really type."""
    rules = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("dh", "d"),
             ("kh", "k"), ("sh", "s"), ("v", "w"), ("tt", "t"), ("ll", "l"),
             ("nn", "n"), ("dd", "d")]
    out = set()
    for _ in range(k * 3):
        w = word
        for a, b in rng.sample(rules, k=2):
            if a in w:
                w = w.replace(a, b)
        if w != word and len(w) > 1:
            out.add(w)
        if len(out) >= k:
            break
    return list(out)


# ----------------------------------------------------------------------
# per-language base-word collection
# ----------------------------------------------------------------------
def kannada_bases():
    words = set(k.lower() for k in KANNADA_ROMAN_WORDS)
    for native in KANNADA_NATIVE_TO_HINDI:
        try:
            words.add(romanize_kannada(native))
        except Exception:
            pass
    return sorted(w for w in words if w.isalpha() and len(w) >= 2)


def hindi_bases(top_n):
    words = set(w.lower() for w in HINDI_ROMAN_WORDS)
    try:
        from wordfreq import top_n_list
        for dev in top_n_list("hi", top_n):
            if not all("\u0900" <= c <= "\u097F" for c in dev):
                continue
            r = romanize_hindi(dev)
            if r.isalpha() and len(r) >= 2:
                words.add(r)
    except ImportError:
        print("[warn] wordfreq not installed -> Hindi limited to lexicon words")
    return sorted(words)


def english_bases(top_n):
    words = set(w.lower() for w in ENGLISH_SEED_WORDS)
    try:
        from wordfreq import top_n_list
        for w in top_n_list("en", top_n):
            if w.isascii() and w.isalpha() and len(w) >= 2:
                words.add(w.lower())
    except ImportError:
        print("[warn] wordfreq not installed -> English limited to lexicon words")
    return sorted(words)


# ----------------------------------------------------------------------
def build_dataset(top_n_hi=1800, top_n_en=1800, test_size=0.2):
    rng = random.Random(SEED)
    kn, hi, en = kannada_bases(), hindi_bases(top_n_hi), english_bases(top_n_en)

    # Words that exist in more than one language (e.g. "to", "me", "kam")
    # are genuinely ambiguous out of context -> remove to avoid label noise.
    kn_s, hi_s, en_s = set(kn), set(hi), set(en)
    ambiguous = (kn_s & hi_s) | (kn_s & en_s) | (hi_s & en_s)
    kn = [w for w in kn if w not in ambiguous]
    hi = [w for w in hi if w not in ambiguous]
    en = [w for w in en if w not in ambiguous]

    # Cap the big classes so Kannada (the small one) isn't drowned out.
    rng.shuffle(hi); rng.shuffle(en)
    cap = 1200
    hi, en = sorted(hi[:cap]), sorted(en[:cap])

    rows, split_info = [], {}
    for label, bases in ((KN, kn), (HI, hi), (EN, en)):
        bases = bases[:]
        rng.shuffle(bases)
        n_test = int(len(bases) * test_size)
        test_bases = set(bases[:n_test])
        for b in bases:
            split = "test" if b in test_bases else "train"
            forms = {b}
            if label == KN:
                if len(b) <= 8:  # long words are mostly inflected verbs already
                    forms |= {add_kn_suffix(b, s) for s in KN_SUFFIXES}
                forms |= set(spelling_variants(b, rng, 2))
            elif label == HI:
                forms |= set(spelling_variants(b, rng, 1))
            for f in forms:
                rows.append((f, label, b, split))
        split_info[label] = (len(bases), n_test)

    df = pd.DataFrame(rows, columns=["token", "label", "group", "split"])
    df = df.drop_duplicates(subset=["token", "split"]).drop_duplicates(subset=["token"])
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True), ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/token_lid_dataset.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df, amb = build_dataset()
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} tokens -> {args.out}")
    print(f"(removed {len(amb)} cross-language ambiguous words)")
    print(df.groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    main()
