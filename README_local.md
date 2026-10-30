# Code-Switching Translation System (Kannada → Hindi)

Translates Kannada-Hindi code-switched text (romanized or native script,
optionally with some English mixed in) into Hindi.

Example:
```
Input : Naanu office hogutthini, thoda late aagutte.
Output: मैं ऑफिस जा रहा हूँ, थोद लते आगुत्ते.
```

## How this differs from the Hindi-English version

| | Hindi-English | Kannada-Hindi |
|---|---|---|
| Scripts involved | Both usually romanized (Latin) | Kannada script + Devanagari (often native), or romanized |
| Language ID | Dictionary + classifier only | **Unicode script detection** (free, high-precision) for native text, dictionary + classifier fallback for romanized text |
| Non-target language handling | Translate Hindi → English | **Translate** Kannada → Hindi (different language family) |
| Target-language handling | N/A (target = English, not in source) | Hindi spans just need **script normalization** (romanized → Devanagari), not translation |

Because Kannada (U+0C80–U+0CFF) and Devanagari (U+0900–U+097F) sit in
different Unicode blocks, native-script text can be language-tagged
almost perfectly with no dictionary at all — this makes the Kannada-Hindi
case *more* robust than Hindi-English whenever people type in native
script, which is common for Kannada speakers.

## Architecture

```
raw text
   │
   ▼
1. Tokenize (Unicode-aware: Latin / Kannada block / Devanagari block)
   │
   ▼
2. Token-level Language ID
   - native Kannada script -> KN   (Unicode block match, no lookup needed)
   - native Devanagari script -> HI (Unicode block match, no lookup needed)
   - romanized text -> dictionary lookup (KN/HI/EN) + char n-gram classifier fallback
   │
   ▼
3. Group consecutive same-tag tokens into spans
   │
   ▼
4a. KN spans  -> normalize to Kannada script (transliterate if romanized)
              -> TRANSLATE to Hindi (dictionary fallback or NMT backend)
4b. HI spans  -> normalize to Devanagari (transliterate if romanized)
              -> pass through (already the target language)
4c. EN/NUM/OTHER spans -> pass through unchanged
   │
   ▼
5. Reassemble into final Hindi sentence
```

Files:
- `lexicon.py` — curated Kannada→Hindi dictionary + Hindi/English seed word lists (for LID only)
- `lang_id.py` — token-level language identification (Unicode script + classifier)
- `transliterator.py` — romanized Kannada → Kannada script, romanized Hindi → Devanagari
- `translator.py` — Kannada → Hindi translation (dictionary backend + optional NMT backend)
- `pipeline.py` — orchestrates the full pipeline
- `demo.py` — runnable example

## Quick start

```bash
pip install -r requirements.txt
python3 demo.py
```

Runs fully offline with the dictionary backend — no downloads needed.

## Getting production-quality translation

The dictionary backend is a small (~100 word) hand-curated gloss table,
included so the pipeline runs with zero setup. For real translation
quality, enable the NMT backend:

```bash
pip install transformers torch sentencepiece
```

```python
from pipeline import CodeSwitchTranslator
from translator import NMTBackend

translator = CodeSwitchTranslator(backend=NMTBackend())
print(translator.translate("Naanu office hogutthini, thoda late aagutte."))
```

### Model options for the NMT backend

| Model | Notes |
|---|---|
| `ai4bharat/indictrans2-indic-indic-1B` | Best quality for Kannada↔Hindi directly (default in `translator.py`); covers all 22 scheduled Indian languages |
| `facebook/nllb-200-distilled-600M` | General multilingual fallback; use `src_lang="kan_Knda"`, `tgt_lang="hin_Deva"` |

Note: unlike the Hindi-English system (where `Helsinki-NLP/opus-mt-hi-en`
is a small bilingual model), there's no widely-used small bilingual
Kannada→Hindi model, so IndicTrans2's indic-to-indic model or NLLB-200
are the practical options.

## Known limitations

- The dictionary backend covers ~100 curated Kannada words; anything
  outside it passes through as (approximate) transliterated Kannada
  script rather than being translated. Enable the NMT backend for real
  coverage.
- English tokens not in the small seed list can occasionally be
  misclassified as Kannada or Hindi by the fallback classifier since
  it only sees isolated tokens, not sentence context.
- Kannada→Kannada-script transliteration of free-form romanized input
  is approximate (romanization isn't standardized).
- No handling of intra-word code-switching or three-way mixing beyond
  simple English loanwords/named entities.


---

## Model building & training (added)

Task: token-level language ID (KN / HI / EN) on romanized tokens - the
hard case in this pipeline. Features: **TF-IDF char 1-4-grams**.

| # | Model | Features |
|---|---|---|
| 1 | Logistic Regression | TF-IDF |
| 2 | Linear SVM (calibrated) | TF-IDF |
| 3 | Random Forest | TF-IDF |
| 4 | BERT (mBERT, fine-tuned) | WordPiece |

```bash
pip install -r requirements.txt
python data_builder.py     # builds data/token_lid_dataset.csv
python train_models.py     # trains 3 TF-IDF models -> results/

# BERT (needs torch + transformers + internet for the weights)
pip install torch transformers
python bert_model.py       # or: --model google/muril-base-cased --epochs 5
python train_models.py     # re-run to merge BERT into the comparison
```

Everything lands in `results/`: per-model confusion matrix and ROC PNGs,
classification reports, `model_comparison.csv/.png`, `roc_all_models.png`,
`cv_results.csv` (5-fold grouped CV), and `errors_<model>.csv`.

Use a trained model inside the pipeline:
`LanguageIdentifier(model_path="results/model_linear_svm.joblib")`.

Caveats: the bundled dataset is semi-synthetic (your lexicon + wordfreq
words + generated suffix/spelling variants; train/test split by base
word so variants never leak). Kannada is the smallest class (~280 base
words). Use `--data your.csv` (columns `token,label[,group]`) with real
annotated tokens for trustworthy numbers.
