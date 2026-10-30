"""
demo.py
-------
Run this to see the Kannada -> Hindi Code-Switching Translation System
in action.

    python3 demo.py
"""

from pipeline import CodeSwitchTranslator

SAMPLE_SENTENCES = [
    # romanized Kannada + romanized Hindi mixed
    "Naanu office hogutthini, thoda late aagutte.",
    "Nanna mane bahut door hai, isliye bega horaduttini.",
    # native Kannada script + native Devanagari mixed
    "ನಾನು ಇಂದು ಮನೆ में हूँ.",
    # romanized Kannada + English
    "Meeting elli ide? Office alli aagide.",
    "Naale exam ide, swalpa nervous aagide.",
]


def main():
    translator = CodeSwitchTranslator(verbose=False)
    print(f"Translation backend in use: {translator.backend.name}\n")

    for sent in SAMPLE_SENTENCES:
        print("Input (Kannada-Hindi mixed):", sent)
        print("Hindi output               :", translator.translate(sent))
        print()

    print("-" * 60)
    print("Token-level language ID breakdown for one example:")
    print(SAMPLE_SENTENCES[0])
    translator.explain(SAMPLE_SENTENCES[0])


if __name__ == "__main__":
    main()
