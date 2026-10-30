"""
lexicon.py
----------
Curated seed data for the Code-Switching Translation System:
Kannada <-> Hindi, output language = Hindi.

Unlike the Hindi-English case, Kannada and Hindi normally use two
DIFFERENT native scripts (Kannada script vs Devanagari), so a lot of
language identification can be done for free via Unicode script
detection (see lang_id.py). The dictionaries below are only needed for
two things:
  1. Handling *romanized* Kannada/Hindi ("Kanglish"/"Hinglish" typed in
     Latin script), where script detection doesn't help.
  2. Actually TRANSLATING Kannada words into Hindi. Kannada (Dravidian)
     and Hindi (Indo-Aryan) are unrelated languages, so unlike
     Hindi->Devanagari transliteration, Kannada->Hindi requires real
     lexical translation, not just a script change.

As before, this is a small, hand-curated demo lexicon. Replace it with
a large parallel dictionary or a trained NMT model for production use.
"""

# ---------------------------------------------------------------------------
# 1. Common ROMANIZED Kannada words -> Hindi gloss (Devanagari).
#    Keys are lowercase ITRANS-ish romanizations.
# ---------------------------------------------------------------------------
KANNADA_ROMAN_WORDS = {
    # pronouns
    "naanu": "मैं", "naavu": "हम", "neenu": "तुम", "neevu": "आप",
    "avanu": "वह", "avalu": "वह", "avaru": "वे", "adu": "वह", "idu": "यह",
    "ivu": "ये", "avu": "वे", "nanna": "मेरा", "ninna": "तुम्हारा",
    "avara": "उनका", "namma": "हमारा", "yaaru": "कौन",

    # question words
    "enu": "क्या", "yenu": "क्या", "yaake": "क्यों", "eeke": "क्यों",
    "elli": "कहाँ", "eshtu": "कितना", "hege": "कैसे", "yaavaga": "कब",
    "yaava": "कौन सा",

    # verbs / copulas
    "ide": "है", "iddu": "है", "iddare": "हैं", "iratte": "होगा",
    "irutte": "होगा", "madu": "करो", "madi": "करो", "maadu": "करो",
    "madtha": "कर रहा", "madutta": "कर रहा", "madidenu": "किया",
    "madide": "किया", "barutthene": "आ रहा हूँ", "baruttini": "आ रहा हूँ",
    "banda": "आया", "bandalu": "आयी", "hogutthini": "जा रहा हूँ",
    "hogide": "गया", "hoglaikke": "जाने के लिए", "gothilla": "पता नहीं",
    "gottu": "पता है", "beku": "चाहिए", "bekagilla": "नहीं चाहिए",
    "nodu": "देखो", "nodidenu": "देखा", "helu": "बोलो", "helidenu": "बोला",
    "kelu": "पूछो", "keliddenu": "पूछा", "kodu": "दो", "kottu": "दिया",
    "tagondu": "लेकर", "tegedu": "लेकर", "sikkitu": "मिला",

    # adjectives / adverbs
    "chennagi": "अच्छे से", "chennagide": "अच्छा है", "olleyadu": "अच्छी बात",
    "kettu": "बुरा", "ketta": "बुरा", "swalpa": "थोड़ा", "jasti": "ज्यादा",
    "kammi": "कम", "dodda": "बड़ा", "chikka": "छोटा", "hosa": "नया",
    "haleya": "पुराना", "bega": "जल्दी", "nidhana": "धीरे",

    # nouns
    "mane": "घर", "kelasa": "काम", "hana": "पैसा", "ooru": "गाँव",
    "shaale": "स्कूल", "neeru": "पानी", "oota": "खाना", "dina": "दिन",
    "naale": "कल", "indu": "आज", "nenne": "कल", "gante": "घंटा",
    "samaya": "समय", "snehita": "दोस्त", "kelasagara": "कर्मचारी",
    "railu": "ट्रेन", "office": "ऑफिस", "meeting": "मीटिंग",

    # connectors / particles
    "mattu": "और", "adare": "लेकिन", "yaakandre": "क्योंकि",
    "konege": "आखिरकार", "innu": "अभी भी",
    "innondu": "एक और", "kooda": "भी", "matte": "फिर से",

    # numbers 1-10
    "ondu": "एक", "eradu": "दो", "muuru": "तीन", "naalku": "चार",
    "aidu": "पांच", "aaru": "छह", "elu": "सात", "entu": "आठ",
    "ombattu": "नौ", "hattu": "दस",

    # days of the week
    "somavaara": "सोमवार", "mangalavaara": "मंगलवार", "budhavaara": "बुधवार",
    "guruvaara": "गुरुवार", "shukravaara": "शुक्रवार", "shanivaara": "शनिवार",
    "bhaanuvaara": "रविवार",

    # family
    "appa": "पिता", "amma": "माँ", "anna": "बड़ा भाई", "akka": "बड़ी बहन",
    "tamma": "छोटा भाई", "tangi": "छोटी बहन", "maga": "बेटा",
    "magalu": "बेटी", "ganda": "पति", "hendati": "पत्नी",

    # greetings / politeness
    "namaskaara": "नमस्ते", "dhanyavaada": "धन्यवाद", "swaagata": "स्वागत",
    "kshamisi": "माफ़ कीजिए",

    # more verbs
    "tinnu": "खाना", "kudi": "पीना", "odu": "पढ़ना", "bare": "लिखना",
    "malagu": "सोना", "eddelu": "उठना", "aadu": "खेलना", "haadu": "गाना",

    # more adjectives
    "sundara": "सुंदर", "bisi": "गरम", "tannage": "ठंडा", "hasi": "कच्चा",
    "sulabha": "आसान", "kashta": "मुश्किल",

    # colors
    "kempu": "लाल", "bili": "सफेद", "kappu": "काला", "hasiru": "हरा",
    "neeli": "नीला", "haladi": "पीला",

    # food
    "hannu": "फल", "tarakaari": "सब्ज़ी",
    "haalu": "दूध", "sakkare": "चीनी", "uppu": "नमक",
}

# ---------------------------------------------------------------------------
# 2. Common ROMANIZED Hindi words. Reused only for the language-ID stage
#    (so the classifier can tell romanized Hindi apart from romanized
#    Kannada and English). Since the target language IS Hindi, these
#    tokens are NOT translated -- only transliterated to Devanagari.
# ---------------------------------------------------------------------------
HINDI_ROMAN_WORDS = {
    "main", "mai", "hum", "tum", "tu", "aap", "wo", "voh", "yeh", "ye",
    "mera", "meri", "tera", "teri", "hamara", "unka", "kya", "kaun",
    "kab", "kahan", "kaise", "kyun", "kyu", "kitna", "hai", "hain",
    "tha", "thi", "the", "ho", "hoga", "hogi", "hoon", "karna", "karta",
    "karti", "karte", "kiya", "kar", "karo", "raha", "rahi", "rahe",
    "chahiye", "dena", "diya", "lena", "liya", "jana", "gaya", "gayi",
    "aana", "aaya", "aayi", "dekhna", "dekha", "bolna", "bola", "kehna",
    "milna", "mila", "pata", "samajh", "accha", "acha", "bura", "theek",
    "thik", "sahi", "galat", "bahut", "bohot", "zyada", "kam", "thoda",
    "abhi", "aaj", "kal", "subah", "shaam", "raat", "din", "ghar",
    "kaam", "paisa", "paise", "khana", "paani", "dost", "log", "baat",
    "aur", "ya", "lekin", "magar", "par", "agar", "toh", "to", "kyunki",
    "jab", "tab", "sirf", "bhi", "nahi", "nahin", "mat", "se", "ka",
    "ki", "ke", "ko", "mein", "me", "pe", "tak", "liye",
}

# ---------------------------------------------------------------------------
# 3. English seed words, for the same reason as above (LID only).
# ---------------------------------------------------------------------------
ENGLISH_SEED_WORDS = [
    "the", "is", "are", "was", "were", "have", "has", "had", "will",
    "would", "should", "could", "can", "must", "and", "but", "or", "if",
    "because", "when", "where", "what", "who", "how", "why", "today",
    "tomorrow", "you", "your", "please", "do", "does", "did", "yes",
    "okay", "sure", "thanks", "sorry", "hello", "hi", "bye", "work",
    "home", "money", "food", "water", "friend", "people", "weather",
    "life", "love", "meeting", "office", "project", "deadline", "email",
    "phone", "call", "message", "traffic", "weekend", "movie", "dinner",
    "lunch", "school", "college", "exam", "job", "salary", "boss",
    "flight", "train", "ticket", "hotel", "trip", "birthday", "party",
    "office", "meeting", "nervous", "exam", "door", "late", "app",
    "internet", "wifi", "laptop", "video", "photo", "song", "cricket",
]


# ---------------------------------------------------------------------------
# 4. Build the Kannada-script -> Hindi dictionary used by the fallback
#    translator, by transliterating every romanized Kannada key once.
# ---------------------------------------------------------------------------
def build_kannada_script_dictionary(translit_to_kannada_fn):
    kn_dict = {}
    for roman, hindi_gloss in KANNADA_ROMAN_WORDS.items():
        kn_key = translit_to_kannada_fn(roman.replace("_", " "))
        kn_dict[kn_key] = hindi_gloss
    return kn_dict


# ---------------------------------------------------------------------------
# 5. Native Kannada script -> Hindi, entered directly (not derived via
#    ITRANS). This exists because ITRANS transliteration of the romanized
#    keys above does not always reproduce the spelling a Kannada speaker
#    would actually type (vowel length, anusvara vs. conjunct nasals,
#    etc. -- e.g. ITRANS "indu" -> "ಇನ್ದು", but the real word is "ಇಂದು").
#    These direct entries take priority over the derived dictionary so
#    genuine native-script input is translated correctly even where
#    transliteration would have missed it.
# ---------------------------------------------------------------------------
KANNADA_NATIVE_TO_HINDI = {
    "ನಾನು": "मैं", "ನೀನು": "तुम", "ನೀವು": "आप", "ನಾವು": "हम",
    "ಅವನು": "वह", "ಅವಳು": "वह", "ಅದು": "वह", "ಇದು": "यह",
    "ಅವರು": "वे", "ಇವರು": "ये", "ನನ್ನ": "मेरा", "ನಿನ್ನ": "तुम्हारा",
    "ನಮ್ಮ": "हमारा", "ಅವರ": "उनका",
    "ಇದೆ": "है", "ಇದ್ದೀನಿ": "हूँ", "ಇದ್ದಾರೆ": "हैं", "ಇಲ್ಲ": "नहीं",
    "ಆಗಿದೆ": "हो गया है", "ಆಗುತ್ತೆ": "हो जाएगा",
    "ಮಾಡು": "करना", "ಮಾಡಿ": "करो", "ಮಾಡಿದೆ": "किया है",
    "ಹೋಗು": "जाना", "ಬಾ": "आओ", "ಬರುತ್ತೇನೆ": "आऊंगा",
    "ಕೊಡು": "देना", "ನೋಡು": "देखो", "ಗೊತ್ತು": "पता है", "ಗೊತ್ತಿಲ್ಲ": "पता नहीं",
    "ಬೇಕು": "चाहिए", "ಬೇಡ": "नहीं चाहिए",
    "ಮನೆ": "घर", "ಕೆಲಸ": "काम", "ಆಫೀಸ್": "ऑफिस", "ಊಟ": "खाना",
    "ನೀರು": "पानी", "ಸ್ನೇಹಿತ": "दोस्त", "ಸಮಯ": "समय", "ದಿನ": "दिन",
    "ಇವತ್ತು": "आज", "ಇಂದು": "आज", "ನಾಳೆ": "कल", "ನಿನ್ನೆ": "बीता हुआ कल",
    "ಬೆಳಿಗ್ಗೆ": "सुबह", "ಸಂಜೆ": "शाम", "ರಾತ್ರಿ": "रात", "ಹಣ": "पैसा",
    "ಮಳೆ": "बारिश", "ಪಾರ್ಟಿ": "पार्टी", "ಸಿನಿಮಾ": "फिल्म", "ಮೀಟಿಂಗ್": "मीटिंग",
    "ಒಳ್ಳೆಯ": "अच्छा", "ಚೆನ್ನಾಗಿದೆ": "अच्छा है", "ಕೆಟ್ಟ": "बुरा",
    "ದೊಡ್ಡ": "बड़ा", "ಚಿಕ್ಕ": "छोटा", "ಹೊಸ": "नया",
    "ಸ್ವಲ್ಪ": "थोड़ा", "ತುಂಬಾ": "बहुत", "ಜಾಸ್ತಿ": "ज़्यादा", "ಕಡಿಮೆ": "कम",
    "ಬೇಗ": "जल्दी", "ಈಗ": "अभी",
    "ಮತ್ತು": "और", "ಆದರೆ": "लेकिन", "ಯಾಕೆ": "क्यों", "ಹೇಗೆ": "कैसे",
    "ಎಲ್ಲಿ": "कहाँ", "ಯಾವಾಗ": "कब", "ಏನು": "क्या", "ಯಾರು": "कौन",
    "ಎಷ್ಟು": "कितना", "ದಯವಿಟ್ಟು": "कृपया",

    # numbers 1-10
    "ಒಂದು": "एक", "ಎರಡು": "दो", "ಮೂರು": "तीन", "ನಾಲ್ಕು": "चार",
    "ಐದು": "पांच", "ಆರು": "छह", "ಏಳು": "सात", "ಎಂಟು": "आठ",
    "ಒಂಬತ್ತು": "नौ", "ಹತ್ತು": "दस",

    # days of the week
    "ಸೋಮವಾರ": "सोमवार", "ಮಂಗಳವಾರ": "मंगलवार", "ಬುಧವಾರ": "बुधवार",
    "ಗುರುವಾರ": "गुरुवार", "ಶುಕ್ರವಾರ": "शुक्रवार", "ಶನಿವಾರ": "शनिवार",
    "ಭಾನುವಾರ": "रविवार",

    # family
    "ಅಪ್ಪ": "पिता", "ಅಮ್ಮ": "माँ", "ಅಣ್ಣ": "बड़ा भाई", "ಅಕ್ಕ": "बड़ी बहन",
    "ತಮ್ಮ": "छोटा भाई", "ತಂಗಿ": "छोटी बहन", "ಮಗ": "बेटा",
    "ಮಗಳು": "बेटी", "ಗಂಡ": "पति", "ಹೆಂಡತಿ": "पत्नी",

    # greetings / politeness
    "ನಮಸ್ಕಾರ": "नमस्ते", "ಧನ್ಯವಾದ": "धन्यवाद", "ಸ್ವಾಗತ": "स्वागत",
    "ಕ್ಷಮಿಸಿ": "माफ़ कीजिए",

    # more verbs
    "ತಿನ್ನು": "खाना", "ಕುಡಿ": "पीना", "ಓದು": "पढ़ना", "ಬರೆ": "लिखना",
    "ಮಲಗು": "सोना", "ಎದ್ದೇಳು": "उठना", "ಆಡು": "खेलना", "ಹಾಡು": "गाना",

    # more adjectives
    "ಸುಂದರ": "सुंदर", "ಬಿಸಿ": "गरम", "ತಣ್ಣಗೆ": "ठंडा", "ಹಸಿ": "कच्चा",
    "ಸುಲಭ": "आसान", "ಕಷ್ಟ": "मुश्किल",

    # colors
    "ಕೆಂಪು": "लाल", "ಬಿಳಿ": "सफेद", "ಕಪ್ಪು": "काला", "ಹಸಿರು": "हरा",
    "ನೀಲಿ": "नीला", "ಹಳದಿ": "पीला",

    # food
    "ಹಣ್ಣು": "फल", "ತರಕಾರಿ": "सब्ज़ी", "ಹಾಲು": "दूध",
    "ಸಕ್ಕರೆ": "चीनी", "ಉಪ್ಪು": "नमक",
}
