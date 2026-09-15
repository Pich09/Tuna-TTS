"""
Khmer text normalization for Tuna-TTS.

This is the "Normalize text" box referenced in PLAN.md section 21 (Dataset
Pipeline). It is the single entry point that MUST be called identically at:

    training time   (data/dataset.py before tokenization)
    validation time (same code path as training)
    inference time  (before feeding text to TunaTokenizer)

Any divergence between training-time and inference-time normalization is a
silent source of train/test mismatch, so keep this module dependency-free
and import it from exactly one place on each side.

Scope (matches the gap identified in PLAN.md review):
    - Khmer digit <-> Arabic digit handling
    - Cardinal number -> Khmer words          (២០២៦ / 2026 -> ពីរពាន់ម្ភៃប្រាំមួយ)
    - Currency        -> Khmer words          ($10, ៛5000, 20000 រៀល)
    - Dates            -> Khmer words          (ថ្ងៃទី5 ខែមករា ឆ្នាំ2026, 05/01/2026)

Not in scope (left for a later pass, see PLAN.md "Text Normalization" note):
    - Time-of-day ("3:45 PM")
    - Percentages / units (kg, km, %)
    - Abbreviation expansion
Extend this module rather than adding a second normalizer, so training and
inference never drift apart.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Digit tables
# ---------------------------------------------------------------------------

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ARABIC_DIGITS = "0123456789"
_KHMER_TO_ARABIC = str.maketrans(KHMER_DIGITS, ARABIC_DIGITS)

DIGIT_WORDS = [
    "សូន្យ", "មួយ", "ពីរ", "បី", "បួន",
    "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន",
]

TENS_WORDS = {
    2: "ម្ភៃ",
    3: "សាមសិប",
    4: "សែសិប",
    5: "ហាសិប",
    6: "ហុកសិប",
    7: "ចិតសិប",
    8: "ប៉ែតសិប",
    9: "កៅសិប",
}

MONTH_NAMES = [
    "មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា",
    "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ",
]


def khmer_digits_to_arabic(text: str) -> str:
    """Convert any Khmer-numeral digits in `text` to Arabic digits."""
    return text.translate(_KHMER_TO_ARABIC)


# ---------------------------------------------------------------------------
# Cardinal number -> Khmer words
# ---------------------------------------------------------------------------

def _convert_below_million(n: int) -> str:
    """Convert an integer in [0, 999_999] to Khmer words."""
    if n == 0:
        return ""

    parts = []

    if n >= 100_000:
        d, n = divmod(n, 100_000)
        parts.append(DIGIT_WORDS[d] + "សែន")
    if n >= 10_000:
        d, n = divmod(n, 10_000)
        parts.append(DIGIT_WORDS[d] + "ម៉ឺន")
    if n >= 1_000:
        d, n = divmod(n, 1_000)
        parts.append(DIGIT_WORDS[d] + "ពាន់")
    if n >= 100:
        d, n = divmod(n, 100)
        parts.append(DIGIT_WORDS[d] + "រយ")

    if n >= 20:
        tens, n = divmod(n, 10)
        parts.append(TENS_WORDS[tens])
        if n > 0:
            parts.append(DIGIT_WORDS[n])
    elif n >= 10:
        parts.append("ដប់")
        if n > 10:
            parts.append(DIGIT_WORDS[n - 10])
    elif n > 0:
        parts.append(DIGIT_WORDS[n])

    return "".join(parts)


def number_to_khmer_words(n: int) -> str:
    """
    Convert an arbitrary (possibly negative) integer to Khmer words.

    Large numbers are formed by recursive grouping on "លាន" (10**6), which
    matches how Khmer speakers actually compose large numbers, e.g.:
        2_026          -> ពីរពាន់ម្ភៃប្រាំមួយ
        1_000_000      -> មួយលាន
        2_500_000_000  -> ពីរពាន់ប្រាំរយលាន
    """
    if n < 0:
        return "ដក" + number_to_khmer_words(-n)
    if n == 0:
        return DIGIT_WORDS[0]
    if n < 1_000_000:
        return _convert_below_million(n)

    million_part, remainder = divmod(n, 1_000_000)
    words = number_to_khmer_words(million_part) + "លាន"
    if remainder:
        words += number_to_khmer_words(remainder)
    return words


def decimal_to_khmer_words(int_part: int, frac_str: str) -> str:
    """
    Convert a decimal split into an integer part and a fractional digit
    string (e.g. "5", "25") to Khmer words, reading fractional digits one
    at a time: 3.14 -> "បីក្បៀសមួយបួន" (three point one four).
    """
    words = number_to_khmer_words(int_part)
    if frac_str:
        words += "ក្បៀស" + "".join(DIGIT_WORDS[int(d)] for d in frac_str)
    return words


_NUMBER_RE = re.compile(r"(?<!\S)(\d+)(?:\.(\d+))?(?!\S*[a-zA-Z])")


def _replace_plain_number(match: re.Match) -> str:
    int_part, frac_part = match.group(1), match.group(2)
    if frac_part:
        return decimal_to_khmer_words(int(int_part), frac_part)
    return number_to_khmer_words(int(int_part))


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d+))?")
_RIEL_SYMBOL_RE = re.compile(r"៛\s?(\d[\d,]*)(?:\.(\d+))?")
_RIEL_WORD_RE = re.compile(r"(\d[\d,]*)(?:\.(\d+))?\s?រៀល")


def _num_str_to_int(num_str: str) -> int:
    return int(num_str.replace(",", ""))


def _replace_dollar(match: re.Match) -> str:
    n = _num_str_to_int(match.group(1))
    words = number_to_khmer_words(n) + "ដុល្លារ"
    if match.group(2):
        words += "ក្បៀស" + "".join(DIGIT_WORDS[int(d)] for d in match.group(2))
    return words


def _replace_riel(match: re.Match) -> str:
    n = _num_str_to_int(match.group(1))
    words = number_to_khmer_words(n) + "រៀល"
    if match.group(2):
        words += "ក្បៀស" + "".join(DIGIT_WORDS[int(d)] for d in match.group(2))
    return words


def _normalize_currency(text: str) -> str:
    text = _DOLLAR_RE.sub(_replace_dollar, text)
    text = _RIEL_SYMBOL_RE.sub(_replace_riel, text)
    text = _RIEL_WORD_RE.sub(_replace_riel, text)
    return text


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

# ថ្ងៃទី5 ខែមករា ឆ្នាំ2026  /  ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026
_LONG_DATE_RE = re.compile(
    r"ថ្ងៃទី\s?(\d{1,2})\s?ខែ(" + "|".join(MONTH_NAMES) + r")\s?ឆ្នាំ\s?(\d{3,4})"
)

# DD/MM/YYYY or DD-MM-YYYY
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)")


def _replace_long_date(match: re.Match) -> str:
    day, month_name, year = match.group(1), match.group(2), match.group(3)
    return (
        "ថ្ងៃទី" + number_to_khmer_words(int(day))
        + " ខែ" + month_name
        + " ឆ្នាំ" + number_to_khmer_words(int(year))
    )


def _replace_numeric_date(match: re.Match) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= month <= 12):
        # Not actually a date (e.g. a fraction-looking slash expression);
        # leave untouched for the generic number pass to handle.
        return match.group(0)
    month_name = MONTH_NAMES[month - 1]
    return (
        "ថ្ងៃទី" + number_to_khmer_words(day)
        + " ខែ" + month_name
        + " ឆ្នាំ" + number_to_khmer_words(year)
    )


def _normalize_dates(text: str) -> str:
    text = _LONG_DATE_RE.sub(_replace_long_date, text)
    text = _NUMERIC_DATE_RE.sub(_replace_numeric_date, text)
    return text


# ---------------------------------------------------------------------------
# Whitespace / punctuation cleanup
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def _clean_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize raw Khmer (and mixed Khmer/English) text into a form suitable
    for tokenization: Khmer numerals, currency, and dates are converted to
    fully spelled-out Khmer words.

    Order matters:
        1. Khmer digits -> Arabic digits (uniform matching for the passes below)
        2. Dates          (consume digit groups before generic numbers do)
        3. Currency       (consume digit groups before generic numbers do)
        4. Plain numbers  (whatever digits remain)
        5. Whitespace cleanup
    """
    text = khmer_digits_to_arabic(text)
    text = _normalize_dates(text)
    text = _normalize_currency(text)
    text = _NUMBER_RE.sub(_replace_plain_number, text)
    text = _clean_whitespace(text)
    return text


if __name__ == "__main__":
    examples = [
        "ខ្ញុំមាន $10 នៅឆ្នាំ 2026។",
        "ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026 ខ្ញុំនឹងទៅសាលា។",
        "05/01/2026",
        "ថ្លៃឡានគឺ ៛5000 ប៉ុណ្ណោះ។",
        "គាត់ដាក់ប្រាក់ 20000 រៀល ក្នុងធនាគារ។",
        "ខ្ញុំមានអាយុ 25 ឆ្នាំ ហើយកម្ពស់ខ្ញុំគឺ 1.75 ម៉ែត្រ។",
        "លេខទូរស័ព្ទ ០៩៦ ៣៣៣ ៤៤៤",
    ]
    for ex in examples:
        print(f"{ex}\n  -> {normalize(ex)}\n")
