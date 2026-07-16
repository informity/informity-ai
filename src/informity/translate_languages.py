"""Module for translate languages."""

# pylint: disable=redefined-outer-name

from dataclasses import dataclass

TRANSLATE_DEFAULT_LANGUAGE = "Spanish"
TRANSLATE_PINNED_LANGUAGE_LIMIT = 6


@dataclass(frozen=True, slots=True)
class TranslateLanguageOption:
    """Class docstring."""
    label: str
    qwen_code: str
    flag_code: str
    native_label: str
    aliases: tuple[str, ...] = ()
    # Model-facing name used in the translation prompt. Overrides `label` when
    # the label is ambiguous (e.g. "Chinese" → "Simplified Chinese") so the LLM
    # produces the correct script/variant without guessing.
    prompt_label: str | None = None

    @property
    def model_name(self) -> str:
        """Model name."""
        return self.prompt_label or self.label


TRANSLATE_LANGUAGE_OPTIONS: tuple[TranslateLanguageOption, ...] = (
    TranslateLanguageOption(
        "Arabic (Standard)",
        "ar",
        "sa",
        "العربية",
        ("Arabic", "Standard Arabic", "ar"),
        prompt_label="Modern Standard Arabic (MSA)",
    ),
    TranslateLanguageOption("Bengali", "bn", "bd", "বাংলা", ("Bangla", "bn")),
    TranslateLanguageOption(
        "Chinese",
        "zh",
        "cn",
        "中文",
        ("Chinese (Simplified)", "Simplified Chinese", "zh-cn", "zh-hans", "zh"),
        prompt_label="Simplified Chinese",
    ),
    TranslateLanguageOption("Czech", "cs", "cz", "Čeština", ("cs",)),
    TranslateLanguageOption("Dutch", "nl", "nl", "Nederlands", ("nl",)),
    TranslateLanguageOption(
        "English", "en", "gb", "English", ("British English", "American English", "en")
    ),
    TranslateLanguageOption("French", "fr", "fr", "Français", ("fr",)),
    TranslateLanguageOption("German", "de", "de", "Deutsch", ("de",)),
    TranslateLanguageOption("Greek", "el", "gr", "Ελληνικά", ("el",)),
    TranslateLanguageOption("Hebrew", "he", "il", "עברית", ("he",)),
    TranslateLanguageOption("Hindi", "hi", "in", "हिन्दी", ("hi",)),
    TranslateLanguageOption("Indonesian", "id", "id", "Bahasa Indonesia", ("id",)),
    TranslateLanguageOption("Italian", "it", "it", "Italiano", ("it",)),
    TranslateLanguageOption("Japanese", "ja", "jp", "日本語", ("ja",)),
    TranslateLanguageOption("Korean", "ko", "kr", "한국어", ("ko",)),
    TranslateLanguageOption("Persian", "fa", "ir", "فارسی", ("Farsi", "fa")),
    TranslateLanguageOption("Polish", "pl", "pl", "Polski", ("pl",)),
    TranslateLanguageOption(
        "Portuguese", "pt", "pt", "Português", ("European Portuguese", "Brazilian Portuguese", "pt")
    ),
    TranslateLanguageOption("Romanian", "ro", "ro", "Română", ("ro",)),
    TranslateLanguageOption("Russian", "ru", "ru", "Русский", ("ru",)),
    TranslateLanguageOption("Spanish", "es", "es", "Español", ("es",)),
    TranslateLanguageOption("Swedish", "sv", "se", "Svenska", ("sv",)),
    TranslateLanguageOption("Thai", "th", "th", "ไทย", ("th",)),
    TranslateLanguageOption("Turkish", "tr", "tr", "Türkçe", ("tr",)),
    TranslateLanguageOption("Ukrainian", "uk", "ua", "Українська", ("uk",)),
    TranslateLanguageOption("Vietnamese", "vi", "vn", "Tiếng Việt", ("vi",)),
)

TRANSLATE_LANGUAGE_LABELS: tuple[str, ...] = tuple(
    language.label for language in TRANSLATE_LANGUAGE_OPTIONS
)
_TRANSLATE_LANGUAGE_ORDER = {
    option.label: index for index, option in enumerate(TRANSLATE_LANGUAGE_OPTIONS)
}

_TRANSLATE_LANGUAGE_LOOKUP: dict[str, TranslateLanguageOption] = {}
for option in TRANSLATE_LANGUAGE_OPTIONS:
    keys = (
        option.label,
        option.label.lower(),
        option.qwen_code,
        option.qwen_code.lower(),
        option.flag_code,
        option.flag_code.lower(),
        option.native_label,
        option.native_label.lower(),
        *option.aliases,
        *(alias.lower() for alias in option.aliases),
    )
    for key in keys:
        if key:
            _TRANSLATE_LANGUAGE_LOOKUP.setdefault(str(key).strip(), option)
            _TRANSLATE_LANGUAGE_LOOKUP.setdefault(str(key).strip().lower(), option)


def find_translate_language_option(value: str | None) -> TranslateLanguageOption | None:
    """Find translate language option."""
    text = str(value or "").strip()
    if not text:
        return None
    return _TRANSLATE_LANGUAGE_LOOKUP.get(text) or _TRANSLATE_LANGUAGE_LOOKUP.get(text.lower())


def normalize_translate_language(value: str | None) -> str:
    """Normalize translate language."""
    text = str(value or "").strip()
    if not text:
        return TRANSLATE_DEFAULT_LANGUAGE
    option = find_translate_language_option(text)
    return option.label if option else TRANSLATE_DEFAULT_LANGUAGE


def normalize_translate_language_list(
    values: object, *, limit: int | None = TRANSLATE_PINNED_LANGUAGE_LIMIT
) -> list[str]:
    """Normalize translate language list."""
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_value in values:
        option = find_translate_language_option(str(raw_value or ""))
        if option is None:
            continue
        language = option.label
        if language in seen:
            continue
        seen.add(language)
        normalized.append(language)
    normalized.sort(
        key=lambda value: _TRANSLATE_LANGUAGE_ORDER.get(value, len(_TRANSLATE_LANGUAGE_ORDER))
    )
    if limit is not None and limit >= 0:
        return normalized[:limit]
    return normalized


def get_translate_language_model_name(value: str | None) -> str:
    """Return the model-facing language name for use in translation prompts.

    Falls back to the display label (and ultimately the input value) so callers
    always get a non-empty string even for unknown languages.
    """
    option = find_translate_language_option(value)
    if option:
        return option.model_name
    return str(value or "the target language").strip() or "the target language"


def get_translate_language_qwen_code(value: str | None) -> str | None:
    """Get translate language qwen code."""
    option = find_translate_language_option(value)
    return option.qwen_code if option else None


def get_translate_language_flag_code(value: str | None) -> str | None:
    """Get translate language flag code."""
    option = find_translate_language_option(value)
    return option.flag_code if option else None


def search_translate_languages(
    query: str, limit: int | None = None
) -> list[TranslateLanguageOption]:
    """Search translate languages."""
    needle = str(query or "").strip().lower()
    options = list(TRANSLATE_LANGUAGE_OPTIONS)
    if not needle:
        return options[: limit or len(options)]
    scored: list[tuple[int, str, TranslateLanguageOption]] = []
    for option in options:
        haystack = " ".join(
            (
                option.label,
                option.native_label,
                option.qwen_code,
                option.flag_code,
                *option.aliases,
            )
        ).lower()
        starts = (
            option.label.lower().startswith(needle)
            or option.native_label.lower().startswith(needle)
            or option.qwen_code.lower().startswith(needle)
            or option.flag_code.lower().startswith(needle)
            or any(alias.lower().startswith(needle) for alias in option.aliases)
        )
        score = 0 if starts else (1 if needle in haystack else 2)
        if score < 2 or needle in haystack:
            scored.append((score, option.label, option))
    scored.sort(key=lambda item: (item[0], item[1]))
    matches = [item[2] for item in scored]
    return matches[: limit or len(matches)]
