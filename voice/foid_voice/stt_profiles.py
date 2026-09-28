from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from foid_protocol.i18n import LANGUAGE_NAMES

# Official Whisper / faster-whisper language tokens (ISO 639-1 plus haw, yue).
WHISPER_LANGUAGES: tuple[str, ...] = (
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
)

AUTO_LANGUAGE = "auto"

GERMAN_TURBO = "cstr/whisper-large-v3-turbo-german-int8_float32"
GERMAN_LARGE = "primeline/whisper-large-v3-german"

DEFAULT_STT_LANGUAGE = "de"
DEFAULT_STT_MODEL = GERMAN_TURBO
DEFAULT_RECOMMENDED_MODEL = "medium"

# Shared multilingual sizes appear only under Auto (language=None).
SHARED_PICKER_LANGUAGES: tuple[str, ...] = (AUTO_LANGUAGE,)

# Dropdown: Auto + languages that have dedicated special models only.
PINNED_LANGUAGES: tuple[str, ...] = (AUTO_LANGUAGE, "de", "en")
PICKER_LANGUAGES: tuple[str, ...] = PINNED_LANGUAGES

RECOMMENDED_BY_LANGUAGE: dict[str, str] = {
    "de": GERMAN_TURBO,
    AUTO_LANGUAGE: "medium",
    "en": "small.en",
}

LEGACY_PROFILES: dict[str, tuple[str, str]] = {
    "de_turbo": ("de", GERMAN_TURBO),
    "de_large": ("de", GERMAN_LARGE),
    "de_small": ("de", "small"),
    "cs_small": (AUTO_LANGUAGE, "small"),
    "cs_medium": (AUTO_LANGUAGE, "medium"),
    "multi_medium": (AUTO_LANGUAGE, "medium"),
}


@dataclass(frozen=True)
class SttModel:
    id: str
    label_key: str
    hint_key: str
    compute: str = "int8"
    for_languages: tuple[str, ...] | None = None
    english_only: bool = False
    german_finetune: bool = False
    specific: bool = False
    shared_pack: bool = False

    def available_for(self, language: str) -> bool:
        """Whether this model may be loaded with the given language setting."""
        if self.english_only and language != "en":
            return False
        if self.german_finetune and language != "de":
            return False
        if self.shared_pack:
            # Multilingual sizes stay quiet defaults for any language lock.
            return True
        if self.for_languages is not None and language not in self.for_languages:
            return False
        return True

    def listed_for(self, language: str) -> bool:
        """Whether this model appears in the settings radio list for language."""
        if not self.specific:
            return False
        if self.shared_pack:
            return language in SHARED_PICKER_LANGUAGES
        return self.available_for(language)

    def as_picker(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label_key": self.label_key,
            "hint_key": self.hint_key,
            "for_languages": list(self.for_languages) if self.for_languages is not None else None,
            "english_only": self.english_only,
            "german_finetune": self.german_finetune,
            "specific": self.specific,
            "shared_pack": self.shared_pack,
        }


def _model(model_id: str, key: str, **kwargs: Any) -> SttModel:
    return SttModel(
        id=model_id,
        label_key=f"stt.model.{key}.label",
        hint_key=f"stt.model.{key}.hint",
        **kwargs,
    )


# German fine-tunes, English .en packs, and the shared multilingual size list (Auto only).
# Official distil-large-* short names are English-oriented; keep as quiet loadables only.
STT_MODELS: tuple[SttModel, ...] = (
    _model(GERMAN_TURBO, "de_turbo", for_languages=("de",), german_finetune=True, specific=True),
    _model(GERMAN_LARGE, "de_large", for_languages=("de",), german_finetune=True, specific=True),
    _model("tiny", "tiny", specific=True, shared_pack=True),
    _model("base", "base", specific=True, shared_pack=True),
    _model("small", "small", specific=True, shared_pack=True),
    _model("medium", "medium", specific=True, shared_pack=True),
    _model("large-v2", "large_v2", specific=True, shared_pack=True),
    _model("large-v3", "large_v3", specific=True, shared_pack=True),
    _model("large-v3-turbo", "large_v3_turbo", specific=True, shared_pack=True),
    _model("tiny.en", "tiny_en", english_only=True, specific=True),
    _model("base.en", "base_en", english_only=True, specific=True),
    _model("small.en", "small_en", english_only=True, specific=True),
    _model("medium.en", "medium_en", english_only=True, specific=True),
    _model("distil-small.en", "distil_small_en", english_only=True, specific=True),
    _model("distil-medium.en", "distil_medium_en", english_only=True, specific=True),
    _model("distil-large-v2", "distil_large_v2", english_only=True),
    _model("distil-large-v3", "distil_large_v3", english_only=True),
)

_MODELS_BY_ID = {item.id: item for item in STT_MODELS}

_MULTILINGUAL_SIZE_IDS = tuple(item.id for item in STT_MODELS if item.shared_pack)


def is_valid_language(code: str) -> bool:
    return code == AUTO_LANGUAGE or code in WHISPER_LANGUAGES


def get_model(model_id: str) -> SttModel:
    spec = _MODELS_BY_ID.get(model_id)
    if spec is None:
        raise KeyError(f"unknown STT model: {model_id}")
    return spec


def models_for_language(language: str) -> list[SttModel]:
    return [item for item in STT_MODELS if item.listed_for(language)]


def recommended_model(language: str) -> str:
    return RECOMMENDED_BY_LANGUAGE.get(language, DEFAULT_RECOMMENDED_MODEL)


def model_allowed(language: str, model_id: str) -> bool:
    spec = _MODELS_BY_ID.get(model_id)
    return spec is not None and spec.available_for(language)


def whisper_language(language: str) -> str | None:
    if language == AUTO_LANGUAGE:
        return None
    return language


def migrate_legacy_profile(profile_id: Any) -> tuple[str, str] | None:
    if profile_id is None:
        return None
    return LEGACY_PROFILES.get(str(profile_id).strip())


def resolve_stt(language: Any, model_id: Any) -> tuple[str, str]:
    lang = str(language or "").strip().lower()
    if lang in {"", "none", "multi", "multilingual"}:
        lang = AUTO_LANGUAGE
    if not is_valid_language(lang):
        lang = DEFAULT_STT_LANGUAGE
    elif lang not in PICKER_LANGUAGES:
        # Curated locks without special models (e.g. former Czech) fall back to Auto.
        lang = AUTO_LANGUAGE
    model = str(model_id or "").strip()
    if not model_allowed(lang, model):
        model = recommended_model(lang)
    return lang, model


def picker_language(language: str) -> str:
    if language in PICKER_LANGUAGES:
        return language
    return AUTO_LANGUAGE


def _locale_sort_key(code: str, locale: str) -> str:
    names = LANGUAGE_NAMES.get(code)
    if not names:
        return code
    return names[1 if locale == "de" else 0].casefold()


def ordered_languages(locale: str = "en") -> list[str]:
    pinned = [code for code in PINNED_LANGUAGES if code in PICKER_LANGUAGES]
    rest = sorted(
        (code for code in PICKER_LANGUAGES if code not in PINNED_LANGUAGES),
        key=lambda code: _locale_sort_key(code, locale),
    )
    return pinned + rest


def picker_catalog() -> dict[str, Any]:
    return {
        "languages": list(PICKER_LANGUAGES),
        "pinned": list(PINNED_LANGUAGES),
        "shared_languages": list(SHARED_PICKER_LANGUAGES),
        "models": [item.as_picker() for item in STT_MODELS],
        "recommended": dict(RECOMMENDED_BY_LANGUAGE),
        "default_recommended": DEFAULT_RECOMMENDED_MODEL,
        "multilingual_sizes": list(_MULTILINGUAL_SIZE_IDS),
    }
