from foid_protocol.i18n import LANGUAGE_NAMES
from foid_voice.config import VoiceSettings
from foid_voice.stt_profiles import (
    AUTO_LANGUAGE,
    GERMAN_LARGE,
    GERMAN_TURBO,
    PICKER_LANGUAGES,
    PINNED_LANGUAGES,
    SHARED_PICKER_LANGUAGES,
    WHISPER_LANGUAGES,
    models_for_language,
    ordered_languages,
    picker_language,
    resolve_stt,
    whisper_language,
)

MULTILINGUAL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]


def test_whisper_languages_are_named():
    missing = [code for code in ("auto", *WHISPER_LANGUAGES) if code not in LANGUAGE_NAMES]
    assert missing == []
    assert len(WHISPER_LANGUAGES) == 100


def test_dropdown_special_languages_only():
    assert PINNED_LANGUAGES == ("auto", "de", "en")
    assert PICKER_LANGUAGES == ("auto", "de", "en")
    assert SHARED_PICKER_LANGUAGES == (AUTO_LANGUAGE,)
    assert "cs" not in PICKER_LANGUAGES
    assert "fr" not in PICKER_LANGUAGES
    assert ordered_languages("en") == ["auto", "de", "en"]
    assert ordered_languages("de") == ["auto", "de", "en"]
    assert picker_language("de") == "de"
    assert picker_language("en") == "en"
    assert picker_language("auto") == "auto"
    assert picker_language("cs") == "auto"
    assert picker_language("fr") == "auto"
    assert picker_language("haw") == "auto"


def test_models_by_language():
    de_ids = [item.id for item in models_for_language("de")]
    en_ids = [item.id for item in models_for_language("en")]
    auto_ids = [item.id for item in models_for_language("auto")]
    cs_ids = [item.id for item in models_for_language("cs")]
    fr_ids = [item.id for item in models_for_language("fr")]
    assert de_ids == [GERMAN_TURBO, GERMAN_LARGE]
    assert auto_ids == MULTILINGUAL_SIZES
    assert cs_ids == []
    assert fr_ids == []
    assert GERMAN_TURBO not in en_ids
    assert en_ids == [
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "distil-small.en",
        "distil-medium.en",
    ]
    for size in MULTILINGUAL_SIZES:
        assert size not in de_ids
        assert size not in en_ids


def test_shared_and_quiet_defaults():
    assert resolve_stt("cs", "") == ("auto", "medium")
    assert resolve_stt("fr", None) == ("auto", "medium")
    assert resolve_stt("cs", "small") == ("auto", "small")
    assert resolve_stt("auto", "") == ("auto", "medium")
    assert resolve_stt("de", "small") == ("de", "small")


def test_legacy_profile_migration():
    mapping = {
        "de_turbo": ("de", GERMAN_TURBO),
        "de_large": ("de", GERMAN_LARGE),
        "de_small": ("de", "small"),
        "cs_small": (AUTO_LANGUAGE, "small"),
        "cs_medium": (AUTO_LANGUAGE, "medium"),
        "multi_medium": (AUTO_LANGUAGE, "medium"),
    }
    for profile, expected in mapping.items():
        settings = VoiceSettings.model_validate({"stt_profile": profile})
        assert (settings.stt_language, settings.stt_model) == expected


def test_invalid_model_falls_back():
    language, model = resolve_stt("auto", GERMAN_TURBO)
    assert language == "auto"
    assert model == "medium"
    assert whisper_language("auto") is None
    assert whisper_language("de") == "de"
