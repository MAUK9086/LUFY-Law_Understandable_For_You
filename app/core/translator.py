"""Translation utilities using the GoogleTranslator backend from deep-translator."""

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "Sanskrit": "sa",
    "Sindhi": "sd",
    "Kashmiri": "ks",
}

# Reverse map: BCP-47 code -> display name for quick lookup
_CODE_TO_NAME: dict[str, str] = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

_MAX_CHUNK_CHARS = 4500


def _resolve_language_code(target_language: str) -> str | None:
    """Resolve a display name or BCP-47 code to a canonical BCP-47 code.

    Args:
        target_language: Either a display name (e.g. "Hindi") or a BCP-47
            code (e.g. "hi").

    Returns:
        The BCP-47 code, or None if the language is not supported.
    """
    if target_language in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[target_language]
    if target_language in _CODE_TO_NAME:
        return target_language
    return None


def translate(text: str, target_language: str) -> str:
    """Translate text to the target language.

    Accepts either a display name ("Hindi") or a BCP-47 code ("hi"). Returns
    the original text unchanged when the target is English, the text is blank,
    or any error occurs during translation.

    Texts longer than 4500 characters are split on paragraph boundaries before
    translation and reassembled afterwards.

    Args:
        text: Text to translate.
        target_language: Target language as a display name or BCP-47 code.

    Returns:
        Translated text, or the original text if translation is unavailable
        or fails.
    """
    if not text or not text.strip():
        return text

    lang_code = _resolve_language_code(target_language)
    if lang_code is None:
        logger.warning("Unknown target language '%s'; skipping translation", target_language)
        return text

    if lang_code == "en":
        return text

    try:
        if len(text) <= _MAX_CHUNK_CHARS:
            return GoogleTranslator(source="auto", target=lang_code).translate(text)

        paragraphs = text.split("\n\n")
        translated_parts: list[str] = []
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) + 2 > _MAX_CHUNK_CHARS:
                if buffer:
                    translated_parts.append(
                        GoogleTranslator(source="auto", target=lang_code).translate(buffer)
                    )
                buffer = para
            else:
                buffer = (buffer + "\n\n" + para).strip() if buffer else para

        if buffer:
            translated_parts.append(
                GoogleTranslator(source="auto", target=lang_code).translate(buffer)
            )

        return "\n\n".join(translated_parts)

    except Exception as exc:
        logger.warning("Translation to '%s' failed: %s", target_language, exc)
        return text
