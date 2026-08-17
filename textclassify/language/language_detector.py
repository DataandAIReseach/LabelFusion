from lingua import Language, LanguageDetectorBuilder

class LanguageDetector:
    """
    Detects the language of a text using the lingua library.
    Fast, free and offline — no LLM needed.
    """

    def __init__(self, languages: list[Language] = None):
        self.detector = (
            LanguageDetectorBuilder
            .from_languages(*languages)
            .build()
            if languages else
            LanguageDetectorBuilder
            .from_all_languages()
            .build()
        )

    def detect(self, text: str) -> str:
        """Returns plain language name e.g. 'german', falls back to 'english'."""
        lang = self.detector.detect_language_of(text)
        if lang is None:
            return 'english'
        return lang.name.lower()