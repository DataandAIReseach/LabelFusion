from __future__ import annotations
from typing import Callable, Iterable, Optional

class PromptLanguage:
    """Language detection + prompt rendering/translation helper."""

    def __init__(
        self,
        default_language: str = "en",
        language: Optional[str] = None,
        auto_detect: bool = True,
        translate_prompts: bool = True,
        translator: Optional[Callable[[str, str, str], str]] = None,
    ):
        self.default_language = default_language
        self.language = language or default_language
        self.auto_detect = auto_detect
        self.translate_prompts = translate_prompts
        # translator(text, target_lang, source_lang) -> translated_text
        self.translator = translator

    def detect_language(self, texts: Iterable[str]) -> str:
        """Detect language from sample texts. Falls back to default."""
        if not self.auto_detect:
            return self.language

        sample = " ".join([str(t) for t in texts if t][:30]).strip()
        if not sample:
            self.language = self.default_language
            return self.language

        try:
            from langdetect import detect  # pip install langdetect
            lang = detect(sample)
            self.language = lang if lang else self.default_language
        except Exception:
            self.language = self.default_language

        return self.language

    def set_language(self, language: str) -> None:
        self.language = language or self.default_language

    def render_prompt(self, english_template: str, **kwargs) -> str:
        """
        Render English template and optionally translate to self.language.
        """
        prompt_en = english_template.format(**kwargs)
        if not self.translate_prompts:
            return prompt_en
        if self.language.startswith("en"):
            return prompt_en
        if self.translator is None:
            return prompt_en  # safe fallback

        try:
            return self.translator(prompt_en, self.language, "en")
        except Exception:
            return prompt_en