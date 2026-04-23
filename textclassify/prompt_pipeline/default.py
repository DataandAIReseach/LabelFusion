from typing import Optional
from .base import PromptPipeline
from .prompt_warehouse import PromptWarehouse
from ..language.language_detector import LanguageDetector
from ..language.prompt_translator import PromptTranslator
from ..services.llm_content_generator import BaseLLMContentGenerator
 
 
class DefaultPromptPipeline(PromptPipeline):
    """
    Default pipeline: detects language via lingua,
    translates via LLM if not English.
    """
 
    def __init__(
        self,
        generator: BaseLLMContentGenerator,
        warehouse: Optional[PromptWarehouse] = None,
        detector: Optional[LanguageDetector] = None,
    ):
        self._warehouse  = warehouse or PromptWarehouse()
        self._detector   = detector or LanguageDetector()
        self._translator = PromptTranslator(generator=generator)
 
    def get_warehouse(self, text: str) -> PromptWarehouse:
        lang = self._detector.detect(text)
        if lang == 'english':
            return self._warehouse
        return self._translator.translate(self._warehouse, lang)
 