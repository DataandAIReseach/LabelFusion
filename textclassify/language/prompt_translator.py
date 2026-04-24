import asyncio
from ..prompt_pipeline.prompt_warehouse import PromptWarehouse
from ..services.llm_content_generator import BaseLLMContentGenerator


class PromptTranslator:
    """Translates PromptWarehouse prompts to a target language via LLM."""

    def __init__(self, generator: BaseLLMContentGenerator):
        self._generator = generator
        self._cache = {}

    def translate(self, warehouse: PromptWarehouse, lang: str) -> PromptWarehouse:
        translated = PromptWarehouse()
        for attr, value in self._get_prompts(warehouse).items():
            setattr(translated, attr, self._translate_prompt(value, lang))
        return translated

    def _translate_prompt(self, text: str, lang: str) -> str:
        cache_key = f"{lang}:{hash(text)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Translate the following prompt template to {lang}.

IMPORTANT:
- Keep all placeholders like {{examples}}, {{labels}}, {{paragraph}} exactly as they are
- Only translate the natural language parts
- Preserve all formatting, newlines and indentation

Text to translate:
{text}"""

        result = asyncio.run(self._generator.generate_content(prompt))
        self._cache[cache_key] = result
        return result

    def _get_prompts(self, warehouse: PromptWarehouse) -> dict:
        return {
            attr: getattr(warehouse, attr)
            for attr in dir(warehouse)
            if not attr.startswith('_')
            and isinstance(getattr(warehouse, attr), str)
        }