class PromptTranslator:
    """Übersetzt Prompts via LLM."""
    
    def __init__(self, client):
        self.client = client
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
        # ... LLM call ...
        self._cache[cache_key] = result
        return result

    def _get_prompts(self, warehouse: PromptWarehouse) -> dict:
        return {
            attr: getattr(warehouse, attr)
            for attr in dir(warehouse)
            if not attr.startswith('_')
            and isinstance(getattr(warehouse, attr), str)
        }