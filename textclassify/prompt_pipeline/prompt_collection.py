from dataclasses import dataclass, field
from typing import Dict, List
from .prompt import Prompt

@dataclass
class PromptCollection:
    """Stores and manages Prompt objects."""
    
    prompts: Dict[str, Prompt] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    
    def add_prompt(self, name: str, prompt: Prompt) -> None:
        """Add a Prompt object to the collection."""
        self.prompts[name] = prompt
        if name not in self.order:
            self.order.append(name)
    
    def get_prompt(self, name: str) -> Prompt:
        """Get a Prompt object by name."""
        return self.prompts.get(name)
    
    def merge_prompts(self, variables: Dict[str, str] = None) -> Prompt:
        """Merge all prompts in order into a single Prompt."""
        merged = Prompt()
        
        for name in self.order:
            if name in self.prompts:
                prompt = self.prompts[name]
                for part in prompt.parts:
                    merged.add_part(part["name"], part["content"])
                    
        if variables:
            merged.set_variables(**variables)
            
        return merged
    
    def clear(self) -> None:
        """Clear all prompts."""
        self.prompts.clear()
        self.order.clear()