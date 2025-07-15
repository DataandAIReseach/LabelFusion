from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PromptCollection:
    """Stores complete prompts created by PromptEngineer."""
    
    prompts: Dict[str, str] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    
    def add_prompt(self, name: str, content: str) -> None:
        """Add a prompt to the collection."""
        self.prompts[name] = content
        if name not in self.order:
            self.order.append(name)
    
    def get_prompt(self, name: str) -> str:
        """Get a prompt by name."""
        return self.prompts.get(name, "")
    
    def get_all_prompts(self) -> str:
        """Get all prompts in order."""
        return "\n\n".join(
            self.prompts[name] 
            for name in self.order 
            if name in self.prompts
        )
    
    def clear(self) -> None:
        """Clear all prompts."""
        self.prompts.clear()
        self.order.clear()