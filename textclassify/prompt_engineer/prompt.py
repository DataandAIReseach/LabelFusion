from typing import List, Dict, Optional, Any

class Prompt:
    def __init__(self):
        self.parts: List[Dict[str, str]] = []  # Each part is {"name": ..., "content": ...}
        self.variables: Dict[str, Any] = {}     # Global variable substitutions

    def add_part(self, name: str, content: str, position: Optional[int] = None):
        """Add a named part to the prompt at the specified position (default: end)."""
        part = {"name": name, "content": content}
        if position is None:
            self.parts.append(part)
        else:
            self.parts.insert(position, part)

    def update_part(self, name: str, content: str):
        """Update content of an existing named part."""
        for part in self.parts:
            if part["name"] == name:
                part["content"] = content
                return
        raise ValueError(f"No part found with name: {name}")

    def remove_part(self, name: str):
        """Remove a named part from the prompt."""
        self.parts = [part for part in self.parts if part["name"] != name]

    def set_variables(self, **kwargs):
        """Set global variables to be used for rendering the prompt."""
        self.variables.update(kwargs)

    def render(self, input_text: Optional[str] = None, extra_vars: Optional[Dict[str, Any]] = None) -> str:
        """Render the full prompt with all variables applied."""
        combined_vars = {**self.variables}
        if extra_vars:
            combined_vars.update(extra_vars)
        if input_text:
            combined_vars["input"] = input_text

        rendered_parts = []
        for part in self.parts:
            try:
                rendered = part["content"].format(**combined_vars)
            except KeyError as e:
                raise ValueError(f"Missing variable in prompt: {e}")
            rendered_parts.append(rendered)

        return "\n\n".join(rendered_parts).strip()

    def __str__(self):
        return self.render()

    def to_dict(self) -> Dict[str, str]:
        """Export parts as a dictionary."""
        return {part["name"]: part["content"] for part in self.parts}

    def reset(self):
        """Reset parts and variables."""
        self.parts.clear()
        self.variables.clear()
