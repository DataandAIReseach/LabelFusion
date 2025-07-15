import random
import pandas as pd
from typing import List, Optional, Dict, Any
from prompt_warehouse import PromptWarehouse
from prompt import Prompt  # Assuming you've implemented the modular Prompt class
from .prompt_collection import PromptCollection

class PromptEngineer:
    """Builds prompts for LLM-based text classification."""

    def __init__(
        self,
        text_column: str,
        label_columns: List[str],
        few_shot_mode: str = "few_shot"
    ):
        self.text_column = text_column
        self.label_columns = label_columns
        self.few_shot_mode = few_shot_mode
        self.role_prompt = None
        
    
    def set_few_shot_mode(self, mode: str):
        assert mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
        self.few_shot_mode = mode

    def set_examples(self, texts, labels):
        n = len(texts)
        indices = list(range(n))
        if self.few_shot_mode == "zero_shot":
            self.examples = []
        elif self.few_shot_mode == "one_shot":
            if n > 0:
                idx = random.choice(indices)
                self.examples = [{'text': texts[idx], 'label': labels[idx]}]
            else:
                self.examples = []
        elif self.few_shot_mode == "few_shot":
            k = min(5, n)
            if k > 0:
                idxs = random.sample(indices, k)
                self.examples = [{'text': texts[i], 'label': labels[i]} for i in idxs]
            else:
                self.examples = []
        elif self.few_shot_mode == "full_coverage":
            if n > 0:
                idxs = random.sample(indices, n)
                self.examples = [{'text': texts[i], 'label': labels[i]} for i in idxs]
            else:
                self.examples = []
        else:
            self.examples = []

    
    def engineer_prompts(self, data: Optional[pd.DataFrame] = None):
        """Set the training data for prompt engineering."""
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            self.data = data
        else:
            self.data = None
        for i, row in self.data.iterrows():
            p = Prompt()
            text = row[self.text_column]
            self.add_role_prompt()
            self.add_context_brainstorm(text)
            
    def add_role_prompt(self, prompt: Prompt) -> Prompt:
        """Add role prompt to the given prompt object."""
        content = self.role_prompt or PromptWarehouse.get_role_prompt()
        prompt.add_part("role", content)
        return prompt

    def add_theory_background(self, prompt: Prompt) -> Prompt:
        """Add theory background to the given prompt object."""
        prompt.add_part("theory", PromptWarehouse.get_theory_background())
        return prompt

    def add_train_data_intro(self, prompt: Prompt) -> Prompt:
        """Add training data introduction to the given prompt object."""
        prompt.add_part("train_data_intro", PromptWarehouse.get_train_data_intro())
        return prompt

    def add_train_data(self, prompt: Prompt) -> Prompt:
        """Add training data prompt to the given prompt object."""
        if self.data is not None:
            prompt.add_part(
                "train_data",
                PromptWarehouse.get_train_data_prompt(self.data.columns.tolist())
            )
        return prompt

    def add_context_brainstorm(self, prompt: Prompt, sample_size: int = 20) -> Prompt:
        """Add context brainstorm to the given prompt object."""
        if self.data is not None:
            prompt.add_part(
                "context_brainstorm",
                PromptWarehouse.get_context_brainstorm(
                    self.data, 
                    self.label_columns, 
                    sample_size
                )
            )
        return prompt

    def add_context_brainstorm_role(self, prompt: Prompt) -> Prompt:
        """Add context brainstorm role to the given prompt object."""
        if self.data is not None:
            prompt.add_part(
                "context_brainstorm_role",
                PromptWarehouse.get_context_brainstorm_role(self.data.columns.tolist())
            )
        return prompt

    def add_create_context(self, prompt: Prompt, keywords: List[str]) -> Prompt:
        """Add context creation prompt to the given prompt object."""
        prompt.add_part(
            "create_context",
            PromptWarehouse.get_create_context(keywords)
        )
        return prompt

    def add_procedure_prompt(self, prompt: Prompt) -> Prompt:
        """Add procedure prompt to the given prompt object."""
        prompt.add_part("procedure", PromptWarehouse.get_procedure_prompt())
        return prompt

    def add_procedure_creator(self, prompt: Prompt) -> Prompt:
        """Add procedure creator prompt to the given prompt object."""
        if self.data is not None:
            prompt.add_part(
                "procedure_creator",
                PromptWarehouse.get_procedure_creator(self.data)
            )
        return prompt

    def add_answer_format(self, prompt: Prompt, is_multi_label: bool) -> Prompt:
        """Add appropriate answer format to the given prompt object."""
        if is_multi_label:
            prompt.add_part(
                "answer_format",
                PromptWarehouse.get_answer_format_multi(self.label_columns)
            )
        else:
            prompt.add_part(
                "answer_format",
                PromptWarehouse.get_answer_format_single(self.label_columns)
            )
        return prompt

    def add_input_text(self, prompt: Prompt, text: str) -> Prompt:
        """Add input text to the given prompt object."""
        prompt.add_part("input", f"Text to analyze: {text}")
        return prompt

    def build_complete_prompt(
        self,
        text: str,
        is_multi_label: bool = False,
        include_theory: bool = True
    ) -> str:
        """Build complete prompt with all components."""
        self.reset()
        
        # Add role prompt (either default or created)
        if self.role_prompt:
            self.prompt.add_part("role", self.role_prompt)
        else:
            self.add_role_prompt()

        # Add theory if requested
        if include_theory:
            self.add_theory_background()

        # Add context and training data
        if self.data is not None:
            self.add_train_data_intro()
            self.add_train_data()

        # Add procedure
        self.add_procedure_prompt()

        # Add answer format based on classification type
        if is_multi_label:
            self.add_answer_format_multi()
        else:
            self.add_answer_format_single()

        # Add input text
        self.prompt.add_part("input", f"Text to analyze: {text}")

        return self.prompt.render()

    def get_full_prompt(self) -> str:
        return self.prompt.render()

    def reset(self):
        self.prompt = Prompt()

    def set_role(self, role_prompt: str) -> None:
        """Set a custom role prompt."""
        self.role_prompt = role_prompt

    def clear_role(self) -> None:
        """Clear the role prompt."""
        self.role_prompt = None

    async def call_llm(self, prompt: str) -> str:
        """Call OpenAI API with optional role prompt."""
        try:
            messages = []
            
            # Add system role message only if role prompt is set
            if self.role_prompt:
                messages.append({"role": "system", "content": self.role_prompt})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            response = await openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise ValueError(f"API call failed: {str(e)}")

