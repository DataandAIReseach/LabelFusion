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
        self.prompt = Prompt()
        self.examples = []
        self.few_shot_mode = few_shot_mode
        self.role_prompt = None
        self.data = None

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

    def generate_role_prompt(self) -> None:
        self.prompt.add_part("role", PromptWarehouse.role_prompt)

    def generate_context_prompt(self, label_type: str = "single", mode: str = "auto") -> None:
        context = PromptWarehouse.get_context_prompt(label_type, mode)
        self.prompt.add_part("context", context)

    def generate_feat_def_prompt(self) -> None:
        self.prompt.add_part("feature_def", "The features are defined as follows:\n" + PromptWarehouse.answer_format_prompt)

    def generate_procedure_prompt(self) -> None:
        self.prompt.add_part("procedure", PromptWarehouse.procedure_prompt)

    def generate_input_prompt(self, input_text: str) -> None:
        self.prompt.add_part("input", f"Text: {input_text}")

    def generate_answer_format_prompt(self) -> None:
        self.prompt.add_part("answer_format", PromptWarehouse.answer_format_prompt)

    def generate_examples_prompt(self):
        if self.examples:
            examples_str = "\n".join(
                f"Example:\nText: {ex['text']}\nLabel: {ex['label']}" for ex in self.examples
            )
            self.prompt.add_part("examples", examples_str)

    def add_role_prompt(self) -> None:
        """Add default role prompt."""
        self.prompt.add_part("role", PromptWarehouse.role_prompt)

    def add_theory_background(self) -> None:
        """Add theory background."""
        self.prompt.add_part("theory", PromptWarehouse.theory_background)

    def add_procedure_prompt(self) -> None:
        """Add default procedure prompt."""
        self.prompt.add_part("procedure", PromptWarehouse.procedure_prompt)

    def add_train_data_intro(self) -> None:
        """Add training data introduction."""
        if self.data is not None:
            prompt = PromptWarehouse.train_data_intro_prompt
            self.prompt.add_part("train_intro", prompt)

    def add_context_brainstorm(self) -> None:
        """Add context brainstorming prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.context_brainstorm_prompt.format(
                data=self.data.to_string(),
                features=self.label_columns
            )
            self.prompt.add_part("context_brainstorm", prompt)

    def add_context_brainstorm_role(self) -> None:
        """Add context brainstorming role prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.context_brainstorm_role_prompt.format(
                data=self.data.columns
            )
            self.prompt.add_part("context_brainstorm_role", prompt)

    def add_create_context(self, keywords: List[str]) -> None:
        """Add context creation prompt."""
        prompt = PromptWarehouse.create_context_prompt.format(
            keywords=keywords
        )
        self.prompt.add_part("create_context", prompt)

    def add_procedure_creator(self) -> None:
        """Add procedure creator prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.procedure_prompt_creator_prompt.format(
                data=self.data.to_string()
            )
            self.prompt.add_part("procedure_creator", prompt)

    def add_train_data(self) -> None:
        """Add training data prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.train_data_prompt.format(
                data=self.data.columns
            )
            self.prompt.add_part("train_data", prompt)

    def add_answer_format_single(self) -> None:
        """Add single-label answer format prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.answer_format_prompt_single.format(
                data=self.data
            )
            self.prompt.add_part("answer_format", prompt)

    def add_answer_format_multi(self) -> None:
        """Add multi-label answer format prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.answer_format_prompt_mult.format(
                data=self.data
            )
            self.prompt.add_part("answer_format", prompt)

    def add_role_creator(self) -> None:
        """Add role creator prompt."""
        if self.data is not None:
            prompt = PromptWarehouse.role_prompt_creator_prompt.format(
                data=self.data.to_string()
            )
            self.prompt.add_part("role_creator", prompt)

    def build_prompt_single_label(self, input_text: str, mode: str = "auto") -> str:
        self.reset()
        self.generate_role_prompt()
        self.generate_context_prompt(label_type="single", mode=mode)
        self.generate_feat_def_prompt()
        self.generate_examples_prompt()
        self.generate_procedure_prompt()
        self.generate_answer_format_prompt()
        self.generate_input_prompt(input_text)
        return self.prompt.render()

    def build_prompt_multiple_labels(self, input_text: str, mode: str = "auto") -> str:
        self.reset()
        self.generate_role_prompt()
        self.generate_context_prompt(label_type="multiple", mode=mode)
        self.generate_feat_def_prompt()
        self.generate_examples_prompt()
        self.generate_procedure_prompt()
        self.generate_answer_format_prompt()
        self.generate_input_prompt(input_text)
        return self.prompt.render()


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

    async def get_classification(
        self,
        text: str,
        is_multi_label: bool = False,
        include_theory: bool = True
    ) -> str:
        """Get classification for input text."""
        self.reset()
        
        # Add prompts in sequence
        if self.role_prompt:
            self.prompt.add_part("role", self.role_prompt)
        else:
            self.prompt.add_part("role", PromptWarehouse.role_prompt)

        if include_theory:
            self.prompt.add_part("theory", PromptWarehouse.theory_background)

        # Add examples if available
        if self.examples:
            self.add_examples_prompt()

        # Add procedure
        self.prompt.add_part("procedure", PromptWarehouse.procedure_prompt)

        # Add answer format
        format_prompt = (PromptWarehouse.answer_format_prompt_mult 
                        if is_multi_label 
                        else PromptWarehouse.answer_format_prompt_single)
        self.prompt.add_part("answer_format", format_prompt)

        # Add input text
        self.prompt.add_part("input", f"Text to analyze: {text}")

        # Get LLM response
        return await self.call_llm(self.prompt.render())