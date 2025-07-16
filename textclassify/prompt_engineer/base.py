import random
import pandas as pd
from typing import List, Optional, Dict, Any
from prompt_warehouse import PromptWarehouse
from .prompt_collection import PromptCollection
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager
from ..prompt_engineer.prompt import Prompt

class PromptEngineer:
    """Builds prompts for LLM-based text classification."""

    def __init__(
        self,
        text_column: str,
        label_columns: List[str],
        few_shot_mode: str = "few_shot",
        provider: str = "openai",
        model_name: str = "gpt-4"
    ):
        self.text_column = text_column
        self.label_columns = label_columns
        self.few_shot_mode = few_shot_mode
        
        # Get API key from environment
        key_manager = APIKeyManager()
        api_key = key_manager.get_key(provider)
        if not api_key:
            raise ValueError(f"No API key found for {provider}")
            
        # Initialize LLM generator
        self.llm_generator = create_llm_generator(
            provider=provider,
            model_name=model_name,
            api_key=api_key
        )
        
        self.role_prompt = None
        self.collection = PromptCollection()
        
    
    def set_few_shot_mode(self, mode: str):
        assert mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
        self.few_shot_mode = mode

    # def set_examples(self, texts, labels):
    #     n = len(texts)
    #     indices = list(range(n))
    #     if self.few_shot_mode == "zero_shot":
    #         self.examples = []
    #     elif self.few_shot_mode == "one_shot":
    #         if n > 0:
    #             idx = random.choice(indices)
    #             self.examples = [{'text': texts[idx], 'label': labels[idx]}]
    #         else:
    #             self.examples = []
    #     elif self.few_shot_mode == "few_shot":
    #         k = min(5, n)
    #         if k > 0:
    #             idxs = random.sample(indices, k)
    #             self.examples = [{'text': texts[i], 'label': labels[i]} for i in idxs]
    #         else:
    #             self.examples = []
    #     elif self.few_shot_mode == "full_coverage":
    #         if n > 0:
    #             idxs = random.sample(indices, n)
    #             self.examples = [{'text': texts[i], 'label': labels[i]} for i in idxs]
    #         else:
    #             self.examples = []
    #     else:
    #         self.examples = []

    
    async def engineer_prompts(
        self,
        data: Optional[pd.DataFrame] = None,
        sample_size: int = 20,
        custom_prompts: Optional[Dict[str, str]] = None,
        custom_role_prompt: Optional[str] = None
    ):
        """Engineer prompts using LLM-based generation.
        
        Args:
            data: Optional DataFrame to use for prompt engineering
            sample_size: Number of examples to use for each prompt generation (default: 20)
            custom_prompts: Optional dict of custom prompts for each stage (role, context, keywords)
            custom_role_prompt: Optional custom role prompt to use
        """
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            self.data = data
        elif self.data is None:
            raise ValueError("No data available for prompt engineering")
        
        init_p = Prompt()
        pc = PromptCollection
        
        # Get custom prompts if provided
        role_prompt = custom_prompts.get('role') if custom_prompts else None
        context_prompt = custom_prompts.get('context') if custom_prompts else None
        keywords_prompt = custom_prompts.get('keywords') if custom_prompts else None
        
        # Generate role prompt
        role_prompt_creator_prompt = self.fill_role_prompt_creator_prompt(
            sample_size=sample_size,
            custom_prompt=PromptWarehouse.role_prompt_creator_prompt,
            custom_role_prompt=custom_role_prompt,
            include_role=False
        )

        role_prompt_str = await self.llm_generator.generate_content(role_prompt_creator_prompt)

        init_p.add_part("role_prompt", role_prompt_str)

        create_context_keywords_str = self.generate_context_keywords(
            sample_size=sample_size,
            custom_prompt=PromptWarehouse.context_brainstorm_role_prompt,
            custom_role_prompt=PromptWarehouse.context_brainstorm_prompt,
            include_role=True  # Include the newly generated role prompt
        )

        context_keywords = await self.llm_generator.generate_content(create_context_keywords_str)


        # Generate context brainstorm with the new role
        context_brainstorm_str = self.fill_context_brainstorm_prompt(
            sample_size=sample_size,
            custom_prompt=PromptWarehouse.create_context_prompt,
            custom_role_prompt=custom_role_prompt,
            include_role=True,
            keywords_content=context_keywords  # Pass the generated keywords
        )
        context = await self.llm_generator.generate_content(context_brainstorm_str)
        
        init_p.add_part("context", context)
        pc = PromptCollection()

        for idx, row in self.data.iterrows():
            p = Prompt()
            p.fuse(init_p.copy())
            # Add all components as parts
            p.add_part("procedure_prompt", )
            p.add_part("train_data_intro_prompt", )
            p.add_part("answer_format_prompt",)
            
            # Add to collection
            self.collection.add_prompt(str(idx), p)
        return pc
        

    def generate_role_prompt_creator_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Generate a role prompt creator prompt using sampled data with text-label pairs.
        
        Args:
            sample_size: Number of examples to use for role creation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            
        Returns:
            str: Role prompt creator content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for role prompt creation")
            
        # Sample data
        sampled_data = self.data.sample(n=min(sample_size, len(self.data)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
            
        # Get prompt template
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = PromptWarehouse.get_role_creator_template()
            
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {ex['labels']}"
            for ex in examples
        ])
        
        prompt_text = prompt_template.format(examples=formatted_examples)
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def generate_context_brainstorm_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Generate a context brainstorming prompt using sampled data.
        
        Args:
            sample_size: Number of examples to use for context brainstorming (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            
        Returns:
            str: Context brainstorming prompt content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for context brainstorming")
            
        # Sample data
        sampled_data = self.data.sample(n=min(sample_size, len(self.data)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
            
        # Get prompt template
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = PromptWarehouse.get_context_brainstorm_template()
            
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {ex['labels']}"
            for ex in examples
        ])
        
        prompt_text = prompt_template.format(examples=formatted_examples)
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def generate_context_keywords(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Generate a prompt to extract keywords from context brainstorming.
        
        Args:
            sample_size: Number of examples to use for keyword generation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            
        Returns:
            str: Prompt for generating context keywords
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("Data must be set before generating context keywords")
            
        # Take a random sample of the data
        sampled_data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)
        
        # Format the data
        formatted_data = []
        for idx, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = [f"{col}: {row[col]}" for col in self.label_columns]
            formatted_data.append(
                f"Text {idx + 1}: {text}\n"
                f"Labels: {', '.join(labels)}"
            )
        
        data_str = "\n\n".join(formatted_data)
        prompts = []
        
        # Add role prompt if requested
        if include_role:
            role_prompt = custom_role_prompt or PromptWarehouse.context_brainstorm_role_prompt.format(
                data=self.data
            )
            prompts.append(role_prompt)
        
        # Add main prompt
        main_prompt = custom_prompt or PromptWarehouse.create_context_prompt.format(
            data=data_str,
            features=", ".join(self.label_columns)
        )
        prompts.append(main_prompt)
        
        return "\n\n".join(prompts)

    def add_role_prompt(self, base_prompt: str = "") -> str:
        """Add role prompt to the base prompt.
        
        Args:
            base_prompt: Existing prompt text to append to (default: "")
            
        Returns:
            str: Combined prompt text
        """
        content = self.role_prompt or PromptWarehouse.get_role_prompt()
        return f"{content}\n\n{base_prompt}" if base_prompt else content

    def add_theory_background(self, base_prompt: str) -> str:
        """Add theory background to the base prompt."""
        theory = PromptWarehouse.get_theory_background()
        return f"{base_prompt}\n\n{theory}"

    def add_train_data_intro(self, base_prompt: str) -> str:
        """Add training data introduction to the base prompt."""
        intro = PromptWarehouse.get_train_data_intro()
        return f"{base_prompt}\n\n{intro}"

    def add_train_data(self, base_prompt: str) -> str:
        """Add training data prompt to the base prompt."""
        if self.data is None:
            return base_prompt
            
        train_data = PromptWarehouse.get_train_data_prompt(self.data.columns.tolist())
        return f"{base_prompt}\n\n{train_data}"

    def add_context_brainstorm(self, base_prompt: str, sample_size: int = 20) -> str:
        """Add context brainstorm to the base prompt."""
        if self.data is None:
            return base_prompt
            
        context = PromptWarehouse.get_context_brainstorm(
            self.data, 
            self.label_columns, 
            sample_size
        )
        return f"{base_prompt}\n\n{context}"

    def add_context_brainstorm_role(self, base_prompt: str) -> str:
        """Add context brainstorm role to the base prompt."""
        if self.data is None:
            return base_prompt
            
        role = PromptWarehouse.get_context_brainstorm_role(self.data.columns.tolist())
        return f"{base_prompt}\n\n{role}"

    def add_create_context(self, base_prompt: str, keywords: List[str]) -> str:
        """Add context creation prompt to the base prompt."""
        context = PromptWarehouse.get_create_context(keywords)
        return f"{base_prompt}\n\n{context}"

    def add_procedure_prompt(self, base_prompt: str) -> str:
        """Add procedure prompt to the base prompt."""
        procedure = PromptWarehouse.get_procedure_prompt()
        return f"{base_prompt}\n\n{procedure}"

    def add_procedure_creator(self, base_prompt: str) -> str:
        """Add procedure creator prompt to the base prompt."""
        if self.data is None:
            return base_prompt
            
        creator = PromptWarehouse.get_procedure_creator(self.data)
        return f"{base_prompt}\n\n{creator}"

    def add_answer_format(self, base_prompt: str, is_multi_label: bool) -> str:
        """Add appropriate answer format to the base prompt."""
        if is_multi_label:
            format_text = PromptWarehouse.get_answer_format_multi(self.label_columns)
        else:
            format_text = PromptWarehouse.get_answer_format_single(self.label_columns)
        return f"{base_prompt}\n\n{format_text}"

    def add_input_text(self, base_prompt: str, text: str) -> str:
        """Add input text to the base prompt."""
        return f"{base_prompt}\n\nText to analyze: {text}"

    def build_complete_prompt(
        self,
        text: str,
        is_multi_label: bool = False,
        include_theory: bool = True
    ) -> str:
        """Build complete prompt with all components."""
        # Start with role prompt
        prompt = self.add_role_prompt()

        # Add theory if requested
        if include_theory:
            prompt = self.add_theory_background(prompt)

        # Add context and training data
        if self.data is not None:
            prompt = self.add_train_data_intro(prompt)
            prompt = self.add_train_data(prompt)

        # Add procedure
        prompt = self.add_procedure_prompt(prompt)

        # Add answer format based on classification type
        prompt = self.add_answer_format(prompt, is_multi_label)

        # Add input text
        prompt = self.add_input_text(prompt, text)

        return prompt

    def get_full_prompt(self) -> str:
        return self.prompt.render()

    def set_role(self, role_prompt: str) -> None:
        """Set a custom role prompt."""
        self.role_prompt = role_prompt

    def clear_role(self) -> None:
        """Clear the role prompt."""
        self.role_prompt = None

    def fill_role_prompt_creator_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Fill a role prompt creator prompt using sampled data with text-label pairs.
        
        Args:
            sample_size: Number of examples to use for role creation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            
        Returns:
            str: Role prompt creator content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for role prompt creation")
            
        # Sample data
        sampled_data = self.data.sample(n=min(sample_size, len(self.data)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
            
        # Get prompt template
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = PromptWarehouse.get_role_creator_template()
            
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {ex['labels']}"
            for ex in examples
        ])
        
        prompt_text = prompt_template.format(examples=formatted_examples)
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_context_brainstorm_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        keywords_content: Optional[str] = None
    ) -> str:
        """Fill a context brainstorming prompt using sampled data.
    
        Args:
            sample_size: Number of examples to use for context brainstorming (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            keywords_content: Optional keywords to include in the prompt
        
        Returns:
            str: Context brainstorming prompt content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for context brainstorming")
        
        # Sample data
        sampled_data = self.data.sample(n=min(sample_size, len(self.data)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
        
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {ex['labels']}"
            for ex in examples
        ])
        
        # Use custom prompt or default template
        if custom_prompt and keywords_content:
            prompt_template = custom_prompt.format(keywords=keywords_content)
        else:
            prompt_template = PromptWarehouse.get_context_brainstorm_template()
        
        prompt_text = prompt_template.format(examples=formatted_examples)
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

