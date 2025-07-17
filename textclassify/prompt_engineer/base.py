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
        data: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        multi_label: bool,  # New parameter
        few_shot_mode: str = "few_shot",
        provider: str = "openai",
        model_name: str = "gpt-4"
    ):
        """Initialize PromptEngineer.
        
        Args:
            data: DataFrame containing text and labels
            text_column: Name of the column containing text
            label_columns: Names of the columns containing labels
            multi_label: Whether this is a multi-label (True) or single-label (False) classification
            few_shot_mode: Mode for few-shot learning (default: "few_shot")
            provider: LLM provider name (default: "openai")
            model_name: Name of the model to use (default: "gpt-4")
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        self.data = data
        self.text_column = text_column
        self.label_columns = label_columns
        self.multi_label = multi_label  # Store the classification type
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
        self.data: Optional[pd.DataFrame] = None
    
    def set_few_shot_mode(self, mode: str):
        assert mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
        self.few_shot_mode = mode


    async def engineer_prompts(
        self,
        data: Optional[pd.DataFrame] = None,
        sample_size: int = 20,
        custom_prompts: Optional[Dict[str, str]] = None,
        custom_role_prompt: Optional[str] = None
    ) -> List[Prompt]:
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
        prompts = []
        
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
            include_role=False,
            keywords_content=context_keywords  # Pass the generated keywords
        )
        context = await self.llm_generator.generate_content(context_brainstorm_str)
        
        init_p.add_part("context", context)
        prompts = []

        for idx, row in self.data.iterrows():
            p = Prompt()
            p.fuse(init_p.copy())
            # Add all components as parts
            procedure_prompt_creator_prompt_str = self.fill_procedure_prompt_creator_prompt(
                sample_size=sample_size,
                custom_prompt=PromptWarehouse.procedure_prompt_creator_prompt,
                include_role=False,
            )

            p.add_part("procedure_prompt", await self.llm_generator.generate_content(procedure_prompt_creator_prompt_str))
            
            
            
            
            train_data_intro_str = self.fill_train_data_intro_prompt(
                sample_size=sample_size,
                custom_prompt=PromptWarehouse.train_data_intro_prompt,
                custom_role_prompt=custom_role_prompt,
                include_role=False,
            )
            p.add_part("train_data_intro_prompt", await self.llm_generator.generate_content(train_data_intro_str))
            
            
            train_data_str = self.fill_train_data_prompt(
                sample_size=sample_size,
                include_role=False
            )
            
            p.add_part("train_data_prompt", train_data_str)
            
            p.add_part("answer_format_prompt", )
            
            # Add to collection
            prompts.append(p)
        return prompts

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
            prompt_template = PromptWarehouse.role_prompt_creator_prompt
            
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
            prompt_template = PromptWarehouse.context_brainstorm_prompt
        
        prompt_text = prompt_template.format(examples=formatted_examples)
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_procedure_prompt_creator_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        context_content: Optional[str] = None
    ) -> str:
        """Fill a procedure prompt creator prompt using sampled data.
        
        Args:
            sample_size: Number of examples to use for procedure creation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            context_content: Optional context to include in the prompt
            
        Returns:
            str: Procedure prompt creator content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for procedure prompt creation")
        
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
        if custom_prompt and context_content:
            prompt_template = custom_prompt.format(context=context_content)
        else:
            prompt_template = PromptWarehouse.procedure_prompt_creator_prompt
        
        prompt_text = prompt_template.format(
            data=formatted_examples,
            features=", ".join(self.label_columns)
        )
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_train_data_intro_prompt(
        self,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        procedure_content: Optional[str] = None
    ) -> str:
        """Fill a training data introduction prompt using sampled data.
    
        Args:
            sample_size: Number of examples to use for training intro (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            procedure_content: Optional procedure content to include in the prompt
        
        Returns:
            str: Training data introduction prompt content
            
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for training data intro")
    
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
        if custom_prompt and procedure_content:
            prompt_template = custom_prompt.format(procedure=procedure_content)
        else:
            prompt_template = PromptWarehouse.train_data_intro_prompt
    
        prompt_text = prompt_template.format(
            data=formatted_examples,
            features=", ".join(self.label_columns)
        )
    
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def _create_output_format(self, multi_label: bool) -> str:
        """Create binary vector format based on classification type.
    
        Args:
            multi_label: Whether to create multi-label (True) or single-label (False) format
        
        Returns:
            str: Formatted binary vector string (e.g. "1 | 0 | 0" or "1 | 1 | 0")
        """
        num_labels = len(self.label_columns)
    
        if multi_label:
            # For multi-label, randomly select one position to be 1
            idx = random.randint(0, num_labels - 1)
            return ' | '.join('1' if i == idx else '0' for i in range(num_labels))
        else:
            # For single-label, randomly select k positions to be 1
            k = random.randint(1, num_labels)
            selected_indices = set(random.sample(range(num_labels), k))
            return ' | '.join('1' if i in selected_indices else '0' for i in range(num_labels))

    def fill_answer_format_prompt(
        self,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Fill the appropriate answer format prompt based on classification type.
    
        Args:
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
        
        Returns:
            str: Formatted answer format prompt
        
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for answer format")

        # Get appropriate template
        prompt_template = custom_prompt if custom_prompt else (
            PromptWarehouse.answer_format_prompt_mult
            if self.multi_label
            else PromptWarehouse.answer_format_prompt_single
        )
        
        # Create formatted output example
        output_format = self._create_output_format(self.multi_label)
        
        # Format the template
        prompt_text = prompt_template.format(
            output_format=output_format,
            labels=' | '.join(self.label_columns)
        )
        
        # Add role prompt if requested
        if not include_role:
            return prompt_text
            
        if custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
            
        return prompt_text

    def fill_train_data_prompt(
        self,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Fill training data prompt with examples based on few-shot mode.
    
        Args:
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
        
        Returns:
            str: Formatted training data prompt
        
        Raises:
            ValueError: If no data is available
        """
        if self.data is None:
            raise ValueError("No data available for training data")

        # Determine number of examples based on few-shot mode
        num_examples = {
            "zero_shot": 0,
            "one_shot": 1,
            "few_shot": 5,
            "full_coverage": len(self.data)
        }[self.few_shot_mode]
    
        if num_examples == 0:
            return ""  # No examples for zero-shot
        
        # Sample data
        sampled_data = self.data.sample(n=min(num_examples, len(self.data)))
    
        # Format examples
        formatted_examples = []
        for idx, row in sampled_data.iterrows():
            text = row[self.text_column]
            label_values = ' | '.join(str(row[col]) for col in self.label_columns)
            formatted_examples.append(
                f"Example {idx + 1}:\n{text}\n{label_values}"
            )
    
        # Join examples with double newlines
        train_data_content = "\n\n".join(formatted_examples)
    
        # Use custom prompt or default template
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = PromptWarehouse.train_data_prompt
    
        # Format final prompt
        prompt_text = prompt_template.format(
            train_data=train_data_content,
            features=", ".join(self.label_columns)
        )
    
        # Add role prompt if requested
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text
