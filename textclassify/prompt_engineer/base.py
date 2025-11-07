import random
import pandas as pd
from typing import List, Optional, Dict, Any
from .prompt_warehouse import PromptWarehouse  # Changed to relative import
from .prompt_collection import PromptCollection
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager
from ..prompt_engineer.prompt import Prompt
import copy
import logging
from tqdm import tqdm
import time

class PromptEngineer:
    """Builds prompts for LLM-based text classification."""

    def __init__(
        self,
        text_column: str,
        label_columns: List[str],
        multi_label: bool,
        few_shot_mode = "few_shot",  # Changed type hint to be more flexible
        provider: str = "openai",
        model_name: str = "gpt-4"  # Default model name
    ):
        """Initialize PromptEngineer.
        
        Args:
            text_column: Name of the column containing text
            label_columns: Names of the columns containing labels
            multi_label: Whether this is a multi-label (True) or single-label (False) classification
            few_shot_mode: Mode for few-shot learning (default: "few_shot") - can be string or int
            provider: LLM provider name (default: "openai")
            model_name: Name of the model to use
        """
        self.text_column = text_column
        self.label_columns = label_columns
        self.multi_label = multi_label
        # Validate and set few_shot_mode
        if isinstance(few_shot_mode, str):
            assert few_shot_mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
            self.few_shot_mode = few_shot_mode
        elif isinstance(few_shot_mode, int):
            assert few_shot_mode >= 0, "Number of examples must be non-negative"
            self.few_shot_mode = few_shot_mode
        else:
            raise ValueError("few_shot_mode must be a string or integer")
        self.model_name = model_name  # Store model_name as an instance variable
        
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
    
    def set_few_shot_mode(self, mode):
        """Set the few-shot mode for training examples.
        
        Args:
            mode: Can be a string ("zero_shot", "one_shot", "few_shot", "full_coverage") 
                  or an integer specifying exact number of examples
        """
        if isinstance(mode, str):
            assert mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
            self.few_shot_mode = mode
        elif isinstance(mode, int):
            assert mode >= 0, "Number of examples must be non-negative"
            self.few_shot_mode = mode
        else:
            raise ValueError("Mode must be a string or integer")


    async def engineer_prompts(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompts: Optional[Dict[str, str]] = None,
        custom_role_prompt: Optional[str] = None,
        custom_context: Optional[str] = None,
        procedure_additions: Optional[str] = None
    ) -> List[Prompt]:
        """Engineer prompts for test data using training examples.
        
        Args:    OpenAI alternative (commented out)
    classifier = OpenAIClassifier(
        config=config,
        text_column='text',
        label_columns=label_columns  # Use the dummy column names as labels
    )
            test_df: DataFrame containing texts to classify
            train_df: DataFrame containing training examples
            sample_size: Number of examples for prompt generation
            custom_prompts: Optional dict of custom prompts
            custom_role_prompt: Optional custom role prompt
            custom_context: Optional pre-existing context to use instead of generating one
            procedure_additions: Optional additional content for procedure prompt (e.g., theory, background)
            
        Returns:
            List[Prompt]: List of engineered prompts for each test text
        """
        if not isinstance(test_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
            raise ValueError("test_df and train_df must be pandas DataFrames")
        
        # Initialize base prompt with shared components
        init_p = Prompt()
        
        # Generate role prompt
        role_prompt_str = await self.llm_generator.generate_content(
            self.fill_role_prompt_creator_prompt(
                train_df=train_df,
                sample_size=sample_size,
                custom_prompt=custom_prompts.get('role') if custom_prompts else None,
                custom_role_prompt=custom_role_prompt,
                include_role=False
            )
        )

        init_p.add_part("role_prompt", role_prompt_str)
        
        # Generate context keywords
        context_keywords = await self.llm_generator.generate_content(
            self.generate_context_keywords(
                train_df=train_df,
                sample_size=sample_size,
                custom_prompt=custom_prompts.get('keywords') if custom_prompts else None,
                include_role=True
            )
        )
        
        # Use custom context if provided, otherwise generate context
        if custom_context:
            context = custom_context
        else:
            context = await self.llm_generator.generate_content(
                self.fill_context_prompt(
                    train_df=train_df,
                    sample_size=sample_size,
                    custom_prompt=custom_prompts.get('context') if custom_prompts else None,
                    include_role=False,
                    keywords_content=context_keywords
                )
            )
        init_p.add_part("context", context)

        procedure_prompt = await self.llm_generator.generate_content(
            self.fill_procedure_prompt_creator_prompt(
                    train_df=train_df,
                    sample_size=sample_size,
                    custom_prompt=custom_prompts.get('procedure') if custom_prompts else None,
                    include_role=False,
                    procedure_additions=procedure_additions
            )
        )

        init_p.add_part("procedure_prompt", procedure_prompt)

        # Generate prompts for each test text
        prompts = []
        for _, row in test_df.iterrows():
            p = Prompt()
            p.fuse(copy.deepcopy(init_p))
            
            # Add procedure
                  
            # Add train data components
            train_data_intro_prompt = self.fill_train_data_intro_prompt(
                    train_df=train_df,
                    sample_size=sample_size,
                    custom_prompt=custom_prompts.get('train_intro') if custom_prompts else None,
                    include_role=False
            )

            p.add_part("train_data_intro_prompt", train_data_intro_prompt)

            train_data = self.fill_train_data_prompt(
                train_df=train_df,  # Add train_df parameter
                custom_role_prompt=custom_prompts.get('train_data') if custom_prompts else None,
                include_role=False
            )
            p.add_part("train_data_prompt", train_data)
            
            # Add answer format
            answer_format = self.fill_answer_format_prompt(
                row=row,
                custom_prompt=custom_prompts.get('answer_format') if custom_prompts else None,
                include_role=False
            )
            p.add_part("answer_format_prompt", answer_format)
            
            prompts.append(p)
        
        return prompts

    def generate_context_keywords(
        self,
        train_df: pd.DataFrame,
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
        if train_df is None:
            raise ValueError("Data must be set before generating context keywords")
            
        # Take a random sample of the data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

        # Format the data
        formatted_data = []
        for idx, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = [f"{col}: {row[col]}" for col in self.label_columns]
            formatted_data.append(
                f"Text {idx + 1}: {text}\n"
                f"Labels: {', '.join(labels)}"
            )
        
        examples = "\n\n".join(formatted_data)
        prompts = []
        
        # Add role prompt if requested
        if include_role:
            role_prompt = custom_role_prompt or PromptWarehouse.context_brainstorm_role_prompt.format(
                labels=", ".join(self.label_columns)
        )
        prompts.append(role_prompt)
    
        # Add main prompt
        main_prompt = custom_prompt or PromptWarehouse.brainstorm_context_keywords_prompt.format(
            examples=examples,
            labels=", ".join(self.label_columns)
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
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        include_multilabel_info: bool = True
    ) -> str:
        """Fill a role prompt creator prompt using sampled data with text-label pairs.
        
        Args:
            train_df: DataFrame containing training examples
            sample_size: Number of examples to use for role creation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            include_multilabel_info: Whether to include multilabel classification info (default: True)
            
        Returns:
            str: Role prompt creator content
            
        Raises:
            ValueError: If no data is available
        """
        if train_df is None:
            raise ValueError("No data available for role prompt creation")
            
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
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
            f"Text: {ex['text']}\nLabels: {' | '.join(str(ex['labels'][col]) for col in self.label_columns)}"
            for ex in examples
        ])
        
        # Add multilabel information if requested
        multilabel_info = ""
        if include_multilabel_info:
            classification_type = "multi-label" if self.multi_label else "single-label"
            multilabel_info = f"\n\nClassification Type: This is a {classification_type} classification task."
            if self.multi_label:
                multilabel_info += " Multiple labels can be assigned to each text."
            else:
                multilabel_info += " Only one label should be assigned to each text."
        
        # Format the template
        try:
            if include_multilabel_info and "{multilabel_info}" in prompt_template:
                prompt_text = prompt_template.format(
                    examples=formatted_examples,
                    multilabel_info=multilabel_info
                )
            else:
                prompt_text = prompt_template.format(examples=formatted_examples)
                if include_multilabel_info:
                    prompt_text += multilabel_info
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_context_prompt(
        self,
        train_df: pd.DataFrame,
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
        if train_df is None:
            raise ValueError("No data available for context brainstorming")
        
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
        
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {' | '.join(str(ex['labels'][col]) for col in self.label_columns)}"
            for ex in examples
        ])
        
        # Use custom prompt or default template
        if custom_prompt and keywords_content:
            prompt_template = custom_prompt.format(keywords=keywords_content)
        else:
            prompt_template = PromptWarehouse.create_context_prompt
        
        # Format prompt with examples and labels
        prompt_text = prompt_template.format(
            examples=formatted_examples,
            labels=", ".join(self.label_columns),
            keywords=keywords_content or ""
        )
        
        # Add role prompt if requested
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_procedure_prompt_creator_prompt(
        self,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        context_content: Optional[str] = None,
        procedure_additions: Optional[str] = None
    ) -> str:
        """Fill a procedure prompt creator prompt using sampled data.
        
        Args:
            sample_size: Number of examples to use for procedure creation (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
            context_content: Optional context to include in the prompt
            procedure_additions: Optional additional content (e.g., theory, background) to include
            
        Returns:
            str: Procedure prompt creator content
            
        Raises:
            ValueError: If no data is available
        """
        if train_df is None:
            raise ValueError("No data available for procedure prompt creation")
        
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
        
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {' | '.join(str(ex['labels'][col]) for col in self.label_columns)}"
            for ex in examples
        ])
        
        # Use custom prompt or default template
        if custom_prompt and context_content:
            prompt_template = custom_prompt.format(context=context_content)
        else:
            prompt_template = PromptWarehouse.procedure_prompt_creator_prompt
        
        prompt_text = prompt_template.format(
            data=formatted_examples,
            labels=", ".join(self.label_columns)
        )
        
        # Add procedure additions if provided
        if procedure_additions:
            prompt_text = f"{prompt_text}\n\nAdditional Context/Theory:\n{procedure_additions}\n\nIncorporate this additional information into your procedure prompt where appropriate."
        
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def fill_train_data_intro_prompt(
        self,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        procedure_content: Optional[str] = None  # Consider removing if not needed
    ) -> str:
        """Fill a training data introduction prompt using sampled data.
    
        Args:
            train_df: DataFrame containing training examples
            sample_size: Number of examples to use for training intro (default: 20)
            custom_prompt: Optional custom prompt to use instead of default
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)
    
        Returns:
            str: Training data introduction prompt content
    
        Raises:
            ValueError: If train_df is None
        """
        if train_df is None:
            raise ValueError("No data available for training data intro")
        
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
        
        # Format examples
        formatted_examples = []
        for idx, row in sampled_data.iterrows():
            text = row[self.text_column]
            label_values = '|'.join(str(row[col]) for col in self.label_columns)
            formatted_examples.append(
                f"Text {idx + 1}:\n{text}\n{label_values}"
            )
        
        # Join examples with double newlines
        examples_text = "\n\n".join(formatted_examples)
        
        # Select and format template
        if custom_prompt and procedure_content:
            prompt_template = custom_prompt.format(procedure=procedure_content)
        else:
            prompt_template = PromptWarehouse.train_data_intro_prompt
        
        try:
            prompt_text = prompt_template.format(
                examples=examples_text,
                labels=", ".join(self.label_columns)
            )
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")
        
        # Add role prompt if requested
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
        row: pd.Series,
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
        if row is None:
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
            paragraph=row[self.text_column],
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
        train_df: pd.DataFrame,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Generate a training data prompt directly from train_df.

        Args:
            train_df: DataFrame containing training examples
            custom_role_prompt: Optional custom role prompt to use instead of default
            include_role: Whether to include the role prompt (default: True)

        Returns:
            str: Generated training data prompt

        Raises:
            ValueError: If no data is available
        """
        if train_df is None or train_df.empty:
            raise ValueError("No data available for training data")

        # Determine number of examples based on few-shot mode
        if isinstance(self.few_shot_mode, int):
            # Direct numeric specification
            num_examples = self.few_shot_mode
        else:
            # String-based mode
            num_examples = {
                "zero_shot": 0,
                "one_shot": 1,
                "few_shot": 5,
                "full_coverage": len(train_df)
            }[self.few_shot_mode]

        if num_examples == 0:
            return ""  # No examples for zero-shot

        # Sample data
        sampled_data = train_df.sample(n=min(num_examples, len(train_df)))

        # Generate the training data prompt
        prompt_lines = []
        prompt_lines.append("Training Data Examples:")
        for idx, row in sampled_data.iterrows():
            text = row[self.text_column]
            label_values = ' | '.join(str(row[col]) for col in self.label_columns)
            prompt_lines.append(f"Example {idx + 1}:")
            prompt_lines.append(f"Text: {text}")
            prompt_lines.append(f"Ratings: {label_values}")
            prompt_lines.append("")  # Add a blank line between examples

        # Combine all lines into the final prompt
        prompt_text = "\n".join(prompt_lines).strip()

        # Add role prompt if requested
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text
    
    def fill_procedure_prompt_creator_prompt(
        self,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        context_content: Optional[str] = None,
        procedure_additions: Optional[str] = None
    ) -> str:
        """Fill a procedure prompt creator prompt using sampled data.
    
        Args:
            train_df: DataFrame containing training examples
            sample_size: Number of examples to use for procedure creation
            custom_prompt: Optional custom prompt template
            custom_role_prompt: Optional custom role prompt
            include_role: Whether to include the role prompt
            context_content: Optional context to include in the prompt
            procedure_additions: Optional additional content (e.g., theory, background) to include
    
        Returns:
            str: Procedure prompt creator content
    
        Raises:
            ValueError: If train_df is None
        """
        if train_df is None:
            raise ValueError("No data available for procedure prompt creation")
    
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
    
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {' | '.join(str(ex['labels'][col]) for col in self.label_columns)}"
            for ex in examples
        ])
    
        # Select prompt template
        if custom_prompt:
            prompt_template = custom_prompt
            # Only apply context formatting if template expects it
            if context_content and '{context}' in custom_prompt:
                prompt_template = prompt_template.format(context=context_content)
        else:
            prompt_template = PromptWarehouse.procedure_prompt_creator_prompt
    
        # Format with available data
        try:
            prompt_text = prompt_template.format(
                data=formatted_examples,
                labels=", ".join(self.label_columns)
            )
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")
    
        # Add procedure additions if provided
        if procedure_additions:
            prompt_text = f"{prompt_text}\n\nAdditional Context/Theory:\n{procedure_additions}\n\nIncorporate this additional information into your procedure prompt where appropriate."
    
        # Add role prompt if requested
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text

    def build_prompt(self, text: str) -> str:
        """Build a simple classification prompt for a single text.
        
        This is a simplified method that creates a basic classification prompt
        without the full engineering process.
        
        Args:
            text: The text to classify
            
        Returns:
            str: The formatted prompt for classification
        """
        # Create a simple classification prompt
        labels_str = ", ".join(self.label_columns)
        
        if self.multi_label:
            prompt = f"""Classify the following text into one or more of these categories: {labels_str}

        Text: {text}

        Please respond with the relevant category names separated by commas, or "none" if no categories apply."""
        else:
            prompt = f"""Classify the following text into one of these categories: {labels_str}

Text: {text}

Please respond with only the most relevant category name."""
        
        return prompt
