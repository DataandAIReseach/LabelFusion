import random
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from .prompt_warehouse import PromptWarehouse
from .prompt_collection import PromptCollection
from .base import PromptPipeline
from .nearest_neighbour_sampler import NearestNeighbourSampler
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager
from .prompt import Prompt
import copy

logger = logging.getLogger(__name__)


class PromptEngineer:
    """Builds prompts for LLM-based text classification."""

    def __init__(
        self,
        text_column: str,
        label_columns: List[str],
        multi_label: bool,
        pipeline: PromptPipeline,
        few_shot_mode="few_shot",
        provider: str = "openai",
        model_name: str = "gpt-4",
        sampler: Optional[NearestNeighbourSampler] = None,
    ):
        """Initialize PromptEngineer.

        Args:
            text_column: Name of the column containing text
            label_columns: Names of the columns containing labels
            multi_label: Whether this is a multi-label (True) or single-label (False) classification
            pipeline: PromptPipeline instance — detects language and provides translated warehouse
            few_shot_mode: Mode for few-shot learning (default: "few_shot") - can be string or int
            provider: LLM provider name (default: "openai")
            model_name: Name of the model to use
            sampler: Optional NearestNeighbourSampler — if provided, uses semantic similarity
                     for few-shot example selection instead of random sampling.
                     fit() will be called once in engineer_prompts() before the test loop.
        """
        self.text_column = text_column
        self.label_columns = label_columns
        self.multi_label = multi_label
        self._pipeline = pipeline
        self._sampler = sampler  # injected — PromptEngineer does not create this

        # Default to English warehouse at init time
        # Will be updated at engineer_prompts() time based on actual train_df language
        self.warehouse = pipeline._warehouse

        # Validate and set few_shot_mode
        if isinstance(few_shot_mode, str):
            assert few_shot_mode in ("zero_shot", "one_shot", "few_shot", "full_coverage")
            self.few_shot_mode = few_shot_mode
        elif isinstance(few_shot_mode, int):
            assert few_shot_mode >= 0, "Number of examples must be non-negative"
            self.few_shot_mode = few_shot_mode
        else:
            raise ValueError("few_shot_mode must be a string or integer")

        self.model_name = model_name

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
        self.data = None

    def set_few_shot_mode(self, mode):
        """Set the few-shot mode for training examples."""
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

        Args:
            test_df: DataFrame containing texts to classify
            train_df: DataFrame containing training examples
            sample_size: Number of examples for prompt generation
            custom_prompts: Optional dict of custom prompts
            custom_role_prompt: Optional custom role prompt
            custom_context: Optional pre-existing context to use instead of generating one
            procedure_additions: Optional additional content for procedure prompt

        Returns:
            List[Prompt]: List of engineered prompts for each test text
        """
        if not isinstance(test_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
            raise ValueError("test_df and train_df must be pandas DataFrames")

        # 1. Detect language from train_df and swap warehouse if needed
        first_text = train_df[self.text_column].iloc[0]
        self.warehouse = await self._pipeline.get_warehouse(first_text)

        # 2. Fit sampler once on full train_df — ownership of fit() lifecycle is here
        if self._sampler is not None:
            self._sampler.fit(train_df, self.text_column)
            logger.info(f"NearestNeighbourSampler fitted on {len(train_df)} training samples")

        # 3. Sample once for shared prompt parts (role, context, procedure)
        sampled_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

        # Initialize base prompt with shared components
        init_p = Prompt()

        # Generate role prompt
        role_prompt_str = await self.llm_generator.generate_content(
            self.fill_role_prompt_creator_prompt(
                train_df=sampled_df,
                custom_prompt=custom_prompts.get('role') if custom_prompts else None,
                custom_role_prompt=custom_role_prompt,
                include_role=False
            )
        )
        init_p.add_part("role_prompt", role_prompt_str)

        # Generate context keywords
        context_keywords = await self.llm_generator.generate_content(
            self.generate_context_keywords(
                train_df=sampled_df,
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
                    train_df=sampled_df,
                    custom_prompt=custom_prompts.get('context') if custom_prompts else None,
                    include_role=False,
                    keywords_content=context_keywords
                )
            )
        init_p.add_part("context", context)

        procedure_prompt = await self.llm_generator.generate_content(
            self.fill_procedure_prompt_creator_prompt(
                train_df=sampled_df,
                custom_prompt=custom_prompts.get('procedure') if custom_prompts else None,
                include_role=False,
                procedure_additions=procedure_additions
            )
        )
        init_p.add_part("procedure_prompt", procedure_prompt)

        train_data_intro = self.fill_train_data_intro_prompt(
            train_df=sampled_df,
            custom_prompt=custom_prompts.get('train_intro') if custom_prompts else None,
            include_role=False
        )

        # 4. Generate prompts for each test text
        prompts = []
        for _, row in test_df.iterrows():
            p = Prompt()
            p.fuse(copy.deepcopy(init_p))

            p.add_part("train_data_intro_prompt", train_data_intro)

            train_data = self.fill_train_data_prompt(
                train_df=train_df,
                query_text=row[self.text_column],
                custom_role_prompt=custom_prompts.get('train_data') if custom_prompts else None,
                include_role=False
            )
            p.add_part("train_data_prompt", train_data)

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
        """Generate a prompt to extract keywords from context brainstorming."""
        if train_df is None:
            raise ValueError("Data must be set before generating context keywords")

        examples = "\n\n".join(
            f"Text {idx + 1}: {row[self.text_column]}\n"
            f"Labels: {', '.join(f'{col}: {row[col]}' for col in self.label_columns)}"
            for idx, (_, row) in enumerate(train_df.iterrows())
        )

        prompts = []
        if include_role:
            role_prompt = custom_role_prompt or self.warehouse.context_brainstorm_role_prompt.format(
                labels=", ".join(self.label_columns)
            )
            prompts.append(role_prompt)

        main_prompt = custom_prompt or self.warehouse.brainstorm_context_keywords_prompt.format(
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
        """Fill a role prompt creator prompt using sampled data with text-label pairs."""
        if train_df is None:
            raise ValueError("No data available for role prompt creation")

        examples = [
            {'text': row[self.text_column], 'labels': {col: row[col] for col in self.label_columns}}
            for _, row in train_df.iterrows()
        ]

        prompt_template = custom_prompt or self.warehouse.role_prompt_creator_prompt

        formatted_examples = "\n".join(
            f"Text: {ex['text']}\nLabels: {' | '.join(str(ex['labels'][col]) for col in self.label_columns)}"
            for ex in examples
        )

        multilabel_info = ""
        if include_multilabel_info:
            classification_type = "multi-label" if self.multi_label else "single-label"
            multilabel_info = f"\n\nClassification Type: This is a {classification_type} classification task."
            multilabel_info += (
                " Multiple labels can be assigned to each text."
                if self.multi_label else
                " Only one label should be assigned to each text."
            )

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
        """Fill a context brainstorming prompt using sampled data."""
        if train_df is None:
            raise ValueError("No data available for context brainstorming")

        formatted_examples = "\n".join(
            f"Text: {row[self.text_column]}\nLabels: {' | '.join(str(row[col]) for col in self.label_columns)}"
            for _, row in train_df.iterrows()
        )

        if custom_prompt and keywords_content:
            prompt_template = custom_prompt.format(keywords=keywords_content)
        else:
            prompt_template = self.warehouse.create_context_prompt

        prompt_text = prompt_template.format(
            examples=formatted_examples,
            labels=", ".join(self.label_columns),
            keywords=keywords_content or ""
        )

        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
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
        """Fill a procedure prompt creator prompt using sampled data."""
        if train_df is None:
            raise ValueError("No data available for procedure prompt creation")

        formatted_examples = "\n".join(
            f"Text: {row[self.text_column]}\nLabels: {' | '.join(str(row[col]) for col in self.label_columns)}"
            for _, row in train_df.iterrows()
        )

        if custom_prompt:
            prompt_template = custom_prompt
            if context_content and '{context}' in custom_prompt:
                prompt_template = prompt_template.format(context=context_content)
        else:
            prompt_template = self.warehouse.procedure_prompt_creator_prompt

        try:
            prompt_text = prompt_template.format(
                data=formatted_examples,
                labels=", ".join(self.label_columns)
            )
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")

        if procedure_additions:
            prompt_text = (
                f"{prompt_text}\n\nAdditional Context/Theory:\n{procedure_additions}\n\n"
                f"Incorporate this additional information into your procedure prompt where appropriate."
            )

        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        return prompt_text

    def fill_train_data_intro_prompt(
        self,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        procedure_content: Optional[str] = None
    ) -> str:
        """Fill a training data introduction prompt using sampled data."""
        if train_df is None:
            raise ValueError("No data available for training data intro")

        formatted_examples = "\n\n".join(
            f"Text {idx + 1}:\n{row[self.text_column]}\n{'|'.join(str(row[col]) for col in self.label_columns)}"
            for idx, (_, row) in enumerate(train_df.iterrows())
        )

        if custom_prompt and procedure_content:
            prompt_template = custom_prompt.format(procedure=procedure_content)
        else:
            prompt_template = self.warehouse.train_data_intro_prompt

        try:
            prompt_text = prompt_template.format(
                examples=formatted_examples,
                labels=", ".join(self.label_columns)
            )
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")

        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        return prompt_text

    def _create_output_format(self, multi_label: bool) -> str:
        """Create binary vector format based on classification type."""
        num_labels = len(self.label_columns)
        if multi_label:
            idx = random.randint(0, num_labels - 1)
            return ' | '.join('1' if i == idx else '0' for i in range(num_labels))
        else:
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
        """Fill the appropriate answer format prompt based on classification type."""
        if row is None:
            raise ValueError("No data available for answer format")

        prompt_template = custom_prompt if custom_prompt else (
            self.warehouse.answer_format_prompt_mult
            if self.multi_label
            else self.warehouse.answer_format_prompt_single
        )

        prompt_text = prompt_template.format(
            paragraph=row[self.text_column],
            output_format=self._create_output_format(self.multi_label),
            labels=' | '.join(self.label_columns)
        )

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
        query_text: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True
    ) -> str:
        """Generate a training data prompt.

        If a NearestNeighbourSampler is injected and query_text is provided,
        selects semantically similar examples. Otherwise falls back to random sampling.

        Args:
            train_df: Full training DataFrame (used for random fallback)
            query_text: The test text to find nearest neighbours for
            custom_role_prompt: Optional custom role prompt
            include_role: Whether to include the role prompt
        """
        if train_df is None or train_df.empty:
            raise ValueError("No data available for training data")

        if isinstance(self.few_shot_mode, int):
            num_examples = self.few_shot_mode
        else:
            num_examples = {
                "zero_shot": 0,
                "one_shot": 1,
                "few_shot": 5,
                "full_coverage": len(train_df)
            }[self.few_shot_mode]

        if num_examples == 0:
            return ""

        # Query sampler if available — fit() was already called in engineer_prompts()
        if self._sampler is not None and query_text is not None:
            sampled_data = self._sampler.sample(query_text=query_text, k=num_examples)
        else:
            sampled_data = train_df.sample(n=min(num_examples, len(train_df)))

        prompt_lines = ["Training Data Examples:"]
        for idx, row in sampled_data.iterrows():
            prompt_lines.append(f"Example {idx + 1}:")
            prompt_lines.append(f"Text: {row[self.text_column]}")
            prompt_lines.append(f"Ratings: {' | '.join(str(row[col]) for col in self.label_columns)}")
            prompt_lines.append("")

        prompt_text = "\n".join(prompt_lines).strip()

        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        return prompt_text

    def build_prompt(self, text: str) -> str:
        """Build a simple classification prompt for a single text."""
        labels_str = ", ".join(self.label_columns)
        if self.multi_label:
            return (
                f"Classify the following text into one or more of these categories: {labels_str}\n\n"
                f"Text: {text}\n\n"
                f"Please respond with the relevant category names separated by commas, "
                f"or \"none\" if no categories apply."
            )
        return (
            f"Classify the following text into one of these categories: {labels_str}\n\n"
            f"Text: {text}\n\n"
            f"Please respond with only the most relevant category name."
        )