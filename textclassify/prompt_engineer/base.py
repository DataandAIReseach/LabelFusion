import random
from prompt_warehouse import PromptWarehouse
from prompt import Prompt  # Assuming you've implemented the modular Prompt class

class PromptEngineer:
    """
    Builds a full prompt by composing parts from PromptWarehouse using a modular Prompt object.
    Supports configurable few-shot modes with random example selection.
    """

    def __init__(self, model_name: str, few_shot_mode: str = "few_shot"):
        self.model_name = model_name
        self.prompt = Prompt()
        self.examples = []
        self.few_shot_mode = few_shot_mode

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

    def get_full_prompt(self) -> str:
        return self.prompt.render()

    def reset(self):
        self.prompt = Prompt()