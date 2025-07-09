from prompt_warehouse import PromptWarehouse

# Example
PromptWarehouse.role_prompt
PromptWarehouse.train_data_intro_prompt
PromptWarehouse.answer_format_prompt

from prompt_warehouse import PromptWarehouse
from prompt import Prompt  # Assuming you've implemented the modular Prompt class

class PromptEngineer:
    """
    Builds a full prompt by composing parts from PromptWarehouse using a modular Prompt object.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt = Prompt()

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

    def build_prompt_single_label(self, input_text: str, mode: str = "auto") -> str:
        self.generate_role_prompt()
        self.generate_context_prompt(label_type="single", mode=mode)
        self.generate_feat_def_prompt()
        self.generate_procedure_prompt()
        self.generate_answer_format_prompt()
        self.generate_input_prompt(input_text)
        return self.prompt.render()

    def build_prompt_multiple_labels(self, input_text: str, mode: str = "auto") -> str:
        self.generate_role_prompt()
        self.generate_context_prompt(label_type="multiple", mode=mode)
        self.generate_feat_def_prompt()
        self.generate_procedure_prompt()
        self.generate_answer_format_prompt()
        self.generate_input_prompt(input_text)
        return self.prompt.render()

    def get_full_prompt(self) -> str:
        return self.prompt.render()

    def reset(self):
        self.prompt = Prompt()



    

    
    
    
    
