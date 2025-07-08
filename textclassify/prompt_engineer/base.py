from prompt_engineer import role_prompt, theory_background, procedure_prompt, train_data_intro_prompt, answer_format_prompt


class PromptEngineer:

    """
    Base class for prompt engineering.
    This class provides a template for creating prompt engineers.
    """

    def __init__(self, model_name: str):
        self.full_prompt = ""

    def generate_prompt_single_label(self, input_data: str) -> str:
        """
        Generate a prompt based on the input data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def generate_prompt_multiple_labels(self, input_data: str) -> str:
        """
        Generate a prompt for multiple labels based on the input data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def generate_theory_prompt(self) -> None:
        theory_background="""
            Given the following dataset, generate a theory prompt. As an example the following theory prompt for stereotypical and hegemonic masculinity is given:
            {theory_background}
        """
        self.full_prompt += theory_background
    
    def generate_role_prompt(self) -> None:
        role_prompt = """
        Given the following dataset, generate a role prompt. As an example the following role prompt for stereotypical and hegemonic masculinity is given:
        {role_prompt}
        """
        self.full_prompt += role_prompt

    
    
    
