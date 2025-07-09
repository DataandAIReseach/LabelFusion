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


    def generate_role_prompt_single(self) -> None:
        """
        Generate a role prompt based on the role of the model.
        This method should be overridden by subclasses.
        """
        self.full_prompt += role_prompt

    def generate_role_prompt_multiple(self) -> None:
        """
        Generate a role prompt for multiple labels.
        This method should be overridden by subclasses.
        """
        self.full_prompt += role_prompt
    
    def generate_context_prompt_single(self) -> None:
        """
        Generate a context prompt for single label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt
    
    def generate_context_prompt_single_auto(self) -> None:
        """
        Generate a context prompt for single label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt

    def generate_context_prompt_single_man(self) -> None:
        """
        Generate a context prompt for single label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt

    def generate_context_prompt_multiple(self) -> None:
        """
        Generate a context prompt for multiple label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt

    def generate_context_prompt_multiple_auto(self) -> None:
        """
        Generate a context prompt for multiple label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt
    
    def generate_context_prompt_multiple_man(self) -> None:
        """
        Generate a context prompt for multiple label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt
    
    def generate_feat_def_prompt(self) -> None:
        """
        Generate a feature definition prompt based on the features of the model.
        This method should be overridden by subclasses.
        """
        self.full_prompt += "The features are defined as follows: \n" + answer_format_prompt
    
    def generate_feat_def_prompt_auto(self) -> None:
        """
        Generate a feature definition prompt for automatic classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += "The features are defined as follows: \n" + answer_format_prompt
    
    def generate_feat_def_prompt_man(self):
        """
        Generate a feature definition prompt for manual classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += "The features are defined as follows: \n" + answer_format_prompt
    
    def generate_procedure_prompt(self) -> None:
        """
        Generate a procedure prompt for single label classification.
        This method should be overridden by subclasses.
        """
        self.full_prompt += procedure_prompt
    
    def generate_train_data_intro_prompt(self) -> None:
        """
        Generate a training data introduction prompt.
        This method should be overridden by subclasses.
        """
        self.full_prompt += train_data_intro_prompt
    
    def generate_mode_prompt(self) -> None:
        """
        Generate a mode prompt based on the mode of the model.
        This method should be overridden by subclasses.
        """
        self.full_prompt += "The model is in single label classification mode."
    
    def answer_format_prompt(self) -> None:
        """
        Generate an answer format prompt based on the answer format of the model.
        This method should be overridden by subclasses.
        """
        self.full_prompt += answer_format_prompt
    

    def generate_procedure_prompt(self) -> None:
        """
        Generate a procedure prompt based on the procedure of the model.
        This method should be overridden by subclasses.
        """
        self.full_prompt += procedure_prompt


    

    
    
    
    
