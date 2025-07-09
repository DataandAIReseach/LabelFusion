class PromptWarehouse:
    
    role_prompt = "You are a political scientist who developed a theory to measure stereotypical, hegemonic masculinity in presidential speeches."

    theory_background = "Stereotypical and hegemonic masculinity represents a political approach rooted in a patriarchal social structure of beliefs and hierarchical values. The notion of masculinity as “hegemonic” pertains to the hierarchy of masculine traits, where hegemonic masculinity is regarded as the most desirable form. 1 The term “stereotypical” signifies characteristics that are assumed as typical within the patriarchal structure, although these traits might be more exaggerated than realistic. The masculinity discussed in this paper always embodies both hegemonic and stereotypical aspects."

    procedure_prompt = "You will be given a presidential speech. Your task is to analyze the speech and identify the presence of stereotypical and hegemonic masculinity. You will provide a detailed analysis that includes specific examples from the text, explaining how these examples reflect the characteristics of stereotypical and hegemonic masculinity as defined in the theory background."

    train_data_intro_prompt = """
    In the following, paragraphs and the corresponding ratings are given for the features internationality, power_centricity, negativity, abstractivity
    and agent_centricity. They are formated in the following way:

    internationality rating|power_centricity|rating|negativity rating|abstractivity rating|agent_centricity rating

    Take these as examples for learning the value of each feature given an unknown text.

    """

    answer_format_prompt = "Given the above training data, make a prediction for the following paragraph: {paragraph}. Keep in mind the Output format: rating|power_centricity rating|negativity rating|abstractivity rating|agent_centricity. No additional text shall be shown."
