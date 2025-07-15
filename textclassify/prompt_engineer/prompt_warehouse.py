class PromptWarehouse:
    
    # role_prompt = "You are a political scientist who developed a theory to measure stereotypical, hegemonic masculinity in presidential speeches."

    # theory_background = "Stereotypical and hegemonic masculinity represents a political approach rooted in a patriarchal social structure of beliefs and hierarchical values. The notion of masculinity as “hegemonic” pertains to the hierarchy of masculine traits, where hegemonic masculinity is regarded as the most desirable form. 1 The term “stereotypical” signifies characteristics that are assumed as typical within the patriarchal structure, although these traits might be more exaggerated than realistic. The masculinity discussed in this paper always embodies both hegemonic and stereotypical aspects."

    # procedure_prompt = "You will be given a presidential speech. Your task is to analyze the speech and identify the presence of stereotypical and hegemonic masculinity. You will provide a detailed analysis that includes specific examples from the text, explaining how these examples reflect the characteristics of stereotypical and hegemonic masculinity as defined in the theory background."

    # train_data_intro_prompt = """
    # In the following, paragraphs and the corresponding ratings are given for the features internationality, power_centricity, negativity, abstractivity
    # and agent_centricity. They are formated in the following way:

    # internationality rating|power_centricity|rating|negativity rating|abstractivity rating|agent_centricity rating

    # Take these as examples for learning the value of each feature given an unknown text.

    context_brainstorm_prompt = f"""
    Given the dataset {{data}}, which includes text samples and ratings for features 
    ({", ".join(features)}), analyze the content and associated ratings to infer the general topic 
    the dataset likely represents. Based on this analysis, generate a list of keywords or concepts 
    that are representative of the topic.
    """

    context_brainstorm_role_prompt = f"""
    You are a brainstormer. You are part of a package for classyfying texts with regard to the features {', '.join(data.columns)}.
    """

    create_context_prompt = f"""
    You are tasked with creating a context prompt that will help in understanding the dataset provided. For that sake take the keywords {', '.join(keywords)} as an input and write a context prompt of 3 to 4 sentences for the underlying dataset.
    """

    procedure_prompt_creator_prompt = """
        You are about to write a procedure prompt designed to guide the analysis of a text using a specific theoretical or conceptual lens.

        A well-structured procedure prompt must include the following main components:

        1. Input Specification – Clearly state what kind of text or data the user or model will receive.
        2. Primary Task Instruction – Define what the user is expected to do with the input.
        3. Theoretical or Analytical Focus – Indicate the specific lens, concept, or framework through which the input should be analyzed.
        4. Depth of Analysis Required – Specify the level of detail, reasoning, or argumentation expected.
        5. Use of Evidence – Instruct the user to ground their analysis in specific examples from the input text.
        6. Connection to Theory or Definitions – Request that findings be linked back to theoretical definitions, frameworks, or scholarly concepts.

        Find below the dataset so that you can assume the concept to be classified.

        {data}

        Now create the procedure prompt and confine yourself to two to three sentences
        
        """

    train_data_prompt = f"""In the following, paragraphs and the corresponding ratings are given for the features {', '.join(data.columns)}.\nThey are formatted in the following way:\n\n    {" rating|".join(data.columns)} rating\n\nTake these as examples for learning the value of each feature given an unknown text.""".strip()


    answer_format_prompt_single = f"""
        Given the above training data, make a prediction for the following paragraph.

        The output format shall be:

            {' | '.join(data['label'].unique())}

        Each value must be either 1 (feature present) or 0 (feature absent).
        Only one value may be 1, as this is a multi-class classification task — 
        meaning the paragraph can belong to only one class (mutually exclusive).

        No additional text shall be shown — output only the classification line.

        Example:
            If the task is to classify sentiment in Amazon product reviews with the following categories:
                very negative
                slightly negative
                neutral
                slightly positive
                very positive

            Then, for a very negative review, the prediction would be:
                1 | 0 | 0 | 0 | 0
        """.strip()
    
    answer_format_prompt_mult = f"""
        Given the above training data, make a prediction for the following paragraph.

        The output format shall be:

            {' | '.join(data['label'].unique())}

        Each value must be either 1 (label present) or 0 (absent).  
        Multiple values may be 1, as this is a multi-label classification task — 
        a paragraph can belong to more than one class.

        No additional text shall be shown — output only the classification line.

        Example:
            If the task is to classify toxic online comments with the following categories:
                toxic
                insult
                threat
                obscene
                identity hate

            Then, for a comment that is both toxic and obscene, the prediction would be:
                1 | 0 | 0 | 1 | 0
        """.strip()
    
    #answer_format_prompt = "Given the above training data, make a prediction for the following paragraph: {paragraph}. Keep in mind the Output format: rating|power_centricity rating|negativity rating|abstractivity rating|agent_centricity. No additional text shall be shown."

    role_prompt_creator_prompt = """
    You are given a series of examples, each consisting of a piece of text and its associated label.
    From these examples, infer the kind of role or persona that would most appropriately be analyzing,
    producing, or categorizing this type of content.

    Your task is to describe the role in one or two sentences only. This could include the speaker’s goals,
    responsibilities, or context — such as a politician addressing voters, a scientist analyzing results,
    a customer writing a review, etc.

    Here are the examples:
    {data}

    Based on the patterns above, describe the role you are taking in this context in one or two sentences:
    """





