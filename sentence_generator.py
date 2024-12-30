language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o'

def generate_sentence_no_context(lang_code):
    '''
    Generate one cognate sentence in the specified language using chosen_llm, without any prior context.
    Returns JSON with the generated sentence and reasoning.
    '''

    num_choices = 6

    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. Your task is to output one (1) {language_codes[lang_code]} sentence. The output must be in JSON format with the following structure:
    
    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of why this sentence uses cognates and is suitable as an independent example>"
    }}
    
    Please keep in mind the following constraints:
    - Try to use cognate words, words that an English speaker can easily identify the meaning of.
    - When possible, use proper nouns that an English speaker would be able to recognize.
    - Ensure the sentence is complete, meaningful, and independent of any external context.
    """

    # Ensure compatibility with the chosen model
    assert SENTENCE_GENERATION_MODEL not in ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct"], "ERROR: GPT-3.5 does not support the completions endpoint"

    # Call the OpenAI API to generate the completion
    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
        ],
        max_tokens=300,
        n=num_choices,
        temperature=1.4,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    # Extract the JSON outputs
    outputs = []
    for choice in response.choices:
        try:
            output = json.loads(choice.message.content)
            outputs.append(output)
        except json.JSONDecodeError:
            print(f"Invalid JSON output: {choice.message.content}")

    return outputs

def generate_next_sentence(lang_code, existing_sentences):
    '''
    Given a list of existing sentences, we want to generate one additional cognate sentence in that language using chosen_llm.
    Returns JSON with the generated sentence and reasoning.
    '''

    num_choices = 6

    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. You are about to receive some text in {language_codes[lang_code]}. Your task is to output one (1) additional {language_codes[lang_code]} sentence that continues where the provided sentence leaves off. The output must be in JSON format with the following structure:
    
    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of why this sentence is a suitable continuation and uses cognates>"
    }}
    
    Please keep in mind the following constraints:
    - Try to use cognate words, words that an English speaker can easily identify the meaning of.
    - When possible, use proper nouns that an English speaker would be able to recognize.
    - Don't include the existing sentence(s) in your response. Your output should consist of one (1) sentence in JSON format.
    """

    # Ensure compatibility with the chosen model
    assert SENTENCE_GENERATION_MODEL not in ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct"], "ERROR: GPT-3.5 does not support the completions endpoint"

    # Format the existing sentences into the user prompt
    user_prompt = "\n".join(existing_sentences)

    # Call the OpenAI API to generate the completion
    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        max_tokens=300,
        n=num_choices,
        temperature=1.4,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    # Extract the JSON outputs
    outputs = []
    for choice in response.choices:
        try:
            output = json.loads(choice.message.content)
            outputs.append(output)
        except json.JSONDecodeError:
            print(f"Invalid JSON output: {choice.message.content}")

    return outputs
