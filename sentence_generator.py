from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o-mini'
SENTENCE_SCORING_MODEL = 'gpt-4o-mini'
num_choices = 3

def generate_sentence_no_context(lang_code):
    '''
    Generate one cognate sentence in the specified language using chosen_llm, without any prior context.
    Returns JSON with the generated sentence and reasoning.
    '''

    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. Your task is to output one (1) {language_codes[lang_code]} sentence. The output must be in JSON format with the following structure:

    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of why this sentence uses cognates and is suitable as an independent example>"
    }}

    Please keep in mind the following constraints:
    - Try to use cognate words, words that an English speaker can easily identify the meaning of.
    - When possible, use proper nouns that an English speaker would be able to recognize.
    - Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
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

    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. You are about to receive some text in {language_codes[lang_code]}. Your task is to output one (1) additional {language_codes[lang_code]} sentence that continues where the provided sentence leaves off. The output must be in JSON format with the following structure:
    
    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of why this sentence is a suitable continuation and uses cognates>"
    }}
    
    Please keep in mind the following constraints:
    - Try to use cognate words, words that an English speaker can easily identify the meaning of.
    - When possible, use proper nouns that an English speaker would be able to recognize.
    - Don't include the existing sentence(s) in your response. Your output should consist of one (1) sentence in JSON format. Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
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

def gpt_scored_rubric_batch(sentences):
    '''
    Given a list of three arbitrary French sentences, let GPT-4 score them based on a rubric that assigns points between 0 and 3.
    Returns JSON output with scores, reasoning, and a list of cognate words for each sentence.
    '''
    if len(sentences) != 3:
        raise ValueError("This function requires exactly three sentences.")

    system_prompt = (
        'You are an expert in French to English translation. I will give you three sentences in French, and I want you to assign one of the following scores to each of them:\n'

        '0 (lowest score): Totally unintelligible to an English speaker.\n'

        '1: Contains some cognate words, but is largely unintelligible to an English speaker.\n'

        '2: Contains many cognate words. An English speaker could partially understand the sentence but they would probably miss a few important words or phrases.\n'

        '3 (highest score): An English speaker with zero French knowledge can guess, with ease, the entire meaning of the sentence. Try to assign this score sparingly. Do not assign this score to a sentence if some nuances or details may be lost on the reader.\n'

        '\n'
        'As an example, consider the following set of sentences:\n'
        '1. “Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain.”\n'
        'Reasoning: An English speaker can make out the sentence through the cognate words: "President Emmanuel Macron assures the Canadian people that the French government is giong to continue to defend Canada against the American menace." There are noncognate words (e.g. "que", "le", "la"), but they are small enough that they can be ignored without missing the meaning of the sentence.\n'
        'Cognate Words: président, Emmanuel Macron, assure, peuple, canadien, gouvernement, français, défendre, Canada, contre, menace, américain\n'
        'Final Score: 3\n'
        '2. "Veux-tu déjeuner avec moi?"\n'
        'Reasoning: The sentence does not contain a single cognate and is totally unintelligible to a monolingual English speaker.\n'
        'Cognate Words: \n'
        'Final Score: 0\n'
        '3. "Lors du repas, la famille royale a été accompagnée de musiciens venus de différentes régions pour divertir."\n'
        'Reasoning: The sentence contains several cognate words and an English speaker can understand the general idea, although some details might be missed.\n'
        'Cognate Words: famille, royale, musiciens, différentes, régions\n'
        'Final Score: 2\n\n'
        'Please format your responses in JSON format as follows:\n'
        '[\n'
        '  {"sentence": "<Sentence 1>", "reasoning": "<Reasoning for Sentence 1>", "cognate_words": [<List of Cognate Words>], "score": <Score for Sentence 1>},\n'
        '  {"sentence": "<Sentence 2>", "reasoning": "<Reasoning for Sentence 2>", "cognate_words": [<List of Cognate Words>], "score": <Score for Sentence 2>},\n'
        '  {"sentence": "<Sentence 3>", "reasoning": "<Reasoning for Sentence 3>", "cognate_words": [<List of Cognate Words>], "score": <Score for Sentence 3>}\n'
        ']\n'
        '\n'
        'Here are the three sentences:\n'
        f'1. {sentences[0]}\n'
        f'2. {sentences[1]}\n'
        f'3. {sentences[2]}\n'
        'Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.'
    )

    print(f"ASKING {SENTENCE_SCORING_MODEL} the following prompt: {system_prompt}")

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'user', 'content': system_prompt}
        ],
        temperature=1
    )

    # Extract and parse the JSON response
    response_text = completion.choices[0].message.content.strip()
    print("Got a response from chatgpt!", response_text)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        raise

    return result

