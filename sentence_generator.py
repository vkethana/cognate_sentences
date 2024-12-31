from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o-mini'
SENTENCE_SCORING_MODEL = 'gpt-4o'
num_choices = 3

def generate_sentence_no_context(lang_code):
    '''
    Generate one cognate sentence in the specified language using chosen_llm, without any prior context.
    Returns JSON with the generated sentence and reasoning.
    '''
    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. Your task is to output one (1) {language_codes[lang_code]} sentence that an English speaker could understand through cognates. The output must be in JSON format with the following structure:

    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of why this sentence uses cognates and is suitable as an independent example>",
        "english_gloss": "<Word-for-word English meanings of cognate words used>"
    }}

    Please follow these guidelines to create highly cognate-rich sentences:
    1. Maximize use of cognate words that share these patterns between {language_codes[lang_code]} and English:
       - Words ending in -tion/-sion (e.g. French 'nation'/'décision')
       - Words ending in -ment (e.g. French 'gouvernement')
       - Academic/technical terms (e.g. 'université', 'télévision')
       - International vocabulary (e.g. 'radio', 'internet')
       - Proper nouns of well-known people, places, or organizations
    
    2. Keep non-cognate words to a minimum and use them only for:
       - Articles (le, la, les)
       - Prepositions (de, à)
       - Basic conjunctions (et, ou)
       - Common verbs when necessary (est, a, va)

    3. Aim for sentences that would score 3 on this rubric:
       0: Totally unintelligible to English speakers
       1: Contains some cognates but meaning unclear
       2: Main idea clear but important details missed
       3: Full meaning clear through cognates despite small connecting words

    Example of an excellent cognate sentence:
    "Le président Emmanuel Macron confirme que la délégation internationale va participer à la conférence importante sur la situation économique européenne."
    (Almost every content word is a recognizable cognate)

    Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt}
        ],
        max_tokens=300,
        n=num_choices,
        temperature=1.2,  # Slightly lower temperature for more focused outputs
        top_p=0.9,
        frequency_penalty=0.2,  # Increased to encourage more diverse vocabulary
        presence_penalty=0.8    # Increased to discourage repetitive patterns
    )

    return [json.loads(choice.message.content) for choice in response.choices 
            if is_valid_json(choice.message.content)]

def generate_next_sentence(lang_code, existing_sentences):
    '''
    Given a list of existing sentences, generate one additional cognate sentence that continues the narrative.
    Returns JSON with the generated sentence and reasoning.
    '''
    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. You will receive some {language_codes[lang_code]} text. Your task is to generate one (1) additional sentence that:
    1. Continues the narrative naturally
    2. Uses many cognate words that English speakers can recognize
    3. Maintains topical and tonal consistency with the previous text

    Output format:
    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of narrative continuity and cognate usage>",
        "english_gloss": "<Word-for-word English meanings of cognate words used>",
        "connection": "<How this sentence follows from the previous content>"
    }}

    Guidelines for cognate-rich continuation:
    1. Reuse cognates from previous sentences when relevant
    2. Introduce new cognates that relate to the established topic
    3. Maintain the same level of formality/style
    4. Use similar sentence structures when appropriate
    5. Connect ideas using cognate transition words where possible
       (e.g. 'finalement', 'généralement', 'naturellement')

    Important: Generate only ONE new sentence. Do not repeat or modify the existing sentences.
    Please do not include Markdown formatting tags (```) in your response.
    """

    # Format existing sentences into the prompt
    narrative_context = "\nContext provided:\n" + "\n".join(existing_sentences)

    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': narrative_context}
        ],
        max_tokens=300,
        n=num_choices,
        temperature=1.3,  # Slightly higher for creative continuation
        top_p=0.9,
        frequency_penalty=0.3,  # Encourage vocabulary variation
        presence_penalty=0.7    # Discourage repetition while maintaining coherence
    )

    return [json.loads(choice.message.content) for choice in response.choices 
            if is_valid_json(choice.message.content)]

def is_valid_json(content):
    """Helper function to validate JSON output"""
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON output: {content}")
        return False

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

def gpt_scored_rubric_individual(sentence):
    '''
    Given a single French sentence, let GPT-4 score it based on a rubric that assigns points between 0 and 3.
    Returns JSON output with the score, reasoning, and a list of cognate words for the sentence.
    '''
    system_prompt = (
        'You are an expert in French to English translation. I will give you one sentence in French, '
        'and I want you to assign one of the following scores to it:\n'
        '0 (lowest score): Totally unintelligible to an English speaker. Example: "Veux-tu déjeuner avec moi?"\n'
        
        '1: Contains some cognate words, but is largely unintelligible to an English speaker. The cognates might allow them '
        'to guess the general topic but not the actual meaning. Example: "Le professeur universitaire présente son document '
        'important à ses étudiants." An English speaker would recognize "professor", "university", "presents", "document", '
        'and "important" but would miss that he is presenting it to his students, making the actual meaning unclear.\n'
        
        '2: Contains many cognate words. An English speaker could understand the main idea but would miss important details '
        'or nuances that change the meaning. Example: "Le patient refuse absolument de prendre ses médicaments malgré les '
        'protestations constantes du docteur." An English speaker would get "patient refuses absolutely to take medications" '
        'and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose '
        'medications and the relationship between the refusal and protestations.\n'
        
        '3 (highest score): An English speaker with zero French knowledge can guess, with ease, the entire meaning of '
        'the sentence. Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va '
        'continuer à défendre le Canada contre la menace américain." The small connecting words (le, que, etc.) can be '
        'ignored without losing meaning.\n'
        '\n'
        'Important scoring notes:\n'
        '- Score 1 sentences have cognates but leave major meaning gaps\n'
        '- Score 2 sentences are mostly understandable but have subtle meaning changes due to missed words\n'
        '- Score 3 should be assigned sparingly - only when missed words don\'t change meaning\n'
        '\n'
        'Please format your response in JSON format as follows:\n'
        '{\n'
        '  "sentence": "<Sentence>",\n'
        '  "reasoning": "<Reasoning for the sentence>",\n'
        '  "cognate_words": [<List of Cognate Words>],\n'
        '  "score": <Score for the Sentence>\n'
        '}\n\n'
        'Here is the sentence:\n'
        f'{sentence}\n'
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
