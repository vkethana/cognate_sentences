from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o-mini'
SENTENCE_SCORING_MODEL = 'o1-preview'
num_choices = 3

def generate_sentence_no_context(lang_code):
    '''
    Generate one cognate sentence with high variance - aiming for more perfect (3) scores
    while accepting occasional low scores.
    '''
    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. Your task is to output one (1) {language_codes[lang_code]} sentence that an English speaker could understand completely through cognates. The output must be in JSON format.

    MOST IMPORTANT: Your primary goal is to create sentences that an English speaker can understand ENTIRELY through cognates. Don't worry about sounding natural in {language_codes[lang_code]} - it's better to sound a bit artificial and be completely understandable than to sound natural but use non-cognate words.

    Output format:
    {{
        "sentence": "<The generated sentence>",
        "reasoning": "<Explanation of cognate choices>",
        "english_gloss": "<Word-for-word English meanings>"
    }}

    STRATEGY FOR CREATING PERFECT COGNATE SENTENCES:
    1. Start with an English sentence using mainly Latin/Greek-derived words
    2. Convert it nearly word-for-word to {language_codes[lang_code]}
    3. Use only these types of words:
       - Direct cognates (-tion, -ment, -able endings)
       - International terms (télévision, internet, radio)
       - Academic/technical vocabulary (université, médical)
       - Famous proper nouns
       - Minimal connecting words (le, la, de)

    EXAMPLE PERFECT (SCORE 3) SENTENCES:
    "Le professeur présente sa publication scientifique à la conférence internationale de médecine."
    "La situation politique américaine cause une grande frustration pour la population européenne."
    "Le président confirme que son administration va continuer les opérations militaires."

    Remember: Don't compromise understandability for naturalness. An awkward but perfectly cognate sentence is better than a natural one with even one crucial non-cognate word.

    Please do not include Markdown formatting tags (```) in your response.
    """

    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt}
        ],
        max_tokens=300,
        n=num_choices,
        temperature=1.8,  # Much higher temperature for more variance
        top_p=0.95,      # Allow more diverse token selection
        frequency_penalty=0.3,
        presence_penalty=1.0    # Maximum presence penalty to force diverse patterns
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

def gpt_scored_rubric_individual(sentence):
    '''
    Given a single French sentence, let GPT-4 score it based on a rubric that assigns points between 0 and 3.
    Returns JSON output with the score, reasoning, and a list of cognate words for the sentence.
    '''
    system_prompt = f"""
    You are an expert in French to English translation. I will give you one sentence in French, and I want you to assign one of the following scores to it:

    0: Completely unintelligible to English speakers.
    Example: "Je veux manger du pain."

    1: Contains some cognate words, but contains words unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the main idea or actual meaning.
    Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

    2: Contains many cognate words. An English speaker might guess the main idea but would miss important details or nuances that change the meaning.
    Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
    An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

    3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

    Important scoring notes:
    - Score 0 sentences have little to no cognates
    - Score 1 sentences have cognates but leave major meaning gaps
    - Score 2 sentences are mostly understandable but have subtle meaning changes due to missed words
    - Score 3 should be assigned sparingly - only when missed words don’t change meaning

    Please format your response in JSON format as follows:
    {{
      "sentence": "<Sentence>",
      "cognate_words": [<List of Cognate Words>],
      "score": <Score for the Sentence>,
      "reasoning": "<Reasoning for the score>"
    }}

    Here is the sentence:
    {sentence}

    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

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
