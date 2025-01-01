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


def generate_difficulty_targeted_sentences(lang_code, target_difficulty, batch_size=10):
    """
    Generate sentences targeting a specific difficulty level (0-3).
    Returns a list of sentences with their scores and metadata.
    """
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English. Generate ONE {language_codes[lang_code]} sentence that would score {target_difficulty} on this cognate difficulty scale:

    0: Completely unintelligible to English speakers. Use common words with no cognates.
    Example: "Je veux manger du pain." (Uses basic vocabulary with no cognates)

    1: Some cognates but meaning unclear. Mix 1-2 cognates with non-cognate essential words.
    Example: "Le professeur intelligent utilise beaucoup de livres." (Has cognates but key verbs/objects aren\'t cognates)

    2: Main idea clear but missing details. Use cognates for main concepts but non-cognates for important details.
    Example: "La décision politique cause des problèmes sérieux." (Core meaning clear but specifics unclear)

    3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

    DIFFICULTY TARGETING STRATEGIES:
    Difficulty 0: Use basic, high-frequency native vocabulary, avoid international words
    Difficulty 1: Use 25-30% cognates in non-crucial positions. Has cognates but leaves major meaning gaps.
    Difficulty 2: Use 50-60% cognates in main concept positions. Sentence is mostly understandable but has subtle meaning changes due to missed words\n
    Difficulty 3: Use 80-90% cognates, especially for key meaning-bearing words. Any small connecting words (le, que, etc.) can be ignored without losing meaning. Should be assigned sparingly - only when missed words don\'t change meaning\n

    Output format:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this should score {target_difficulty}>",
        "cognate_words": [<List of cognates used>]
    }}
    """

    sentences = []
    for _ in range(batch_size):
        # Adjust temperature based on difficulty - higher for extreme scores
        temp = 1.4 if target_difficulty in [0, 3] else 1.0
        
        response = client.chat.completions.create(
            model=SENTENCE_GENERATION_MODEL,
            messages=[{'role': 'system', 'content': system_prompt}],
            temperature=temp,
            max_tokens=200
        )
        
        try:
            generated = json.loads(response.choices[0].message.content)
            # Score the generated sentence
            score = gpt_scored_rubric_individual(generated['sentence'])
            
            sentences.append({
                'sentence': generated['sentence'],
                'target_difficulty': target_difficulty,
                'actual_score': score['score'],
                'cognate_words': generated['cognate_words'],
                'reasoning': generated['reasoning']
            })
        except json.JSONDecodeError:
            print(f"Failed to parse response: {response.choices[0].message.content}")
            continue
            
    return sentences

def build_difficulty_database(lang_code, sentences_per_difficulty=1000):
    """
    Build a database of sentences across all difficulty levels.
    Includes validation and balancing steps.
    """
    database = []
    difficulties = [0, 1, 2, 3]
    
    for difficulty in difficulties:
        print(f"Generating difficulty {difficulty} sentences...")
        sentences = []
        
        while len(sentences) < sentences_per_difficulty:
            batch = generate_difficulty_targeted_sentences(
                lang_code, 
                difficulty,
                batch_size=min(50, sentences_per_difficulty - len(sentences))
            )
            
            # Filter for sentences that matched their target difficulty
            matched = [s for s in batch if s['actual_score'] == s['target_difficulty']]
            sentences.extend(matched)
            
            print(f"Progress: {len(sentences)}/{sentences_per_difficulty} for difficulty {difficulty}")
        
        database.extend(sentences)
        # Print the database out so I can see what's going on
        print(database)
    
    
    return database

def validate_sentence_distribution(database):
    """
    Analyze the distribution of sentences across difficulty levels
    and provide statistics about the dataset.
    """
    stats = {
        'total_sentences': len(database),
        'by_difficulty': {},
        'average_length': {},
        'cognate_counts': {}
    }
    
    for difficulty in [0, 1, 2, 3]:
        difficulty_sentences = [s for s in database if s['actual_score'] == difficulty]
        stats['by_difficulty'][difficulty] = len(difficulty_sentences)
        stats['average_length'][difficulty] = sum(len(s['sentence'].split()) 
                                                for s in difficulty_sentences) / len(difficulty_sentences)
        stats['cognate_counts'][difficulty] = sum(len(s['cognate_words']) 
                                                for s in difficulty_sentences) / len(difficulty_sentences)
    
    return stats

if __name__ == "__main__":
    # Generate a set of sentences for each difficulty level
    lang_code = 'fr'
    sentences_per_difficulty = 10
    database = build_difficulty_database(lang_code, sentences_per_difficulty)
    
    # Validate the distribution and characteristics of the generated sentences
    stats = validate_sentence_distribution(database)
    
    print(f"Total sentences: {stats['total_sentences']}")
    print("Sentences by difficulty:", stats['by_difficulty'])
    print("Average sentence length by difficulty:", stats['average_length'])
    print("Average cognate count by difficulty:", stats['cognate_counts'])
    print("Sample sentence:", database[0])
