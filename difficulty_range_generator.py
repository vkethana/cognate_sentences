import datetime
import os
from openai import OpenAI
import json
import random
import math

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'o1-preview'
SENTENCE_SCORING_MODEL = 'o1-preview'
NUM_SENTENCES_GENERATED_PER_LLM_CALL = 5
NUM_SENTENCES_PER_STORY = 11

def gpt_scored_rubric_batch(sentences):
    '''
    Score multiple French sentences at once using GPT-4.

    Args:
        sentences: List of sentences to score
    Returns:
        List of scoring results
    '''

    system_prompt = f"""
    You are an expert in French to English translation. I will give you {NUM_SENTENCES_GENERATED_PER_LLM_CALL}sentences in French, and I want you to score each of them on a scale from 0-3 using the following rubric:

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

    For each sentence, provide a JSON object with these fields:
    {{
      "sentence": "<Sentence>",
      "cognate_words": [<List of Cognate Words>],
      "reasoning": "<Reasoning for the score>",
      "score": <Numerical for the Sentence (0-3)>
    }}

    Please format your response as a JSON array of these objects. You should have {NUM_SENTENCES_GENERATED_PER_LLM_CALL} objects in your array.

    Here are the sentences to score:
    {json.dumps(sentences, ensure_ascii=False)}
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'user', 'content': system_prompt}
        ],
        temperature=1
    )
    
    response_text = completion.choices[0].message.content.strip()
    try:
        results = json.loads(response_text)
        return results
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        raise

def generate_difficulty_targeted_sentences_batch(lang_code, target_difficulty, output_file, batch_size):
    """
    Generate multiple sentences at once targeting a specific difficulty level.
    """
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English. Generate {batch_size} {language_codes[lang_code]} sentences aiming for difficulty level {target_difficulty} on this cognate difficulty scale:

    0: Completely unintelligible to English speakers.
    Example: "Je veux manger du pain."

    1: Contains some cognate words, but is largely unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the actual meaning.
    Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

    2: Contains many cognate words. An English speaker could understand the main idea but would miss important details or nuances that change the meaning.
    Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
    An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

    3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

    DIFFICULTY TARGETING STRATEGIES:
    Difficulty 0: Use basic, high-frequency native vocabulary, avoid international words
    Difficulty 1: Use 25-30% cognates in non-crucial positions. Has cognates but leaves major meaning gaps.
    Difficulty 2: Use 50-60% cognates in main concept positions. Sentence is mostly understandable but has subtle meaning changes due to missed words\n
    Difficulty 3: Use 80-90% cognates, especially for key meaning-bearing words. Any small connecting words (le, que, etc.) can be ignored without losing meaning. Should be assigned sparingly - only when missed words don\'t change meaning\n

    Output format:
    Return a JSON array where each object has:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {str(target_difficulty)},
        "reasoning": "<Why this should score {target_difficulty}>",
        "cognate_words": [<List of cognates used>]
    }}

    Generate {batch_size} different sentences meeting these criteria.
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    try:
        # Generate batch of sentences
        response = client.chat.completions.create(
            model=SENTENCE_GENERATION_MODEL,
            messages=[{'role': 'user', 'content': system_prompt}],
            temperature=1.0
            #max_tokens=600
        )
        print("Got back this from the LLM")
        print(response.choices[0].message.content)
        
        generated_batch = json.loads(response.choices[0].message.content)
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract just the sentences for batch scoring
        sentences_to_score = [item['sentence'] for item in generated_batch]
        
        # Score all sentences in one batch
        score_results = gpt_scored_rubric_batch(sentences_to_score)
        
        # Combine generation and scoring results
        batch_sentences = []
        for gen, score in zip(generated_batch, score_results):
            sentence_data = {
                'sentence': gen['sentence'],
                'target_difficulty': target_difficulty,
                'proposed_cognate_words': gen['cognate_words'],
                'generation_reasoning': gen['reasoning'],
                'actual_score': score['score'],
                'actual_score_reasoning': score['reasoning'],
                'actual_cognate_words': score['cognate_words'],
                'generation_timestamp': timestamp
            }
            batch_sentences.append(sentence_data)
        
        save_sentences_batch(batch_sentences, output_file)
        
        for i, sentence in enumerate(batch_sentences):
            print(f"Generated sentence {i+1}/{len(batch_sentences)} - Target: {target_difficulty}, Actual: {sentence['actual_score']}")
            
        return batch_sentences
        
    except Exception as e:
        print(f"Error generating sentences: {e}")
        # Make traceback detailed
        import traceback
        traceback.print_exc()

        return []


def is_valid_json(content):
    """Helper function to validate JSON output"""
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON output: {content}")
        return False

def save_sentences_batch(sentences, output_file):
    """
    Append a batch of sentences to a JSON file, creating the file if it doesn't exist.
    """
    try:
        # Read existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                'sentences': [],
                'metadata': {
                    'total_count': 0,
                    'difficulty_counts': {'0': 0, '1': 0, '2': 0, '3': 0},  # Use strings as keys
                    'last_updated': None
                }
            }
        
        # Add new sentences
        data['sentences'].extend(sentences)
        
        # Update metadata
        data['metadata']['total_count'] = len(data['sentences'])
        data['metadata']['last_updated'] = datetime.datetime.now().isoformat()
        
        # Update difficulty counts - convert score to string for JSON compatibility
        for sentence in sentences:
            score = str(sentence['actual_score'])  # Convert to string
            data['metadata']['difficulty_counts'][score] += 1
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error saving to {output_file}: {str(e)}")  # Print full error message
        # Save to backup file if main save fails
        backup_file = output_file + '.backup'
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)


def generate_story_batch(lang_code, story_length, seed_sentence=None):
    """
    Generate a story in batches of NUM_SENTENCES_GENERATED_PER_LLM_CALL sentences at a time.
    Each batch continues from the last sentence of the previous batch.
    
    Args:
        lang_code (str): Language code ('fr' for French)
        story_length (int): Target number of sentences in the story
        seed_sentence (dict, optional): Initial sentence to start the story
        
    Returns:
        tuple: (story_data dictionary, output_filename)
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'batch_stories/{lang_code}_batch_story_{timestamp}.json'
    
    # Initialize story data structure
    story_data = {
        'story': [],
        'metadata': {
            'language': lang_code,
            'target_difficulty': None,
            'actual_difficulty_mean': None,
            'creation_date': datetime.datetime.now().isoformat(),
            'sentence_count': 0
        }
    }
    
    # If no seed sentence provided, generate one with random difficulty
    if not seed_sentence:
        target_difficulty = random.randint(0, 3)
        story_data['metadata']['target_difficulty'] = target_difficulty
        
        print(f"Generating seed sentence with difficulty: {target_difficulty}")
        initial_batch = generate_difficulty_targeted_sentences_batch(
            lang_code, 
            target_difficulty, 
            output_file,
            2
        )
        if initial_batch:
            initial_batch = sorted(initial_batch, key=lambda x: abs(x['actual_score'] - target_difficulty))
            seed_sentence = initial_batch[0]
    
    if seed_sentence:
        print("Seed sentence found!", seed_sentence)
        story_data['story'].append(seed_sentence)
        target_difficulty = seed_sentence['target_difficulty']
        story_data['metadata']['target_difficulty'] = target_difficulty
        
        # Calculate how many full batches we need
        remaining_sentences = story_length - 1  # -1 for seed sentence
        num_full_batches = math.ceil(remaining_sentences / NUM_SENTENCES_GENERATED_PER_LLM_CALL)
        print("Let's generate the story in", num_full_batches, "batches")

        try:
            # Generate full batches
            for batch_num in range(num_full_batches):
                print(f"Generating batch {batch_num + 1}/{num_full_batches}")
                
                # Get the latest sentence in the story
                latest_sentence = story_data['story'][-1]['sentence']
                
                # Generate batch of continuation sentences
                system_prompt = f"""
                You are a fluent speaker of both {language_codes[lang_code]} and English.
                Generate exactly {NUM_SENTENCES_GENERATED_PER_LLM_CALL} {language_codes[lang_code]} sentences that:
                1. Continue this story that started with: {latest_sentence}
                2. Form a coherent narrative where each sentence follows from the previous one
                3. Target difficulty level {target_difficulty} using these criteria:

                    Level 0: Completely unintelligible to English speakers.
                    Example: "Je veux manger du pain."

                    Level 1: Contains some cognate words, but is largely unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the actual meaning.
                    Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

                    Level 2: Contains many cognate words. An English speaker could understand the main idea but would miss important details or nuances that change the meaning.
                    Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
                    An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

                    Level 3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
                    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

                    DIFFICULTY TARGETING STRATEGIES:
                    Difficulty 0: Use basic, high-frequency native vocabulary, avoid international words
                    Difficulty 1: Use 25-30% cognates in non-crucial positions. Has cognates but leaves major meaning gaps.
                    Difficulty 2: Use 50-60% cognates in main concept positions. Sentence is mostly understandable but has subtle meaning changes due to missed words\n
                    Difficulty 3: Use 80-90% cognates, especially for key meaning-bearing words. Any small connecting words (le, que, etc.) can be ignored without losing meaning. Should be assigned sparingly - only when missed words don\'t change meaning\n

                Format your response as a JSON array of {NUM_SENTENCES_GENERATED_PER_LLM_CALL} objects:
                {{
                    "sentence": "<Generated sentence>",
                    "target_difficulty": {target_difficulty},
                    "reasoning": "<Why this continues the story from the previous sentence in this JSON array AND why this sentence matches difficulty>",
                    "cognate_words": [<List of cognates used>]
                }}

                Important: Each sentence must directly follow from the previous one to form a coherent story.
                Generate {NUM_SENTENCES_GENERATED_PER_LLM_CALL} sentences meeting these criteria (difficulty level and story continuation).
                Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
                """
                
                response = client.chat.completions.create(
                    model=SENTENCE_GENERATION_MODEL,
                    messages=[{'role': 'user', 'content': system_prompt}],
                    temperature=1.0
                )
                
                # Parse generated sentences
                generated_batch = json.loads(response.choices[0].message.content)
                
                # Score the batch
                sentences_to_score = [item['sentence'] for item in generated_batch]
                score_results = gpt_scored_rubric_batch(sentences_to_score)
                
                # Combine generation and scoring data
                timestamp = datetime.datetime.now().isoformat()
                for gen, score in zip(generated_batch, score_results):
                    sentence_data = {
                        'sentence': gen['sentence'],
                        'target_difficulty': target_difficulty,
                        'proposed_cognate_words': gen['cognate_words'],
                        'generation_reasoning': gen['reasoning'],
                        'actual_score': score['score'],
                        'actual_score_reasoning': score['reasoning'],
                        'actual_cognate_words': score['cognate_words'],
                        'generation_timestamp': timestamp
                    }
                    story_data['story'].append(sentence_data)
                
                # Update metadata and save after each batch
                story_data['metadata']['sentence_count'] = len(story_data['story'])
                story_data['metadata']['actual_difficulty_mean'] = sum(
                    s['actual_score'] for s in story_data['story']
                ) / len(story_data['story'])
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(story_data, f, ensure_ascii=False, indent=2)
            
                
        except Exception as e:
            print(f"Error generating story: {e}")
            import traceback
            traceback.print_exc()
        
        return story_data, output_file
    
    return None, None

# New main function for story generation
if __name__ == "__main__":
    for _ in range(40):
        print("Generating new story batch...")
        story_data, output_file = generate_story_batch(
            lang_code='fr',
            story_length=NUM_SENTENCES_PER_STORY  # Will generate in batches of NUM_SENTENCES_GENERATED_PER_LLM_CALL
        )
        print(f"\nStory batch generated and saved to: {output_file}")
        print("Story metadata:", story_data['metadata'])
