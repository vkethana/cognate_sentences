import datetime
import os
from openai import OpenAI
import json
import random

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o-mini'
SENTENCE_SCORING_MODEL = 'o1-preview'
LLM_CALL_BATCH_SIZE = 5
SENTENCES_PER_DIFF_LEVEL = 10

def gpt_scored_rubric_batch(sentences):
    '''
    Score multiple French sentences at once using GPT-4.
    
    Args:
        sentences: List of sentences to score
    Returns:
        List of scoring results
    '''

    system_prompt = f"""
    You are an expert in French to English translation. I will give you {LLM_CALL_BATCH_SIZE}sentences in French, and I want you to score each of them on a scale from 0-3 using the following rubric:

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

    Please format your response as a JSON array of these objects. You should have {LLM_CALL_BATCH_SIZE} objects in your array.

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

def generate_difficulty_targeted_sentences_batch(lang_code, target_difficulty, output_file):
    """
    Generate multiple sentences at once targeting a specific difficulty level.
    """
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English. Generate {LLM_CALL_BATCH_SIZE} {language_codes[lang_code]} sentences aiming for difficulty level {target_difficulty} on this cognate difficulty scale:

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

    Generate {LLM_CALL_BATCH_SIZE} different sentences meeting these criteria.
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    try:
        # Generate batch of sentences
        response = client.chat.completions.create(
            model=SENTENCE_GENERATION_MODEL,
            messages=[{'role': 'system', 'content': system_prompt}],
            temperature=1.0,
            max_tokens=600
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

def build_adaptive_database(lang_code, target_counts={0: SENTENCES_PER_DIFF_LEVEL, 1: SENTENCES_PER_DIFF_LEVEL, 2: SENTENCES_PER_DIFF_LEVEL, 3: SENTENCES_PER_DIFF_LEVEL}, output_file='sentence_database.json'):
    """
    Build database of sentences, continuing until we have enough at each difficulty level.
    Saves all generated sentences, even if they don't match their target difficulty.
    """
    # Convert target_counts keys to strings for comparison with JSON data
    target_counts = {str(k): v for k, v in target_counts.items()}
    current_counts = {'0': 0, '1': 0, '2': 0, '3': 0}
    
    while True:
        # Find difficulty level that needs more sentences
        needed_difficulties = [int(d) for d, count in current_counts.items() 
                             if count < target_counts[d]]
        
        if not needed_difficulties:
            break
            
        # Generate batch targeting the most-needed difficulty
        target_diff = needed_difficulties[0]
        print("Currently targeting difficulty level:", target_diff)
        
        generate_difficulty_targeted_sentences_batch(
            lang_code,
            target_diff,
            output_file
        )
        
        # Update counts from file
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_counts = data['metadata']['difficulty_counts']
            
            print("\nCurrent progress:")
            for diff, count in current_counts.items():
                target = target_counts[diff]
                print(f"Difficulty {diff}: {count}/{target} ({count/target*100:.1f}%)")
            print("\n")
        except Exception as e:
            print(f"Error updating counts: {str(e)}")
            break

def analyze_sentence_distribution(json_file):
    """
    Analyze the distribution of sentences in an existing JSON file.
    Returns statistics about the dataset.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = data['sentences']
        stats = {
            'total_sentences': len(sentences),
            'by_target_difficulty': {'0': 0, '1': 0, '2': 0, '3': 0},
            'by_actual_difficulty': {'0': 0, '1': 0, '2': 0, '3': 0},
            'match_rate': {},
            'average_length': {},
            'cognate_counts': {}
        }
        
        # Calculate statistics
        for sentence in sentences:
            target = str(sentence['target_difficulty'])
            actual = str(sentence['actual_score'])
            
            stats['by_target_difficulty'][target] += 1
            stats['by_actual_difficulty'][actual] += 1
            
            # Track sentence lengths and cognate counts by actual difficulty
            if 'sentence' in sentence:
                length = len(sentence['sentence'].split())
                if actual not in stats['average_length']:
                    stats['average_length'][actual] = []
                stats['average_length'][actual].append(length)
            
            if 'cognate_words' in sentence:
                if actual not in stats['cognate_counts']:
                    stats['cognate_counts'][actual] = []
                stats['cognate_counts'][actual].append(len(sentence['cognate_words']))
        
        # Calculate match rates
        for diff in ['0', '1', '2', '3']:
            target_count = stats['by_target_difficulty'][diff]
            if target_count > 0:
                matches = sum(1 for s in sentences 
                            if str(s['target_difficulty']) == diff 
                            and str(s['actual_score']) == diff)
                stats['match_rate'][diff] = matches / target_count
        
        # Calculate averages
        for diff in stats['average_length']:
            if stats['average_length'][diff]:
                stats['average_length'][diff] = sum(stats['average_length'][diff]) / len(stats['average_length'][diff])
            if stats['cognate_counts'][diff]:
                stats['cognate_counts'][diff] = sum(stats['cognate_counts'][diff]) / len(stats['cognate_counts'][diff])
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        raise

def start_new_database(lang_code='fr', sentences_per_difficulty=SENTENCES_PER_DIFF_LEVEL):
    """
    Start a fresh sentence database generation.
    Creates a timestamped file and generates sentences until targets are reached.
    """
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{lang_code}_sentences_{timestamp}.json'
    
    # Initialize empty file with metadata
    initial_data = {
        'sentences': [],
        'metadata': {
            'total_count': 0,
            'difficulty_counts': {'0': 0, '1': 0, '2': 0, '3': 0},
            'language': lang_code,
            'creation_date': datetime.datetime.now().isoformat(),
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    print(f"Starting new sentence database in: {output_file}")
    
    # Use the existing build_adaptive_database function
    target_counts = {str(i): sentences_per_difficulty for i in range(4)}
    build_adaptive_database(
        lang_code=lang_code,
        target_counts=target_counts,
        output_file=output_file
    )

def resume_adaptive_database(input_file, target_counts={0: SENTENCES_PER_DIFF_LEVEL, 1: SENTENCES_PER_DIFF_LEVEL, 2: SENTENCES_PER_DIFF_LEVEL, 3: SENTENCES_PER_DIFF_LEVEL},
                           output_file=None, lang_code='fr'):
    """
    Resume sentence generation from an existing JSON file.
    
    Args:
        input_file (str): Path to existing JSON file to resume from
        target_counts (dict): Target number of sentences for each difficulty level
        output_file (str, optional): New output file path. If None, will modify input file
        lang_code (str): Language code for generation
    """
    # Convert target_counts keys to strings for JSON compatibility
    target_counts = {str(k): v for k, v in target_counts.items()}
    
    # If no output file specified, use input file
    if output_file is None:
        output_file = input_file
    elif input_file != output_file:
        # If using new output file, copy existing data
        with open(input_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    try:
        # Load existing progress
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            current_counts = data['metadata']['difficulty_counts']
        
        print("\nResuming from existing file:")
        print(f"Total sentences so far: {len(data['sentences'])}")
        for diff, count in current_counts.items():
            target = target_counts[diff]
            print(f"Difficulty {diff}: {count}/{target} ({count/target*100:.1f}%)")
        print("\n")
        
        # Continue generation
        while True:
            # Find difficulty levels that need more sentences
            print("Current counts:", current_counts)
            needed_difficulties = [(int(d), count) for d, count in current_counts.items() 
                                 if count < target_counts[d]]
            needed_difficulties = sorted(needed_difficulties, key=lambda x: x[1])
            needed_difficulties = [d for d, _ in needed_difficulties]

            # make needed_difficulties sorted in ascending order based on the count of sentneces
            # in other words, the first element should be the difficulty with the leastn umber of sentences associated with it

            if not needed_difficulties:
                print("All targets reached! Generation complete.")
                break
            
            # Generate batch targeting the most-needed difficulty
            target_diff = needed_difficulties[0]
            print("Currently targeting difficulty level:", target_diff)
            
            generate_difficulty_targeted_sentences_batch(
                lang_code,
                target_diff,
                output_file
            )
            
            # Update counts from file
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_counts = data['metadata']['difficulty_counts']
            
            print("\nCurrent progress:")
            for diff, count in current_counts.items():
                target = target_counts[diff]
                print(f"Difficulty {diff}: {count}/{target} ({count/target*100:.1f}%)")
            print("\n")
            
    except Exception as e:
        print(f"Error resuming generation: {str(e)}")
        raise

def generate_story_continuation(lang_code, target_difficulty, previous_sentences, output_file):
    """
    Generate a batch of sentences that continue a story, targeting a specific difficulty level.
    Returns the sentence closest to target difficulty.
    """
    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. 
    Generate {LLM_CALL_BATCH_SIZE} different {language_codes[lang_code]} sentences that:
    1. Continue this story naturally: {json.dumps(previous_sentences, ensure_ascii=False)}
    2. Target difficulty level {target_difficulty} using these criteria:

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

    Output format:
    Return a JSON array where each object has:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this continues the story AND matches difficulty>",
        "cognate_words": [<List of cognates used>]
    }}
    Generate {LLM_CALL_BATCH_SIZE} different sentences meeting these criteria (difficulty level and story continuation).
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    try:
        # Generate continuation options
        response = client.chat.completions.create(
            model=SENTENCE_GENERATION_MODEL,
            messages=[{'role': 'system', 'content': system_prompt}],
            temperature=1.0,
            max_tokens=600
        )
        
        # Parse and score the generated sentences
        generated_batch = json.loads(response.choices[0].message.content)
        sentences_to_score = [item['sentence'] for item in generated_batch]
        score_results = gpt_scored_rubric_batch(sentences_to_score)
        
        # Find sentence closest to target difficulty
        best_sentence = None
        min_diff = float('inf')
        
        for gen, score in zip(generated_batch, score_results):
            diff = abs(score['score'] - target_difficulty)
            if diff < min_diff:
                min_diff = diff
                best_sentence = {
                    'sentence': gen['sentence'],
                    'target_difficulty': target_difficulty,
                    'proposed_cognate_words': gen['cognate_words'],
                    'generation_reasoning': gen['reasoning'],
                    'actual_score': score['score'],
                    'actual_score_reasoning': score['reasoning'],
                    'actual_cognate_words': score['cognate_words'],
                    'generation_timestamp': datetime.datetime.now().isoformat()
                }
        
        return best_sentence
        
    except Exception as e:
        print(f"Error generating continuation: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_story(lang_code='fr', story_length=10):
    """
    Generate a complete story with consistent difficulty level.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{lang_code}_story_{timestamp}.json'
    
    # Initialize story data
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
    
    # Randomly choose target difficulty
    target_difficulty = random.randint(0, 3)
    story_data['metadata']['target_difficulty'] = target_difficulty
    
    print(f"Generating story with target difficulty: {target_difficulty}")
    
    # Generate first sentence using existing batch generator
    initial_batch = generate_difficulty_targeted_sentences_batch(lang_code, target_difficulty, output_file)
    if initial_batch:
        story_data['story'].append(initial_batch[0])
        
        # Generate remaining sentences
        while len(story_data['story']) < story_length:
            print(f"Generating sentence {len(story_data['story']) + 1}/{story_length}")
            
            previous_sentences = [item['sentence'] for item in story_data['story']]
            next_sentence = generate_story_continuation(
                lang_code, 
                target_difficulty,
                previous_sentences,
                output_file
            )
            
            if next_sentence:
                story_data['story'].append(next_sentence)
            
            # Save progress after each sentence
            story_data['metadata']['sentence_count'] = len(story_data['story'])
            story_data['metadata']['actual_difficulty_mean'] = sum(
                s['actual_score'] for s in story_data['story']
            ) / len(story_data['story'])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, ensure_ascii=False, indent=2)
    
    return story_data, output_file

# New main function for story generation
if __name__ == "__main__":
    story_data, output_file = generate_story(lang_code='fr', story_length=10)
    print(f"\nStory generated and saved to: {output_file}")
    print(f"Target difficulty: {story_data['metadata']['target_difficulty']}")
    print(f"Actual mean difficulty: {story_data['metadata']['actual_difficulty_mean']:.2f}")
