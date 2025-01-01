import datetime
import os
from openai import OpenAI
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o-mini'
SENTENCE_SCORING_MODEL = 'o1-preview'
num_choices = 3

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

def build_adaptive_database(lang_code, target_counts={0: 1000, 1: 1000, 2: 1000, 3: 1000}, output_file='sentence_database.json'):
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
        
        generate_difficulty_targeted_sentences(
            lang_code,
            target_diff,
            output_file,
            batch_size=50
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

def generate_difficulty_targeted_sentences(lang_code, target_difficulty, output_file, batch_size=10):
    """
    Generate sentences targeting a specific difficulty level (0-3).
    Saves all valid sentences regardless of whether they hit the target difficulty.
    """
    system_prompt = f"""You are a fluent speaker of both {language_codes[lang_code]} and English. Generate ONE {language_codes[lang_code]} sentence aiming for difficulty level {target_difficulty} on this cognate difficulty scale:

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
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this should score {target_difficulty}>",
        "cognate_words": [<List of cognates used>],
        "generation_timestamp": "<Current timestamp>"
    }}
    """

    batch_sentences = []
    for i in range(batch_size):
        try:
            # Adjust temperature based on difficulty
            temp = 1.4 if target_difficulty in [0, 3] else 1.0
            
            response = client.chat.completions.create(
                model=SENTENCE_GENERATION_MODEL,
                messages=[{'role': 'system', 'content': system_prompt}],
                temperature=temp,
                max_tokens=200
            )
            
            generated = json.loads(response.choices[0].message.content)
            generated['generation_timestamp'] = datetime.datetime.now().isoformat()
            
            # Score the generated sentence
            score = gpt_scored_rubric_individual(generated['sentence'])
            
            sentence_data = {
                'sentence': generated['sentence'],
                'target_difficulty': target_difficulty,
                'proposed_cognate_words': generated['cognate_words'],
                'generation_reasoning': generated['reasoning'],
                'actual_score': score['score'],
                'actual_score_reasoning': score['reasoning'],
                'actual_cognate_words': score['cognate_words'],
                'generation_timestamp': generated['generation_timestamp']
            }
            
            batch_sentences.append(sentence_data)
            
            # Save every 5 sentences to avoid losing data
            if len(batch_sentences) % 5 == 0:
                save_sentences_batch(batch_sentences, output_file)
                batch_sentences = []
                
            print(f"Generated sentence {i+1}/{batch_size} - Target: {target_difficulty}, Actual: {score['score']}")
            
        except Exception as e:
            print(f"Error generating sentence: {e}")
            continue
    
    # Save any remaining sentences
    if batch_sentences:
        save_sentences_batch(batch_sentences, output_file)

def resume_adaptive_database(input_file, target_counts={0: 1000, 1: 1000, 2: 1000, 3: 1000}, 
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
            needed_difficulties = [int(d) for d, count in current_counts.items() 
                                 if count < target_counts[d]]
            
            if not needed_difficulties:
                print("All targets reached! Generation complete.")
                break
            
            # Generate batch targeting the most-needed difficulty
            target_diff = needed_difficulties[0]
            
            generate_difficulty_targeted_sentences(
                lang_code,
                target_diff,
                output_file,
                batch_size=50
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

def start_new_database(lang_code='fr', sentences_per_difficulty=1000):
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

# Simple usage example:
if __name__ == "__main__":
    # Start a new database with 1000 sentences per difficulty level
    start_new_database(lang_code='fr', sentences_per_difficulty=10)
