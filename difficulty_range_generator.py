from sentence_generator import *
import datetime
import os

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
                'actual_score': score['score'],
                'cognate_words': generated['cognate_words'],
                'reasoning': generated['reasoning'],
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

# Usage example
if __name__ == "__main__":
    
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'french_sentences_{timestamp}.json'
    
    build_adaptive_database(
        'fr',
        target_counts={0: 1000, 1: 1000, 2: 1000, 3: 1000},
        output_file=output_file
    )
