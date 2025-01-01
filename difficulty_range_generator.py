from sentence_generator import *

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
