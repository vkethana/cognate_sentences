import datetime
import os
from openai import OpenAI
import json
import random
import math
from utils import *

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o'
SENTENCE_SCORING_MODEL = 'o1-preview'
data_directory = 'batch_stories_4o_generate_o1_score'

def gpt_scored_rubric_batch(sentences):
    '''
    Score multiple French sentences at once using GPT-4.

    Args:
        sentences: List of sentences to score
    Returns:
        List of scoring results
    '''

    system_prompt = f"""
    You are an expert in French to English translation. I will give you {len(sentences)} sentences in French, and I want you to score each of them on a scale from 0-3 using the following rubric:

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

    Please format your response as a JSON array of these objects. You should have {len(sentences)} objects in your array.

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

def generate_story(lang_code, num_sentences, target_difficulty):
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English.
    Generate exactly {num_sentences} {language_codes[lang_code]} sentences that:
    1. Form a coherent narrative where each sentence follows from the previous one
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

    Format your response as a JSON array of {num_sentences} objects:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this sentence matches difficulty. If this is not the first sentence, also explain why this continues the story from the previous sentence in this JSON array.>",
        "cognate_words": [<List of cognates used>]
    }}

    Important: Each sentence must directly follow from the previous one to form a coherent story.
    Generate {num_sentences} sentences meeting these criteria (difficulty level and story continuation).
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """
    
    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[{'role': 'user', 'content': system_prompt}],
        temperature=1.0
    )
    
    # Parse generated sentences
    return json.loads(response.choices[0].message.content)

def generate_story_batch(lang_code, story_length):
    """
    Generate a story consisting of story_length sentences.

    Args:
        lang_code (str): Language code ('fr' for French)
        story_length (int): Target number of sentences in the story

    Returns:
        tuple: (story_data dictionary, output_filename)
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{data_directory}/{lang_code}_batch_story_{timestamp}.json'

    # Initialize story data structure
    #target_difficulty = random.randint(0, 3)
    target_difficulty = random.choice([0, 1, 2, 3, 3, 3])
    story_data = {
        'story': [],
        'metadata': {
            'language': lang_code,
            'target_difficulty': None,
            'actual_difficulty_mean': None,
            'creation_date': datetime.datetime.now().isoformat(),
            'sentence_count': 0,
            'target_difficulty': target_difficulty,
            'generation_model': SENTENCE_GENERATION_MODEL,
            'scoring_model': SENTENCE_SCORING_MODEL
        }
    }

    sentences = generate_story(lang_code, story_length, target_difficulty)

    sentences_to_score = [item['sentence'] for item in sentences]
    score_results = gpt_scored_rubric_batch(sentences_to_score)

    # Combine generation and scoring data
    timestamp = datetime.datetime.now().isoformat()
    for gen, score in zip(sentences, score_results):
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
    
    return story_data, output_file

# New main function for story generation
if __name__ == "__main__":
    for _ in range(40):
        print("Generating new story batch...")
        story_data, output_file = generate_story_batch(
            lang_code='fr',
            story_length=10
        )
        print(f"\nStory batch generated and saved to: {output_file}")
        print("Story metadata:", story_data['metadata'])
