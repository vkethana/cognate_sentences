# app.py
from openai import OpenAI
from flask import Flask, render_template, jsonify, request
import os
import json
import random
from typing import List, Dict
import math

app = Flask(__name__)

STORIES_DIR = "batch_stories"
SENTENCE_SCORING_MODEL = 'gpt-4o-mini'

def llm_score_translation(original: str, translation: str, api_key: str) -> Dict:
    """
    Scores a translation using the Language Model API.
    
    Args:
        original: The original sentence in the source language
        translation: The translated sentence in the target language
        api_key: The OpenAI API key
        
    Returns:
        is_correct: A boolean indicating if the translation is correct
        wrong_morphemes: A list of morphemes that were incorrect
    """
    client = OpenAI(api_key=api_key)
    system_prompt = f"""
        You are an expert in French-English translation. I will give you a sentence in French and a sentence in English. (The input will be provided in JSON format as described below.) Your job is to tell me whether the English sentence is a correct translation of the French sentence. If it is not, please identify words/morphemes that were incorrectly translated or are missing in the translation. You will be using a JSON format to provide your response.
        Here's the input format:
        {{
            "original": <The original French sentence>,
            "translation": <The attempted translation in English>
        }}

        Here's the output format:
        {{
          "is_correct": <a boolean indicating whether the translation is correct>,
          "incorrect_morphemes": [A list of morphemes in the original French sentence that were incorrectly translated or are missing in the translation sentence. Make sure that anything you include in the list is a character-for-character match from the original French sentence. Do not include words that the translation got correct or words that are not in the `original` sentence.],
          "reasoning": "<Reasoning for your scoring. You may be brief if the translation is correct.>",
        }}

        Here's an example. Suppose I give you the input:
        {{
            "original": "Voulez-vous aller manger avec moi",
            "translation": "Do you want to go eat with me?"
        }}
        You would respond with:
        {{
          "is_correct": true,
          "incorrect_morphemes": [],
          "reasoning": "The translation is correct."
        }}

        But if I gave you the input:
        {{
            "original": "Voulez-vous aller manger avec moi",
            "translation": "Does he want to go sing with me tomorrow?"
        }}
        You would respond with:
        {{
          "is_correct": false,
          "incorrect_morphemes": ["vous", "manger"],
          "reasoning": "The translation has an incorrect pronoun and verb. It also has an extra word."
        }}

        Observe that in this last example, the user included extraneous words, but none of those extraneous words made it into the incorrect_morphemes field. 

        Now let's get started. Here's your input:
        {{
          "original": "{original}",
          "translation": "{translation}"
        }}
        Note: Please avoid including Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
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
        print("Translation scoring results")
        print(results)
        return results
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        raise

def calculate_story_difficulty(story_data: Dict) -> float:
    """Calculate average difficulty of a story based on sentence difficulties."""
    difficulties = [sentence['actual_score'] for sentence in story_data['story']]
    return sum(difficulties) / len(difficulties)

def get_story_candidates(target_difficulty: float, seen_stories: List[str], tolerance: float = 0.5) -> str:
    """
    Returns a random story filename that:
    1. Hasn't been seen before
    2. Has difficulty close to target_difficulty
    
    Args:
        target_difficulty: The target difficulty level (0-3)
        seen_stories: List of previously seen story filenames
        tolerance: How far from target difficulty we're willing to go
    """
    story_files = [f for f in os.listdir(STORIES_DIR) if f.endswith('.json')]
    available_stories = [f for f in story_files if f not in seen_stories]
    
    if not available_stories:
        # If all stories have been seen, reset the seen stories list
        available_stories = story_files
    
    # Calculate difficulties for available stories
    story_difficulties = {}
    for filename in available_stories:
        with open(os.path.join(STORIES_DIR, filename), 'r', encoding='utf-8') as f:
            story_data = json.load(f)
            story_difficulties[filename] = calculate_story_difficulty(story_data)
    
    # Find stories within tolerance range
    candidates = [
        filename for filename, difficulty in story_difficulties.items()
        if abs(difficulty - target_difficulty) <= tolerance
    ]
    
    if candidates:
        return random.choice(candidates)
    else:
        # If no stories within tolerance, pick the closest one
        return min(story_difficulties.items(), key=lambda x: abs(x[1] - target_difficulty))[0]

def load_story(filename: str) -> Dict:
    """Loads and returns story data from a JSON file."""
    with open(os.path.join(STORIES_DIR, filename), 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/score_translation', methods=['POST'])
def score_translation():
    # Get the OpenAI API key from headers to simulate real authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid API key'}), 401

    # Get the translation data
    data = request.get_json()
    original = data.get('original')
    translation = data.get('translation')
    api_key = auth_header.split('Bearer ')[1].strip()

    if not original or not translation:
        return jsonify({'error': 'Missing original or translation text'}), 400

    wrong_morphemes = []
    scoring_results = llm_score_translation(original, translation, api_key)
    try: 
        is_correct, wrong_morphemes = bool(scoring_results['is_correct']), list(scoring_results['incorrect_morphemes'])
    except:
        is_correct = False
        wrong_morphemes = []
        feedback = "Error: Could not score the translation."

    return jsonify({
        'isCorrect': is_correct,
        'wrongMorphemes': wrong_morphemes,
        'feedback': 'Great job!' if is_correct else 'Try again with a different translation.'
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-sentence', methods=['POST'])
def get_sentence():
    data = request.get_json()
    
    if data.get('needNewStory'):
        # Get user's current difficulty and seen stories
        user_difficulty = float(data.get('userDifficulty', 2.5))  # Default to middle difficulty
        seen_stories = data.get('seenStories', [])
        
        # Select appropriate story
        story_file = get_story_candidates(user_difficulty, seen_stories)
        story_data = load_story(story_file)
        sentence_data = story_data['story'][0]
        
        return jsonify({
            'sentence': sentence_data['sentence'],
            'storyFile': story_file,
            'isLastSentence': False,
            'storyDifficulty': calculate_story_difficulty(story_data),
            'sentenceDifficulty': sentence_data['actual_score']
        })
    
    else:
        # Get next sentence from current story
        story_file = data['storyFile']
        sentence_index = int(data['sentenceIndex'])
        
        story_data = load_story(story_file)
        
        # Check if we've reached the end of the story
        if sentence_index >= len(story_data['story']):
            return jsonify({
                'isLastSentence': True
            })
        
        sentence_data = story_data['story'][sentence_index]
        return jsonify({
            'sentence': sentence_data['sentence'],
            'isLastSentence': sentence_index == len(story_data['story']) - 1,
            'sentenceDifficulty': sentence_data['actual_score']
        })

@app.route('/story_list', methods=['GET'])
def get_story_list():
    """Returns a list of all available stories with their metadata."""
    story_files = [f for f in os.listdir(STORIES_DIR) if f.endswith('.json')]
    stories = []
    
    for filename in story_files:
        with open(os.path.join(STORIES_DIR, filename), 'r', encoding='utf-8') as f:
            story_data = json.load(f)
            stories.append({
                'title': filename.replace('.json', ''),
                'difficulty': calculate_story_difficulty(story_data),
                'num_sentences': len(story_data['story'])
            })
    
    return jsonify(stories)

@app.route('/story_list/<story_title>', methods=['GET'])
def get_story_details(story_title):
    """Returns detailed information about a specific story."""
    try:
        filename = f"{story_title}.json"
        with open(os.path.join(STORIES_DIR, filename), 'r', encoding='utf-8') as f:
            story_data = json.load(f)
            
        sentences = [
            {
                'text': sentence['sentence'],
                'difficulty': sentence['actual_score']
            }
            for sentence in story_data['story']
        ]
            
        return jsonify({
            'title': story_title,
            'sentences': sentences
        })
    except FileNotFoundError:
        return jsonify({'error': 'Story not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
