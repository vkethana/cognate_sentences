# app.py
from flask import Flask, render_template, jsonify, request
import os
import json
import random
from typing import List, Dict
import math

app = Flask(__name__)

STORIES_DIR = "batch_stories"

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
    '''
    # Get the OpenAI API key from headers to simulate real authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid API key'}), 401
        
    # Get the translation data
    data = request.get_json()
    original = data.get('original')
    translation = data.get('translation')
    
    if not original or not translation:
        return jsonify({'error': 'Missing original or translation text'}), 400
    
    # Randomly approve or reject the translation (50% chance each)
    '''
    import random
    is_correct = random.choice([True, False])
    
    return jsonify({
        'isCorrect': is_correct,
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
        user_difficulty = float(data.get('userDifficulty', 1.5))  # Default to middle difficulty
        seen_stories = data.get('seenStories', [])
        
        # Select appropriate story
        story_file = get_story_candidates(user_difficulty, seen_stories)
        story_data = load_story(story_file)
        sentence_data = story_data['story'][0]
        
        return jsonify({
            'sentence': sentence_data['sentence'],
            'storyFile': story_file,
            'isLastSentence': False,
            'storyDifficulty': calculate_story_difficulty(story_data)
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
            'isLastSentence': sentence_index == len(story_data['story']) - 1
        })

if __name__ == '__main__':
    app.run(debug=True)
