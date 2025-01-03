# app.py
from flask import Flask, render_template, jsonify, request
import os
import json
import random

app = Flask(__name__)

STORIES_DIR = "batch_stories"

def get_random_story_file():
    """Returns a random story filename from the batch_stories directory."""
    story_files = [f for f in os.listdir(STORIES_DIR) if f.endswith('.json')]
    return random.choice(story_files)

def load_story(filename):
    """Loads and returns story data from a JSON file."""
    with open(os.path.join(STORIES_DIR, filename), 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-sentence', methods=['POST'])
def get_sentence():
    data = request.get_json()
    
    if data.get('needNewStory'):
        # Select a random story file
        story_file = get_random_story_file()
        story_data = load_story(story_file)
        sentence_data = story_data['story'][0]
        
        return jsonify({
            'sentence': sentence_data['sentence'],
            'storyFile': story_file,
            'isLastSentence': False
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
    
    if not original or not translation:
        return jsonify({'error': 'Missing original or translation text'}), 400
    
    # Randomly approve or reject the translation (50% chance each)
    import random
    is_correct = random.choice([True, False])
    
    return jsonify({
        'isCorrect': is_correct,
        'feedback': 'Great job!' if is_correct else 'Try again with a different translation.'
    })

if __name__ == '__main__':
    app.run(debug=True)
