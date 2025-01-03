from flask import Flask, jsonify, render_template, request, session
import json
import os
import random
from typing import List, Dict
import numpy as np

app = Flask(__name__)
app.secret_key = 'ilikeicecream123'  # Required for session management

class TranslationAttempt:
    def __init__(self):
        self.attempts = 0
        self.success = False

class StoryManager:
    def __init__(self, data_dir: str, num_tries: int = 3):
        self.data_dir = data_dir
        self.num_tries = num_tries
        self.current_file = None
        self.current_story = None
        self.current_index = 0
        self.translation_attempts = {}  # Only tracks attempts within current story

    def get_next_sentence(self) -> Dict:
        # Validate current index
        if self.current_index < 0 or self.current_index >= len(self.current_story['story']):
            self.current_index = 0  # Reset index if invalid

        # Select a new story if needed
        if self.current_story is None or self.current_index >= len(self.current_story['story']):
            return self.select_story(session.get('user_difficulty', 3.0))

        current_sentence = self.current_story['story'][self.current_index]
        self.current_index += 1

        return {
            'sentence': current_sentence['sentence'],
            'isNewStory': self.current_index == 1,
            'language': 'fr',
            'needsQuiz': True,
            'sentenceDifficulty': current_sentence['actual_score'],
            'storyDifficulty': self._calculate_story_difficulty(self.current_story),
        }

    def _load_story(self, filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _calculate_story_difficulty(self, story: Dict) -> float:
        return np.mean([sentence['actual_score'] for sentence in story['story']])
        
    def select_story(self, user_difficulty: float) -> Dict:
        # Get all available stories
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        # Calculate difficulties and distances from user_difficulty
        stories_with_scores = []
        for filename in json_files:
            story = self._load_story(os.path.join(self.data_dir, filename))
            difficulty = self._calculate_story_difficulty(story)
            distance = abs(difficulty - user_difficulty)
            stories_with_scores.append((filename, story, distance))
            
        # Sort by distance and select from top 3 randomly
        stories_with_scores.sort(key=lambda x: x[2])
        selection_pool = stories_with_scores[:3]
        selected = random.choice(selection_pool)
        
        self.current_file = selected[0]
        self.current_story = selected[1]
        self.current_index = 0
        self.translation_attempts = {}  # Reset attempts for new story
        
        return self.get_next_sentence()

# Create story manager instance
story_manager = StoryManager('batch_stories', num_tries=3)

@app.route('/')
def home():
    if 'user_difficulty' not in session:
        print("Resetting everything")
        # New session detected
        session['user_difficulty'] = 3.0
        story_manager.current_file = None
        story_manager.current_story = None
        story_manager.current_index = 0
        story_manager.translation_attempts = {}
    return render_template('index.html')

@app.route('/next-sentence', methods=['POST'])
def next_sentence():
    data = request.get_json()
    current_story_file = data.get('currentStory')
    current_index = data.get('currentIndex', 0)

    # If the client provides a story file, load it into the StoryManager
    if current_story_file:
        if current_story_file != story_manager.current_file:
            # If the file is different, load the new story
            story_manager.current_file = current_story_file
            story_manager.current_story = story_manager._load_story(
                os.path.join(story_manager.data_dir, current_story_file)
            )
            story_manager.current_index = current_index
        else:
            # If it's the same file, just update the index
            story_manager.current_index = current_index
    else:
        # If no file is provided, select a new story
        story_manager.select_story(session.get('user_difficulty', 3.0))
    
    # Get the next sentence
    response = story_manager.get_next_sentence()
    response.update({
        'storyFile': story_manager.current_file,
        'currentIndex': story_manager.current_index,
    })
    return jsonify(response)

@app.route('/check-translation', methods=['POST'])
def check_translation():
    story_manager = StoryManager('batch_stories', num_tries=3)
    data = request.get_json()
    result = story_manager.check_translation(
        data.get('original', ''),
        data.get('translation', '')
    )
    return jsonify({
        **result,
        'userDifficulty': session.get('user_difficulty', 3.0)
    })

@app.route('/get-stats')
def get_stats():
    return jsonify({
        'userDifficulty': session.get('user_difficulty', 3.0)
    })

if __name__ == '__main__':
    app.run(debug=True)
