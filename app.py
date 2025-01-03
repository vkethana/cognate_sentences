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
        self.shown_stories = set()
        self.translation_attempts = {}

    def get_next_sentence(self) -> Dict:
        # If there's no current story or we've reached the end, select a new one
        if (self.current_story is None or 
            self.current_index >= len(self.current_story['story'])):
            self.current_index = 0
            return self.select_story(session.get('user_difficulty', 3.0))
            
        current_sentence = self.current_story['story'][self.current_index]
        self.current_index += 1

        result = {
            'sentence': current_sentence['sentence'],
            'isNewStory': self.current_index == 1,
            'current_index': self.current_index,
            #'language': self.current_story['metadata']['language'],
            'needsQuiz': True,  # We'll quiz on every sentence
            'sentenceDifficulty': current_sentence['actual_score'],
            'storyDifficulty': self._calculate_story_difficulty(self.current_story)
        }
        print("returning result", result)
        return result

    def _load_story(self, filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _calculate_story_difficulty(self, story: Dict) -> float:
        return np.mean([sentence['actual_score'] for sentence in story['story']])
        
    def select_story(self, user_difficulty: float) -> Dict:
        # Get all available stories
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        available_files = [f for f in json_files if f not in self.shown_stories]
        
        if not available_files:
            self.shown_stories.clear()  # Reset if all stories have been shown
            available_files = json_files
            
        # Calculate difficulties and distances from user_difficulty
        stories_with_scores = []
        for filename in available_files:
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
        self.shown_stories.add(self.current_file)
        self.translation_attempts = {}
        
        return self.get_next_sentence()
        
        
    def check_translation(self, original: str, translation: str) -> Dict:
        if original not in self.translation_attempts:
            self.translation_attempts[original] = TranslationAttempt()
            
        attempt = self.translation_attempts[original]
        attempt.attempts += 1
        
        # TODO: Implement proper translation checking
        is_correct = random.random() < 0.5  # Temporary random check
        
        if is_correct:
            attempt.success = True
            self._update_user_difficulty(attempt.attempts)
            
        return {
            'correct': is_correct,
            'attemptsLeft': self.num_tries - attempt.attempts if not is_correct else 0
        }
        
    def _update_user_difficulty(self, attempts: int):
        current_difficulty = session.get('user_difficulty', 3.0)
        
        # Adjust difficulty based on attempts needed
        if attempts == 1:
            adjustment = 0.2  # Big increase for getting it right first try
        elif attempts == 2:
            adjustment = 0.1  # Smaller increase for second try
        else:
            adjustment = 0.05  # Minimal increase for third try
            
        new_difficulty = min(max(current_difficulty + adjustment, 0), 3)
        session['user_difficulty'] = new_difficulty

# Create story manager instance
story_manager = StoryManager('batch_stories', num_tries=3)

@app.route('/')
def home():
    if 'user_difficulty' not in session:
        session['user_difficulty'] = 3.0
    return render_template('index.html')

@app.route('/next-sentence')
def next_sentence():
    return jsonify(story_manager.get_next_sentence())

@app.route('/check-translation', methods=['POST'])
def check_translation():
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
