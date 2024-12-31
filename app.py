from flask import Flask, jsonify, render_template, request
import json
import os
import random

app = Flask(__name__)

class TranslationChecker:
    @staticmethod
    def check_translation(original, proposed):
        # Dummy implementation that returns True 50% of the time
        return random.random() < 0.5

class StoryManager:
    def __init__(self, data_dir, quiz_frequency=3):
        self.data_dir = data_dir
        self.current_file = None
        self.current_sentences = []
        self.current_index = 1  # Start at 1 as discussed
        self.quiz_frequency = quiz_frequency
        self.sentences_since_quiz = 0
        self.load_new_file()
    
    def load_new_file(self):
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not json_files:
            raise Exception("No JSON files found in data directory")
        
        selected_file = random.choice(json_files)
        
        with open(os.path.join(self.data_dir, selected_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.current_sentences = data['sentences']
            self.current_index = 1  # Start at 1 since we'll return first sentence immediately
            self.current_file = selected_file
            return {
                'sentence': self.current_sentences[0]['sentence'],
                'isNewStory': True,
                'language': data.get('language', 'en'),
                'needsQuiz': False
            }
    
    def get_next_sentence(self):
        if self.current_index >= len(self.current_sentences):
            return self.load_new_file()

        self.sentences_since_quiz += 1
        needs_quiz = self.sentences_since_quiz >= self.quiz_frequency
        
        sentence = self.current_sentences[self.current_index]['sentence']
        self.current_index += 1
        
        response = {
            'sentence': sentence,
            'isNewStory': False,
            'language': None,
            'needsQuiz': needs_quiz
        }
        
        if needs_quiz:
            self.sentences_since_quiz = 0
            
        return response

    def check_translation(self, original, proposed):
        return TranslationChecker.check_translation(original, proposed)

# Create story manager instance
story_manager = StoryManager('data', quiz_frequency=3)

@app.route('/')
def home():
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
    return jsonify({'correct': result})

if __name__ == '__main__':
    app.run(debug=True)
