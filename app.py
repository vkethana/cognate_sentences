from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from openai_beam_search import run_beam_search, init_beam_search
from random import choice

app = Flask(__name__, static_folder="templates/static")

# it's helpful to not hard-code the lang codes
lang_codes = {
  'es': 'Spanish',
  'fr': 'French',
  'en': 'English'
}

src_lang = 'en'
target_lang = 'fr'
candidates = []

def get_highlighted(sentence, cognate_list):
    if (sentence == None or sentence == ""):
      return ""
    highlighted_sentence = sentence
    if (cognate_list == None or len(cognate_list) == 0):
      return highlighted_sentence
    for word in cognate_list:
        highlighted_sentence = re.sub(r'\b({})\b'.format(re.escape(word)), r'<span class="highlight">\1</span>', highlighted_sentence)
    return highlighted_sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    global candidates
    candidates = []
    sentence_starters = ["le président George Bush", "la ville de New York", "la ville de San Francisco", "le gouvernement américain", "le premier ministre Justin Trudeau"]
    #sentence_starters = ["el presidente de Argentina", "en el país de México", "la ciudad de Nueva York", "barcelona es"]
    # if you want to test beam search with a different language, make sure you change target_lang = 'es'

    first_sentence = choice(sentence_starters)
    candidates = init_beam_search(first_sentence, 3)
    sentences_to_render = [(get_highlighted(c.sentence, c.cognates), c.score_breakdown) for c in candidates]

    return render_template('index.html', sentences=sentences_to_render, inside_beam_search=True)

@app.route('/extend_sentence', methods=['POST'])
def extend_sentence():
    global candidates
    candidates = run_beam_search(candidates, 3)
    sentences_to_render = [(get_highlighted(c.sentence, c.cognates), c.score_breakdown) for c in candidates]
    return render_template('index.html', sentences=sentences_to_render, inside_beam_search=True)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
@app.route('/extend_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
