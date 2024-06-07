from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from openai_beam_search import run_beam_search, init_beam_search
from random import choice
from gpt_scored_search import evaluate_translation

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

@app.route('/evaluate_translation', methods=['POST'])
def eval_translation():
  # Get the user's input
  data = request.get_json()
  print(data)
  user_translation = data['user_translation']
  original_sentence = data['original_sentence']

  # Check whether the two are equal
  is_correct = evaluate_translation(original_sentence, user_translation)
  print("is_correct=", is_correct)
  # Return a response
  return jsonify(is_correct)


@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    global candidates
    candidates = []
    sentence_starters = ["le président George Bush", "la ville de New York", "la ville de San Francisco", "le gouvernement américain", "le premier ministre Justin Trudeau"]
    sentence_starters.extend([ "Le", "L'", "Les", "De", "Au", "Par", "De plus", "La", "En", "Créée par", "Cette", "Pour", "Une", "Un climat", "Une précaution", "Je" ])
    #sentence_starters = ["el presidente de Argentina", "en el país de México", "la ciudad de Nueva York", "barcelona es"]
    # if you want to test beam search with a different language, make sure you change target_lang = 'es
    output = {
      'sentence': "Je suis Vijay"
    }
    return jsonify(output)
    '''
    first_sentence = choice(sentence_starters)
    candidates = init_beam_search(first_sentence, 3)
    sentences_to_render = [(get_highlighted(c.sentence, c.cognates), c.score_breakdown) for c in candidates]
    print("Sentences_to_render = {}".format(sentences_to_render))
    if len(sentences_to_render) == 0:
      return render_template('index.html', sentence="", breakdown = {}, inside_beam_search=True)
    else:
      return render_template('index.html', sentence=sentences_to_render[0][0], breakdown = str(sentences_to_render[0][1]), inside_beam_search=True)
    '''

@app.route('/extend_sentence', methods=['POST'])
def extend_sentence():
    data = request.get_json()
    output = {
      'sentence': data['original_sentence'] + " et "
    }
    return jsonify(output)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
@app.route('/extend_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
