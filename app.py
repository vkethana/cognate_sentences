from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from openai_beam_search import run_beam_search, init_beam_search, get_cognates, score_sentence
from random import choice
from gpt_scored_search import evaluate_translation, gpt_extend_sentence, gpt_generate_new_sentence, get_wrong_words

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
  print("Evaluating translation...")
  data = request.get_json()
  print(data)
  user_translation = data['user_translation']
  original_sentence = data['original_sentence']

  # Check whether the two are equal
  is_correct = evaluate_translation(original_sentence, user_translation)
  print("Is correct:", is_correct)
  wrong_words = []

  if (not bool(is_correct)):
    print("Getting wrong words...")
    wrong_words = get_wrong_words(original_sentence, user_translation)

  output = {
    'is_correct': is_correct,
    'wrong_words': wrong_words
  }
  print("Returning" + str(output))
  # Return a response
  return jsonify(output)

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    sentence = gpt_generate_new_sentence()
    cognate_list = get_cognates(sentence.split(" "))
    highlighted_sentence = get_highlighted(sentence, cognate_list)
    print("highlighted_sentence=", highlighted_sentence)
    output = {
      'sentence': highlighted_sentence,
      'score': score_sentence(highlighted_sentence)
    }

    return jsonify(output)

@app.route('/extend_sentence', methods=['POST'])
def extend_sentence():
    data = request.get_json()
    extended_sentence = gpt_extend_sentence(data['original_sentence'])
    output = {
      'sentence': extended_sentence,
      'score': score_sentence(extended_sentence)
    }
    return jsonify(output)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
@app.route('/extend_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
