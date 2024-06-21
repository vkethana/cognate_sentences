from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from random import choice, randint
from backend import evaluate_translation, get_wrong_words, get_sentence_starter, run_search, gpt_rank, identify_cognates, init_beam_search, make_sentence_object
import json

app = Flask(__name__, static_folder="templates/static")
USE_PREGENERATED_DATA = True

# it's helpful to not hard-code the lang codes
lang_codes = {
  'es': 'Spanish',
  'fr': 'French',
  'en': 'English'
}

src_lang = 'en'
target_lang = 'fr'
candidates = []

# read production_data/ui_mockup.json
with open('production_data/preloaded.json', 'r') as f:
    stories = json.load(f)['stories']

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

  if (not is_correct):
    print("Getting wrong words...")
    wrong_words = get_wrong_words(original_sentence, user_translation)

  output = {
    'is_correct': is_correct,
    'wrong_words': wrong_words
  }

  print("Returning" + str(output))
  # Return a response
  return jsonify(output)

def one_step_forward(curr_sentence):
    new_beams = run_search(curr_sentence)
    best_sentence_index = gpt_rank([i.sentence for i in new_beams]) - 1
    best_sentence_object = new_beams[best_sentence_index]

    highlighted_sentence = get_highlighted(best_sentence_object.sentence, best_sentence_object.cognates)
    print("highlighted_sentence=", highlighted_sentence)

    output = {
      'sentence': highlighted_sentence,
      'score': best_sentence_object.score_breakdown['total_score'],
      "is_pregenerated": False
    }
    return output

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    if USE_PREGENERATED_DATA:
      index = randint(0, len(stories) - 1)
      sentence_list = stories[index]['sentences']
      sentence = sentence_list[0]['raw_text']
      score = sentence_list[0]['score']
      output = {
        'sentence': sentence,
        'score': score,
        'is_pregenerated': True,
        "story_index": index,
        "sentence_index": 0
      }
      return jsonify(output)
    else:
      sentence_starter = get_sentence_starter()
      sentence_object = make_sentence_object(sentence_starter)

      beams = init_beam_search(sentence_object, 3)
      return jsonify(one_step_forward(beams))

@app.route('/extend_sentence', methods=['POST'])
def extend_sentence():
    data = request.get_json()
    if ('story_index' in data.keys()):

      story_index = int(data['story_index'])
      sentence_index = (int(data['sentence_index']) + 1 ) % len(stories[story_index]['sentences'])

      sentence_list = stories[story_index]['sentences']
      sentence = sentence_list[sentence_index]['raw_text']
      score = sentence_list[sentence_index]['score']

      output = {
        'sentence': sentence,
        'score': score,
        'is_pregenerated': True,
        "story_index": story_index,
        "sentence_index": sentence_index
      }
      return jsonify(output)
    else:
      data = request.get_json()
      sentence_object = make_sentence_object(data['original_sentence'])

      beams = init_beam_search(sentence_object, 3)
      return jsonify(one_step_forward(beams))

if __name__ == '__main__':
    app.run(debug=True)
