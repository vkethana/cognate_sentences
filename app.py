from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
import json
from random import choice, randint

from backend import *

app = Flask(__name__, static_folder="templates/static")
USE_PREGENERATED_DATA = True

if USE_PREGENERATED_DATA:
  with open('data/test_jul5.json', 'r') as f:
      stories = json.load(f)

# it's helpful to not hard-code the lang codes
lang_codes = {
  'es': 'Spanish',
  'fr': 'French',
  'en': 'English'
}

src_lang = 'en'
target_lang = 'fr'
candidates = []

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

def turn_sentence_index_into_data_to_show_to_user(story_index, sentence_index):
    sentence_list = stories[str(story_index)]['sentences']
    sentence = sentence_list[sentence_index]['unhighlighted_sentence']
    cognates = sentence_list[sentence_index]['cognates']
    score = sentence_list[sentence_index]['score']

    output = {
    'sentence': sentence,
    'cognates': cognates,
    'score': score,
    'is_pregenerated': True,
    "story_index": story_index,
    "sentence_index": sentence_index
    }
    return output

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    if USE_PREGENERATED_DATA:
      story_index = randint(0, len(stories) - 1)
      return turn_sentence_index_into_data_to_show_to_user(story_index, 0)

    else:
      # This won't work anymore because there is no highlighted sentence attribute
      # Will need to manually recreate it if you want to bring back this functionality (deprecated on purpose)
      sentence_starter = get_sentence_starter()
      sentence_object = make_sentence_object(sentence_starter)

      beams = init_beam_search(sentence_object, 3)
      return jsonify(get_sentence_as_json(one_step_forward(beams)))

@app.route('/extend_sentence', methods=['POST'])
def extend_sentence():
    data = request.get_json()
    if ('story_index' in data.keys()):
      story_index = data['story_index'] # this is an str, not an int
      sentence_index = (int(data['sentence_index']) + 1)  % len(stories[story_index]['sentences'])
      return turn_sentence_index_into_data_to_show_to_user(story_index, sentence_index)

    else:
      data = request.get_json()
      sentence_object = make_sentence_object(data['original_sentence'])

      beams = init_beam_search(sentence_object, 3)
      return jsonify(get_sentence_as_json(one_step_forward(beams)))

if __name__ == '__main__':
    app.run(debug=True)
