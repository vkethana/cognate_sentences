from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from generator import Generator
from database import Database
from scoring import score_sentence

app = Flask(__name__, static_folder="templates/static")

# instantiate new generator object with default attributes
# there are NO global variables for user target lang, difficulty, etc.
# the generator object will be updated with the user's choices
# so you don't need to keep track of these attributes separately
database = Database('data/europarl-en-es-3.pik')
generator = Generator(source='en', target='es', difficulty=0.2, database=database)
allow_other_languages = False

# it's helpful to not hard-code the lang codes
lang_codes = {
  'es': 'Spanish',
  'fr': 'French',
  'en': 'English'
}

@app.route('/')
def index():
    # Load the user cookie to see what src_lang and target_lang are

    # If they don't exist, set them to default values
    src_lang = 'en'
    target_lang = 'es'
    difficulty = 0.5

    src_lang = request.cookies.get('src_lang')
    target_lang = request.cookies.get('target_lang')

    return render_template('index.html', lang_codes=lang_codes, src_lang=src_lang, target_lang=target_lang, difficulty=difficulty, allow_other_languages=allow_other_languages)

# Route to update source_lang and target_lang variables
@app.route('/update_lang', methods=['POST'])
def update_lang():
    # Retrieve selected lang values from the request
    global generator
    src_lang = request.form.get('src_lang')
    target_lang = request.form.get('target_lang')
    difficulty = request.form.get('difficulty')
    difficulty = str(round(float(difficulty), 2))

    # store difficulty as a cookie
    response = jsonify({'status': 'success', 'src_lang': src_lang, 'target_lang': target_lang})
    response.set_cookie('difficulty', difficulty)
    print(response)

    print("Retreived data", src_lang, target_lang, difficulty)
    # Update the sentence generator object
    print("Updating generator with difficulty = ", difficulty)
    generator = Generator(source=src_lang, target=target_lang, difficulty=float(difficulty), database=database)

    # Return a response indicating success
    return response

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
  sentence = generator.get_sentence()
  print("Grabbed sentence with difficulty", sentence.difficulty, "for generator with difficulty", generator.difficulty)

  # check to see if difficulty value stored in cookie
  try:
    difficulty = request.cookies.get('difficulty')
    difficulty = round(float(difficulty), 2)
    print("Grabbing cookie ", difficulty , " with type", type(difficulty))
    if difficulty == None:
      difficulty = 0.5
  except:
    difficulty = 0.5

  results = score_sentence(sentence.sentence.replace(",", "").replace(".", "").split(" "), sentence.cognate_list)
  results.update({"difficulty": sentence.difficulty})
  results.update({"num_cognates": len(sentence.cognate_list)})
  #acceptable = results['num_cognates'] > 0 and results['difficulty'] > 0.2 and results['avg_gap'] < 5.0 and results['biggest_gap'] < 8.0
  acceptable = results['num_cognates'] > 0
  results.update({"acceptable": acceptable})
  print(results)

  return render_template('index.html', sentence=sentence.get_highlighted(), lang_codes=lang_codes, allow_other_languages=allow_other_languages, difficulty=difficulty)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
