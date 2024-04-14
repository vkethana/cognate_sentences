from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from generator import Generator

app = Flask(__name__, static_folder="templates/static")

# instantiate new generator object with default attributes
# there are NO global variables for user target lang, difficulty, etc.
# the generator object will be updated with the user's choices
# so you don't need to keep track of these attributes separately
sentence_generator = Generator(source='en', target='es', difficulty='easy')

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
    difficulty = 'easy'

    # src_lang = request.cookies.get('src_lang')
    # target_lang = request.cookies.get('target_lang')
    # difficulty = request.cookies.get('difficulty') 

    return render_template('index.html', lang_codes=language_codes, src_lang=src_lang, target_lang=target_lang, difficulty=difficulty)

# Route to update source_lang and target_lang variables
@app.route('/update_lang', methods=['POST'])
def update_lang():
    # Retrieve selected lang values from the request
    src_lang = request.form.get('src_lang')
    target_lang = request.form.get('target_lang')
    #difficulty = request.form.get('difficulty')

    print("Source and Target lang are: ", src_lang, target_lang)
    # Update the sentence generator object
    #sentence_generator = Generator(source=src_lang, target=target_lang, difficulty=difficulty)

    # Return a response indicating success
    return jsonify({'status': 'success', 'src_lang': src_lang, 'target_lang': target_lang})


@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
  sentence = generator.generate_sentence()
  # return sentence and sentence.get_cognate_list()

# Helper method for highlighting words in a sentence
def highlight_words_in_sentence(sentence, words_to_highlight):
    for word in words_to_highlight:
        # Shamelessly taken from ChatGPT lol
        # Create a regex pattern with the word, but make it case-insensitive
        pattern = re.compile(re.escape(word), re.IGNORECASE)

        # Find all case-insensitive matches of the word in the original sentence
        matches = pattern.findall(sentence)

        # Iterate over matches and replace them with the original case in the original sentence
        for match in matches:
            original_case_match = re.search(re.escape(match), sentence)
            sentence = sentence.replace(original_case_match.group(), f'<span class="highlight">{original_case_match.group()}</span>')
    return sentence

if __name__ == '__main__':
    app.run(debug=True)
