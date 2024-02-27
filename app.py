from flask import Flask, render_template, request, redirect, url_for, jsonify
from get_article import get_article
from cognate_analysis import cognate_analysis, sentence_to_word_list
import re

src_lang = 'en'
target_lang = 'es'
language_codes = {
    'English': 'en',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'French': 'fr',
    'Italian': 'it',
    'Dutch': 'nl',
    'German': 'de',
}

with open('data/auxilary_dict_es_en.py', 'r') as f:
  es_en_dict = eval(f.read())

app = Flask(__name__, static_folder="templates/static")

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


@app.route('/')
def index():
    return render_template('index.html', language_codes=language_codes, src_lang=src_lang, target_lang=target_lang)

# Route to update source_lang and target_lang variables
@app.route('/update_lang', methods=['POST'])
def update_lang():
    global src_lang, target_lang  # Use global keyword to access and modify global variables

    # Retrieve selected language values from the request
    src_lang = request.form.get('src_lang')
    target_lang = request.form.get('target_lang')

    #assert((src_lang in language_codes.keys()) or (src_lang in language_codes.values()))
    #assert((target_lang in language_codes.keys()) or (target_lang in language_codes.values()))

    print("Source and Target Language are: ", src_lang, target_lang)
    #src_lang = language_codes[src_lang]
    #target_lang = language_codes[target_lang]

    # Optionally, you can perform validation or additional processing here

    # Return a response indicating success
    return jsonify({'status': 'success', 'src_lang': src_lang, 'target_lang': target_lang})


@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    auxilary_dict = None
    if (src_lang, target_lang) == ('en', 'es'):
      auxilary_dict = es_en_dict
    sentence = get_article(src_lang, target_lang, use_vikidia=False)
    #sentence = "Star Wars: Return of the Jedi, conocido en español como... El profesor puede hablar el japones y el ingles"
    #print("Successfully grabbed article sentences = ", sentence)
    word_list = sentence_to_word_list(sentence, trim_small_words=True)
    if auxilary_dict:
      cognates, non_cognates, ratio = cognate_analysis(word_list, src_lang, target_lang, auxilary_dict)
    else:
      cognates, non_cognates, ratio = cognate_analysis(word_list, src_lang, target_lang)
    print("Ratio=", ratio)

    sentence = highlight_words_in_sentence(sentence, list(cognates.keys()))

    cognates_highlighted = cognates.copy()

    for key, value in cognates.items():
      cognates_highlighted[key] = f'<span class="highlight">{value}</span>'

    print("cognates highlighted: ", cognates_highlighted)

    all_word_definitions = non_cognates

    try:
      all_word_definitions = cognates_highlighted | non_cognates
    except:
      print("Could not merge dictionaries together")
    print("Word_Defs_List:", all_word_definitions)

    # cognates = list(cognates) # cognates needs to be a list format -- not a set -- for Flask to read it
    #print("Cognates=", cognates)
    #print("Cognates being passed into as list to Flask:", list(cognates.keys()))
    #print("Versus:", cognates.keys())
    #print("old_sentence=", sentence)
    #print("new_sentence=", sentence)
    #print("non_cognates=",non_cognates)
    return render_template('index.html', sentence=sentence, word_definitions=all_word_definitions, language_codes=language_codes, src_lang=src_lang, target_lang=target_lang)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
