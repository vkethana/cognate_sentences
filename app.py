from flask import Flask, render_template, request, redirect, url_for
from get_article import get_article
from cognate_analysis import cognate_analysis, sentence_to_word_list
import re

src_lang = 'en'
target_lang = 'fr'

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
    return render_template('index.html')


@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    sentence = get_article(src_lang, target_lang)
    #sentence = "Star Wars: Return of the Jedi, conocido en español como... El profesor puede hablar el japones y el ingles"
    print("Successfully grabbed article sentences = ", sentence)
    word_list = sentence_to_word_list(sentence, trim_small_words=True)
    cognates, non_cognates, ratio = cognate_analysis(word_list, src_lang, target_lang)

    # cognates = list(cognates) # cognates needs to be a list format -- not a set -- for Flask to read it
    print("Cognates=", cognates)
    print("Cognates being passed into as list to Flask:", list(cognates.keys()))
    print("Versus:", cognates.keys())
    print("old_sentence=", sentence)
    sentence = highlight_words_in_sentence(sentence, list(cognates.keys()))
    print("new_sentence=", sentence)
    print("non_cognates=",non_cognates)
    return render_template('index.html', sentence=sentence, word_definitions=non_cognates)

# Handle requests to /generate_sentence without POST method
@app.route('/generate_sentence', methods=['GET'])
def generate_sentence_redirect():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
