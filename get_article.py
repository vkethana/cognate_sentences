import requests
import wikitextparser
from bs4 import BeautifulSoup
import wikipedia
import re
import json
import random
from cognate_analysis import get_target_lang_translation

def get_wikipedia(language_code, char_cutoff=200):
    """
    Fetches a random sentence from a Wikipedia article in the specified language, limited by character count.

    Args:
    - language_code (str): The language code for the desired Wikipedia language edition. Defaults to 'es' (Spanish).
    - char_cutoff (int): The maximum number of characters to include in the random sentence. Defaults to 200.

    Returns:
    - str: A random sentence from a Wikipedia article, truncated to the specified character cutoff.
    """
    wikipedia.set_lang(language_code)

    # Get a random article title
    random_article_title = wikipedia.random(1)

    # Get the content of the article
    page = wikipedia.page(random_article_title)

    # Extract sentences from the content
    sentences = page.content.split('. ')

    # Randomly select a sentence
    random_sentence = sentences[0]
    return random_sentence

def get_vikidia(language_code):
    """
    Fetches a random article from Vikidia, the encyclopedia for children, in the specified language.

    Args:
    - language_code (str): The language code for the desired language. Defaults to 'es' (Spanish).

    Returns:
    - str or None: The text content of the random article if successful, or None if the request fails.
    """
    # Wikipedia URL for a random article in the specified language
    wiki_random_url = f"https://{language_code}.wikipedia.org/wiki/Special:Random"

    # Send a GET request to Wikipedia's random article URL
    response = requests.get(wiki_random_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the content of the article (you may need to adjust this based on Wikipedia's HTML structure)
        article_content = soup.find('div', {'id': 'mw-content-text'})

        # Extract text from the article content
        article_text = "\n".join(p.get_text() for p in article_content.find_all('p'))
        return article_text
    else:
        print(f"Failed to fetch random Wikipedia article. Status code: {response.status_code}")

def get_webster_json(word):
    # incomplete for now
    pass

def get_webster(src_lang, target_lang):
    # Read JSON data from file
    with open('data/learner.json', 'r') as file:
        json_data = file.read()

    json_data = json_data.replace('{it}', '').replace('{/it}', '')
    json_data = json_data.replace('{phrase}', '').replace('{/phrase}', '')

    # Parse the JSON data
    data = json.loads(json_data)

    # Function to recursively search for English sentences in the JSON data
    def extract_english_sentences(obj, sentences):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "t" and isinstance(value, str):
                    sentences.append(value)
                else:
                    extract_english_sentences(value, sentences)
        elif isinstance(obj, list):
            for item in obj:
                extract_english_sentences(item, sentences)

    # List to store extracted English sentences
    english_sentences = []

    # Extract English sentences
    extract_english_sentences(data, english_sentences)

    cntr = 0
    # Print the extracted English sentences
    for sentence in english_sentences:
        #print(str(cntr) + ": " + sentence)
        cntr += 1

    #print("asdfsadf")
    random_sentence = random.choice(english_sentences);
    target_lang_translation = get_target_lang_translation(random_sentence, target_lang, src_lang)
    print("Grabbed sentence = ", random_sentence, " and turned it into target language ",  target_lang_translation)
    return target_lang_translation

def clean_article(article_text):
    """
    Cleans the given article text by removing newline characters and text within brackets.

    Args:
    - article_text (str): The input text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    # Remove newline characters
    cleaned_text = article_text.replace('\n', '')

    # Remove text within brackets using regular expressions
    #cleaned_text = re.sub(r'[\[\(].*?[\]\)]', '', cleaned_text)

    return cleaned_text

def get_article(src_lang, target_lang, use_vikidia=False):
    if use_vikidia:
      article_text = get_vikidia(target_lang).split()
      article_text = " ".join(article_text)
    else:
      article_text = get_webster(src_lang, target_lang)
      while (len(article_text.split()) < 6):
          article_text = get_webster(src_lang, target_lang)

    return clean_article(article_text)
