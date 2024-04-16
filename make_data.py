from Levenshtein import distance as lev_distance
from deep_translator import GoogleTranslator
import re
import requests
import wikitextparser
from bs4 import BeautifulSoup
import wikipedia
import re
import json
import random
from sentence import Sentence
import dill

# Code to get list of blocked words
with open('secrets/blocked_words.txt', 'r') as file:
  # Words are in list like so: a,b,c,d
  # Split by comma and turn into list
  blocked_words = set(file.read().split(','))
  # Remove any whitespace
  blocked_words = {word.strip() for word in blocked_words}

def get_edit_ratio(a, b):
    # the levenshtein distance (minimum number of edit operations) between the two words.
    # lower is better, as it implies the two words are cognate
    dist = lev_distance(a, b)
    assert (len(a) != 0 and len(b) != 0), "ERROR: one of the words is of length zero"
    max_len = max(len(a), len(b))
    # print(str(a) + " and " + str(b) + " are " + str(dist)
    # + " edit operations apart and max length is " + str(max_len))
    edit_ratio = round(dist / max_len, 2)
    # print(str(a) + " and " + str(b) + " have edit ratio " + str(edit_ratio))
    return edit_ratio

def get_target_lang_translation(word, src_lang, target_lang, auxilary_dictionary = None):
    if auxilary_dictionary and word in auxilary_dictionary:
        print("Fast-translating the word", word, " because it's in the auxilary dictionary under", auxilary_dictionary[word])
        return auxilary_dictionary[word]
    else:
      # assert(variable word does not contain more than 1 word)
      # return translator.translate(word, src=target_lang, dest=src_lang).text
      translation = GoogleTranslator(source=target_lang, target=src_lang).translate(word)
      #print(word, " became ", translation, " OK?")
      #print(src_lang, target_lang)
      return translation
    #return translator.translate(word, lang_tgt=lang_tgt, lang_src=lang_src)

def get_cognate(a, src_lang, target_lang, auxilary_dictionary = None):
    b = get_target_lang_translation(a, src_lang, target_lang, auxilary_dictionary)
    edit_ratio = get_edit_ratio(a, b)
    #print("Does " + a + " equal " + b + "?")
    return b if edit_ratio <= 0.60 else None

def sentence_to_word_list(sentence, trim_small_words = False):
    '''
    Cleans a sentence by removing all punctuation, lowercasing all letters

    Args:
    - sentence (str): a full sentence string
    - trim_small_words (bool): whether words of length 2 and under should be excluded
    Returns:
    - list: a list of words with no punctuation or spaces in them
    '''
    sentence = re.sub(r'[^\w\s]', '', sentence) # strip all punctuation
    sentence = sentence.lower() # lowercase the sentence
    word_list = sentence.split() # split into individual words

    if trim_small_words:
        return [i for i in word_list if len(i) > 2]
    else:
        return word_list

def cognate_analysis(words, src_lang, target_lang, auxilary_dict = None):
    ''' 
    Args:
    - words (list): a list of words
    - src_lang (str): the source language of the words
    - target_lang (str): the target language of the words
    - auxilary_dict (dict): a dictionary of words that have already been translated
    Returns:
    - dict: a dictionary of cognates
    - dict: a dictionary of non-cognates
    - float: the ratio of cognates to total words
    '''

    total = len(words)
    score = 0
    # Initialize empty dicts
    cognates_with_translation = dict()
    non_cognates_with_translation = dict()
    # Iterate thru all words in sentence
    for word in words:
        # check if the word is a cognate
        cognate = get_cognate(word, src_lang, target_lang, auxilary_dict)
        if cognate:
            # add to dictionary
            #cognate = cognate.replace(' ', '').lower()
            # print out
            #print("Cognate detected:", word, "=", cognate)
            cognates_with_translation[word] = cognate
            score += 1
        else:
            # otherwise add to list of english words
            english_translation = get_target_lang_translation(word, src_lang, target_lang, auxilary_dict).lower()
            non_cognates_with_translation[word] = english_translation
    # return the 2 dicts, and score ratio
    return cognates_with_translation, non_cognates_with_translation, score/total

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
    cleaned_text = re.sub(r'[\[\(].*?[\]\)]', '', cleaned_text)

    return cleaned_text

def get_article(src_lang, target_lang, use_vikidia=True):
    if use_vikidia:
      article_text = get_vikidia(target_lang).split()
      article_text = " ".join(article_text)
    else:
      article_text = get_webster(src_lang, target_lang)
      while (len(article_text.split()) < 6):
          article_text = get_webster(src_lang, target_lang)

    return clean_article(article_text)

def read_news_data(src_lang, target_lang):
  lang_code = src_lang + '-' + target_lang
  filename = 'data/europarl-' + lang_code + '.txt'
  line_limit = 250 # only read a preset number of lines bc the file is HUGE

  # open the file in read mode
  with open(filename, 'r') as file:
    lines = [file.readline().strip() for _ in range(line_limit)]

  # remove items that have blocked words
  lines = [line for line in lines if not any(word in line for word in blocked_words)]
  return lines

if __name__ == "__main__":
  #print(get_article('en', 'es'))
  src_lang = 'en'
  target_lang = 'es'

  lines = read_news_data(src_lang, target_lang)
  final_dataset = {}
  final_dataset[src_lang + '-' + target_lang] = []
  for i in range(0, 25):
    s = lines[i]
    cognates, non_cognates, ratio = cognate_analysis(sentence_to_word_list(s), src_lang, target_lang)
    # Remember that sentence takes in args (sentence, difficulty, cognate_percentage, cognate_list):
    print("Processing ", s)
    s = Sentence(s, 'medium', round(ratio, 2), cognates.keys())
    final_dataset[src_lang + '-' + target_lang].append(s)
    print(s)

  filename = 'data/europarl-' + src_lang + '-' + target_lang + '.pik'
  with open(filename, 'wb') as f:
    dill.dump(final_dataset, f)
