from Levenshtein import distance as lev_distance
from google_trans_new import google_translator
import re

SOURCE_LANG = 'en'
TARGET_LANG = 'fr'

translator = google_translator()
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

# 2. cognate tester
def get_english_translation(word, lang_tgt=SOURCE_LANG, lang_src=TARGET_LANG):
    # assert(variable word does not contain more than 1 word)
    # return translator.translate(word, src=target_lang, dest=src_lang).text
    return translator.translate(word, lang_tgt=lang_tgt, lang_src=lang_src)

def get_cognate(a):
    b = get_english_translation(a)
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

def cognate_analysis(words):
    # Given sentence, split it into individual words, discard non-ASCII chars and retrun list of cognates
    total = len(words)
    score = 0
    cognates_with_translation = dict()
    non_cognates_with_translation = dict()
    for word in words:
        cognate = get_cognate(word)
        if cognate:
            cognate = cognate.replace(' ', '').lower()
            print("Cognate detected:", word, "=", cognate)
            cognates_with_translation[word] = cognate
            score += 1
        else:
            english_translation = get_english_translation(word).replace(' ', '').lower()
            non_cognates_with_translation[word] = english_translation

    return cognates_with_translation, non_cognates_with_translation, score/total
