import re
from Levenshtein import distance as lev_distance

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

def get_edit_ratio(a, b):
    # the levenshtein distance (minimum number of edit operations) between the two words.
    # lower is better, as it implies the two words are cognate
    dist = lev_distance(a, b)

    if (min(len(a), len(b)) <= 5):
      # Must be a near-perfect match if less than or equal to 5 chars
      if dist <= 1:
        return 0.0
      else:
        return 1.0

    assert (len(a) != 0 and len(b) != 0), "ERROR: one of the words is of length zero"
    avg_len = (len(a) + len(b)) / 2
    edit_ratio = round(dist / avg_len, 2)

    return edit_ratio
