import re
import unicodedata
from Levenshtein import distance as lev_distance
import json
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet') # Need this for synonym checking

def clean_word(word):
   word = re.sub(r'[^\w\s]', '', word).lower() # strip all punctuation and lowercase the word
   return ''.join(c for c in unicodedata.normalize('NFD', word)
                  if unicodedata.category(c) != 'Mn')

def get_edit_ratio(a, b):
    # the levenshtein distance (minimum number of edit operations) between the two words.
    # lower is better, as it implies the two words are cognate
    a = clean_word(a)
    b = clean_word(b)
    #assert (len(a) != 0 and len(b) != 0), "ERROR: one of the words is of length zero"
    if (len(a) == 0 or len(b) == 0):
      print("WARNING: One of the words is of length zero")
      print("The words are ", a, " and ", b)

    dist = lev_distance(a, b)
    if (min(len(a), len(b)) <= 2):
      return 1.0 # no two-letter words should be cognates

    if (min(len(a), len(b)) <= 4):
      # Must be a near-perfect match if less than or equal to 4 chars
      if dist > 1: # if edit distance is more than one
        return 1.0
      else:
        return 0.0

    avg_len = (len(a) + len(b)) / 2
    edit_ratio = round(dist / avg_len, 2)

    return edit_ratio

def get_aux_dict(file_path):
  '''
  Load a txt file of words and their translations into a dictionary
  Used to speed up the get_target_lang_translation function
  Helpful but not strictly necessary for the beam search algorithm to work
  '''

  aux_dict = {}
  # Open JSON file and read the contents
  with open(file_path, 'r') as file:
    # Step 3: Load the JSON data as a dictionary
    aux_dict = json.load(file)
  assert(len(aux_dict) > 0), "ERROR: auxiliary dictionary should not be empty"
  print("Loaded auxiliary dictionary " + file_path + " with", len(aux_dict), "entries.")
  return aux_dict

class Node:
  '''
  Wrapper class for sentences that includes their score breakdown and cognate list.
  Makes process of passing around these 3 variables less clunky
  '''

  def __init__(self, sentence, cognates, score_breakdown, prompt_that_generated_me = None, seed_words = None, parent_sentence = None):
    assert(type(sentence) == str), "ERROR: sentence should be a string. What was passed in is: " + str(sentence) + 'of type ' + str(type(sentence))
    self.sentence = sentence
    self.cognates = cognates
    self.score_breakdown = score_breakdown
    self.score = score_breakdown["total_score"]
    self.seed_words = seed_words
    self.parent_sentence = parent_sentence

    if prompt_that_generated_me != None:
      self.prompt = prompt_that_generated_me.replace("\n", "  ")

def decompose_sentence(sentence):
  '''
  Split a sentence into a list of words
  '''
  assert type(sentence) == str, "ERROR: sentence should be a string. What was passed in is: " + str(sentence) + 'of type ' + str(type(sentence))
  words = [i for i in sentence.split() if i.lower().islower()] # we check if every string, lowercased, contains at least one lowercase letter
  # this will remove words that are just punctuation or numbers
  return words

# Function to get synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)[:10]  # Return top 10 synonyms


def word_in_wordnet(word):
  # Check if there are any synsets for the given word
  synsets = wordnet.synsets(word)
  if synsets:
      return True
  else:
      return False
