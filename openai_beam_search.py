from openai import OpenAI
import os
from utils import sentence_to_word_list, get_edit_ratio
from deep_translator import GoogleTranslator
import re
from scoring import gap_heuristic

class Node:
  def __init__(self, sentence, cognates, score_breakdown):
    self.sentence = sentence
    self.cognates = cognates
    self.score_breakdown = score_breakdown
    self.score = score_breakdown["total_score"]

def get_aux_dict(filename):
  '''
  Load a txt file of words and their translations into a dictionary
  '''

  aux_dict = {}
  with open(filename, "r") as f:
    for line in f:
      word, translation = line.strip().split(" ")
      aux_dict[word] = translation
  print("Loaded auxiliary dictionary with", len(aux_dict), "entries.")
  return aux_dict

def get_target_lang_translation(word, src_lang, target_lang):
  '''
  Args:
  - word (str): a word
  - src_lang (str): the word's current langauge
  - target_lang (str): the language that we want to translate the word into
  - auxilary_dictionary (dict): a dictionary of words that have already been translated.
  - I source my auxiliary dictionaries from here:
    https://github.com/facebookresearch/MUSE?tab=readme-ov-file
  '''
  if aux_dict and word in aux_dict:
      # Uncomment this line and your terminal will be flooded with translations. but useful for seeing what kind of words get fast-translated
      #print("Fast-translating the word", word, " because it's in the auxilary dictionary under", aux_dict[word])
      return aux_dict[word]
  else:
    translation = GoogleTranslator(source=src_lang, target=target_lang).translate(word)
    return translation

def call_gpt(prompt):
  '''
  call the gpt-3.5 turbo model for the beamsearch algorithm
  '''
  response = client.completions.create(model="gpt-3.5-turbo-instruct",
  prompt=prompt,
  max_tokens=20,
  n=4,
  stop=[".", "!", "?"],
  temperature=0.7,
  top_p=0.9,
  frequency_penalty=0,
  presence_penalty=0.6)
  return response

def decompose_sentence(sentence):
  '''
  Split a sentence into a list of words
  Ignore words of length <= 2
  Also force all words to be lowercase and remove all punctuation
  '''
  words = sentence_to_word_list(sentence, False)
  words = [re.sub(r'[^\w\s]','', i.lower()) for i in words]
  return words

def get_cognates(words):
  '''
  Given a list of words, return a set of cognates. A "cognate" is defined as < 40% edit distance between the word and its translation (might make the rule stricter later)
  '''
  cognates = set()
  for w in words:
    translation = get_target_lang_translation(w, src_lang="es", target_lang="en")

    # note that words which begin with an uppercase letter are automatically considered cognates for now
    if w[0].isupper() or get_edit_ratio(w, translation) < 0.25:
      cognates.add(w)
  return cognates

def get_score_breakdown(words, cognates):
  '''
  Heuristic function that returns the ratio of cognates to total words in a sentence
  (In this implementation, sentences are represented as Node objects, not raw strings)

  There are three main factors that play into the score:
    1) The ratio of cognates to total words
    2) The average gap between cognates
    3) The biggest gap between cognates (not currently used)

  RULES:
  - If a sentence has more than 80% cognates, approve it for sure
  - If a sentence has less than 20% cognates, reject it for sure
  - Otherwise, use gap heuristic to score the sentence
  '''

  ratio = len(cognates) / len(words)
  #print("   Ratio of cognates to total words:", ratio)

  # some simple rules to throw out obviously good or bad sentences
  if ratio > 0.8:
    return {"total_score": 1.0}
  elif ratio < 0.2:
    return {"total_score": 0.0}

  gap_analysis = gap_heuristic(words, cognates) # get word-gap analysis from the scoring.py file
  #print("   Gap analysis:", gap_analysis)
  biggest_gap = gap_analysis['biggest_gap'] # this metric isnt really used right now
  # avg_gap typically ranges from 1-8. But for scoring purposes we force it to be between 0-1
  # the reason why we use min() is to prevent the score from going negative
  avg_gap_normalized = 1 - (min(gap_analysis['avg_gap'], 8) / 8)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap_normalized > 7 or biggest_gap > 10:
    return {"total_score": 0.0}

  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  breakdown = {
    "cognate_ratio": round(ratio, 2),
    "avg_gap_between_consecutive_cognates": round(gap_analysis['avg_gap'], 2),
    "avg_gap_normalized": round(avg_gap_normalized, 2),
    "total_score": round(max(0.25 * ratio + 0.75 * avg_gap_normalized, 0.0), 2)
  }
  return breakdown

def get_candidates_from_node(currNode):
  '''
  Given a sentence starter, finish the sentence using GPT-3.5 completion
  This method should output more than one option for the completion
  The results should be Node objects, which contain the sentence, cognates,
  and score (includes existing sentence)

  Then later, we use beam search to trim down the result.
  (We pick the sentences which have the highest heuristic score)

  Node that the Node object requires a sentence, a set of cognates, and a score
  '''

  response = call_gpt(currNode.sentence)
  choices = []
  for i, choice in enumerate(response.choices):
      #print(f"Choice {i+1}:")
      text = choice.text.strip().replace("\n", " ")

      # We run cognate analysis on just the new part of the sentence, so that we don't
      # have to check the same thing twice
      cognates = get_cognates(decompose_sentence(text))
      cognates.update(currNode.cognates)

      text = currNode.sentence + " " + text
      newNode = Node(text, cognates, get_score_breakdown(decompose_sentence(text), cognates))
      choices.append(newNode)
      #print(f"{newNode.sentence}, with score {newNode.score}")
      #print("-" * 50)
  # Pick the choice with the largest score
  choices.sort(key=lambda x: x.score, reverse=True)
  return choices

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
aux_dict = get_aux_dict("data/es_en_dict.txt")
beam_size = 3

first_sentence = "El presidente de Argentina dijó"
print("Starting sentence:", first_sentence)
first_cognates = get_cognates(decompose_sentence(first_sentence))
first_node = Node(first_sentence, first_cognates, get_score_breakdown(first_sentence, first_cognates))

# Run first iteration of for loop manually
candidates = get_candidates_from_node(first_node)
candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
candidates = candidates[0:beam_size]

for _ in range(3):
  new_candidates = []
  for c in candidates:
    print("considering new candidate...")
    new_candidates.extend(get_candidates_from_node(c))
    print("done considering new candidate.")
  new_candidates = sorted(new_candidates, key = lambda x: x.score, reverse=True)
  candidates = new_candidates[0:beam_size]
  print(f"Final candidates after iteration {_ + 1} of beam search, are:")
  for c in candidates:
    print("\033[92m", c.sentence, "   [", c.cognates, "]   ",  c.score_breakdown, "\033[0m.")
  print("-" * 50)
