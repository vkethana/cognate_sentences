from openai import OpenAI
import os
from utils import sentence_to_word_list, get_edit_ratio
from deep_translator import GoogleTranslator
import re
from scoring import gap_heuristic

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

def get_cognates(words):
  '''
  Given a list of words, return a set of cognates. A "cognate" is defined as < 40% edit distance between the word and its translation (might make the rule stricter later)
  '''
  cognates = set()
  for w in words:
    translation = get_target_lang_translation(w, src_lang="es", target_lang="en")
    if get_edit_ratio(w, translation) < 0.25:
      cognates.add(w)
  return cognates

def get_score(sentence, existing_cognates):
  '''
  Heuristic function that returns the ratio of cognates to total words in a sentence

  There are three main factors that play into the score:
    1) The ratio of cognates to total words
    2) The average gap between cognates
    3) The biggest gap between cognates (not currently used)

  RULES:
  - If a sentence has more than 80% cognates, approve it for sure
  - If a sentence has less than 20% cognates, reject it for sure
  - Otherwise, use gap heuristic to score the sentence
  '''
  words = decompose_sentence(sentence)
  cognates = get_cognates(words)
  print("   Found cognates:", cognates)
  ratio = len(cognates) / len(words)
  print("   Ratio of cognates to total words:", ratio)

  # some simple rules to throw out obviously good or bad sentences
  if ratio > 0.8:
    return 1.0
  elif ratio < 0.2:
    return 0.0

  cognates.update(existing_cognates) # update cognate list with existing cognates for the gap heuristic, because otherwise all words before the current generation run will be considered non cognate
  gap_analysis = gap_heuristic(words, cognates) # get word-gap analysis from the scoring.py file
  print("   Gap analysis:", gap_analysis)
  biggest_gap = gap_analysis['biggest_gap'] # this metric isnt really used right now
  # avg_gap typically ranges from 1-8. But for scoring purposes we force it to be between 0-1
  # the reason why we use min() is to prevent the score from going negative
  avg_gap = 1 - (min(gap_analysis['avg_gap'], 8) / 8)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap > 7 or biggest_gap > 10:
    return 0.0

  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  return round(max(0.25 * ratio + 0.75 * avg_gap, 0.0), 2)

def decompose_sentence(sentence):
  '''
  Split a sentence into a list of words
  Ignore words of length <= 2
  Also force all words to be lowercase and remove all punctuation
  '''
  words = sentence_to_word_list(sentence, trim_small_words = True)
  words = [re.sub(r'[^\w\s]','', i.lower()) for i in words if i[0].islower()] # exclude proper nouns
  return words

def call_gpt(prompt):
  '''
  Call the GPT-3.5 turbo model for the beamsearch algorithm
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

def extend_sentence(sentence, existing_cognates):
  '''
  Given a sentence, extend it by calling the GPT model and picking the best choice
  "Best choice" is defined as one that maximizes the heuristic score
  '''
  response = call_gpt(sentence)
  choices = []
  for i, choice in enumerate(response.choices):
      print(f"Choice {i+1}:")
      text = choice.text.strip().replace("\n", "_")
      score = get_score(text, existing_cognates)
      choices.append((text, score))
      print(f"{text}, with score {score}")
      print("-" * 50)
  # Pick the choice with the largest score
  choices.sort(key=lambda x: x[1], reverse=True)
  best_choice = choices[0]
  return (sentence + " " + best_choice[0]), get_cognates(decompose_sentence(best_choice[0]))

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

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sentence = "El presidente de Argentina dijó"
aux_dict = get_aux_dict("data/es_en_dict.txt")
print("Starting sentence:", sentence)
existing_cognates = get_cognates(decompose_sentence(sentence))

for _ in range(2):
  sentence, existing_cognates = extend_sentence(sentence, existing_cognates)
  print("\033[92m" + "Extended sentence to: " + sentence + "\033[0m .")

print("Final sentence score: ", get_score(sentence, existing_cognates))
