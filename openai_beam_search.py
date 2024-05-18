from openai import OpenAI
import os
from utils import sentence_to_word_list, get_edit_ratio
from deep_translator import GoogleTranslator
import re
from random import choice

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection

def get_aux_dict(filename):
  '''
  Load a txt file of words and their translations into a dictionary
  '''

  aux_dict = {}
  with open(filename, "r") as f:
    for line in f:
      word, translation = line.strip().split(" ")
      aux_dict[word] = translation
  print("Loaded auxiliary dictionary " + filename + " with", len(aux_dict), "entries.")
  return aux_dict

if os.path.exists("production_data/" + src_lang + "_" + target_lang + "_dict.txt"):
  aux_dict = get_aux_dict("production_data/" + src_lang + "_" + target_lang + "_dict.txt") # load auxiliary dictionary for fast translations
else:
  print("Failed to load auxiliary dictionary. Translations will be much slower since every word has to be google translated (!!)")

class Node:
  '''
  Wrapper class for sentences
  '''

  def __init__(self, sentence, cognates, score_breakdown):
    self.sentence = sentence
    self.cognates = cognates
    self.score_breakdown = score_breakdown
    self.score = score_breakdown["total_score"]

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

  pre_prompt = "You are about to receive a sentence in some foreign language. Please complete the sentence in that language as coherently as possible. You may include additional sentences afterward. Please try to generate human-like text."
  response = client.completions.create(model="gpt-3.5-turbo-instruct",
  prompt=pre_prompt + prompt,
  max_tokens=20,
  n=4,
  stop=None,
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

  assert type(sentence) == str, "ERROR: sentence should be a string"

  words = sentence_to_word_list(sentence, False)
  words = [re.sub(r'[^\w\s]','', i) for i in words]
  return words

def get_cognates(words):
  '''
  Given a list of words, return a set of cognates. A "cognate" is defined as < 40% edit distance between the word and its translation (might make the rule stricter later)
  '''

  cognates = set()
  for w in words:
    translation = get_target_lang_translation(w, src_lang, target_lang)
    if (src_lang =='fr' and w[:2].lower() == "l'"): # remove "l'" from the beginning of the word (for french)
      w = w[2:]

    # note that words which begin with an uppercase letter are automatically considered cognates for now
    if get_edit_ratio(w, translation) < 0.35:
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
    - The gap heuristic is weighted average of all the factors
  '''

  if (len(words) == 0):
    assert False, "ERROR: Scoring an empty sentence"

  if (type(words) == str):
    assert False, "ERROR: words should be a list of words, not a string"

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
  avg_gap_normalized = 1 - (min(gap_analysis['avg_gap'], 6) / 6)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap_normalized > 7 or biggest_gap > 10:
    return {"total_score": 0.0}

  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  breakdown = {
    "cognate_ratio": round(ratio, 2),
    "avg_gap_between_consecutive_cognates": round(gap_analysis['avg_gap'], 2),
    "avg_gap_normalized": round(avg_gap_normalized, 2),
    "total_score": round(max(0.60 * ratio + 0.40 * avg_gap_normalized, 0.0), 2)
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
      print(f"Choice {i+1}:")
      text = choice.text.strip().replace("\n", " ")
      # Truncate the text to the last space
      # This prevents the model from outputting a half-finished word, 
      # which would then get split in half awkwardly during the next iteration
      last_space_index = text.rfind(' ')
      if last_space_index != -1:
          text = text[:last_space_index]

      print("   Original text:", currNode.sentence)
      print("   Newly-added text:", text)

      # We run cognate analysis on just the new part of the sentence, so that we don't
      # have to check the same thing twice
      cognates = get_cognates(decompose_sentence(text))
      cognates.update(currNode.cognates)

      text = currNode.sentence + " " + text
      newNode = Node(text, cognates, get_score_breakdown(decompose_sentence(text), cognates))

      # if text does not contain any lowercase letters a-z, then we reject it
      # this is important because sometimes the model outputs incoherent text in ALLCAPS or only numbers
      if not re.search("[a-z]", text):
        continue
      else:
        choices.append(newNode)
  return choices

def init_beam_search(first_sentence, beam_size):
  '''
  Given a starting sentence (the root node of the beam search tree), generate three "candidate" sentences to start the beam search
  '''

  print("Starting sentence:", first_sentence)
  # run first iteration of for loop manually (this gets the beam search going by generating the first node of the tree)
  first_cognates = get_cognates(decompose_sentence(first_sentence))
  first_node = Node(first_sentence, first_cognates, get_score_breakdown(decompose_sentence(first_sentence), first_cognates))
  candidates = get_candidates_from_node(first_node)
  candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
  candidates = candidates[0:beam_size]
  return candidates

def run_beam_search(candidates, beam_size):
  '''
  Run the beam search algorithm for one iteration
  Assumes that a list which contains beam_size nodes ("candidates") has already been generated
  '''

  new_candidates = []
  for c in candidates:
    new_candidates.extend(get_candidates_from_node(c))
  new_candidates = sorted(new_candidates, key = lambda x: x.score, reverse=True)
  candidates = new_candidates[0:beam_size]
  return candidates

def sliding_window_helper(sentence, word_set):
  '''
  Given a sentence and a list of words,
  calculate the largest consecutive window of words that are NOT in word_set and return the length of the window
  E.g. "I am a student named Victor and ice cream is my favorite food", {"am", "student", "victor", "ice", "food"} -> 4
  Explanation: The largest window is "a" "named" "and" "is" which has length 4
  '''

  if (type(sentence) == str):
    assert (False)

  end_of_window = lambda word: word.lower() in word_set
  window_sizes = dict()

  curr_count = 0
  max_count = 0

  for i in range(0, len(sentence)):
    if end_of_window(sentence[i]):
      if curr_count > 0:
        if curr_count in window_sizes:
          window_sizes[curr_count] += 1
        else:
          window_sizes[curr_count] = 1
      max_count = max(max_count, curr_count)
      curr_count = 0

    if not sentence[i][0].isupper():
        # If the word is uppercase, don't include it in the sliding window
        # But also, it shouldn't be a gap-stopper. Basically, just ignore it 
        # Unless its a cognate, then it should count positively toward the scoring function
        curr_count += 1
  if curr_count > 0:
    if curr_count in window_sizes.keys():
      window_sizes[curr_count] += 1
    else:
      window_sizes[curr_count] = 1

  max_count = max(max_count, curr_count)
  #print(window_sizes)
  #print(max_count)
  # Return the value corresponding to the largest key in the dictionary
  return window_sizes, max_count

def gap_heuristic(word_list, word_set):
  '''
  Gap heuristic function that returns the biggest gap between cognates, the number of gaps, and the average gap between cognates.
  Assumes that cognates have already been computed under word_set
  '''

  # Make sure that word_list is a list
  if type(word_list) == str:
    assert False, "word_list should be a list of words, not a string"

  window_sizes, max_count = sliding_window_helper(word_list, word_set)
  num_gaps = 0
  gap_count = 0

  for key in window_sizes.keys():
    num_gaps += window_sizes[key]
    gap_count += key * window_sizes[key]

  avg_gap = 0
  if num_gaps != 0:
    avg_gap = gap_count / num_gaps

  results = {
    #"sentence": sentence,
    #"word_set": word_set,
    "biggest_gap": max_count,
    "num_gaps": num_gaps,
    "avg_gap": round(avg_gap, 1)
  }
  return results

if __name__ == "__main__":
  '''
  sentence = "I am a student named Victor and ice cream is my favorite food".split(" ")
  word_set = {"am", "student", "victor", "ice", "food"}
  print(gap_heuristic(sentence, word_set))
  sentence = ["aa", "cc", "bb", "cc", "cc", "cc", "bb", "aa", "cc", "cc", "cc", "cc", "cc", "cc", "bb", "cc"]
  word_set = {"aa", "bb"}
  print(gap_heuristic(sentence, word_set))
  '''
  sentence = "El presidente de Argentina dijó que el país está 'en la etapa final de un largo ciclo de crecimiento económico'".split(" ")
  print(sentence)
  word_set = {'argentina', 'económico', 'presidente', 'final', 'presidente'}
  print(gap_heuristic(sentence, word_set))
if __name__ == "__main__":
  # Generate a starting sentence for GPT-3.5 to complete
  sentence_starters = ["le président George Bush", "la ville de New York", "la ville de San Francisco", "le gouvernement américain", "le premier ministre Justin Trudeau"]
  #sentence_starters = ["el presidente de Argentina", "en el país de México", "la ciudad de Nueva York", "barcelona es"]
  # if you want to test beam search with a different language, make sure you change target_lang = 'es'

  first_sentence = choice(sentence_starters)
  candidates = init_beam_search(first_sentence, 3)

  for _ in range(5):
    candidates = run_beam_search(candidates, 3)
    print("\033[1m" + f"Final candidates after iteration {_ + 1} of beam search, are:" + "'\033[0m")
    for c in candidates:
      print("\033[92m", c.sentence, "   [", c.cognates, "]   ",  c.score_breakdown, "\033[0m.")
      print("-" * 50)
