from openai import OpenAI
import os
from utils import get_edit_ratio, get_aux_dict, Node, decompose_sentence, clean_word, get_synonyms, word_in_wordnet
from deep_translator import GoogleTranslator
import re
from random import choice, sample

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection

# Seed words to help with cognate generation. These don't have to be used
use_seed_words = True
seed_words = ['symboles', 'Gustave', 'France.', 'unique.', 'orchidées', 'caractérisé', 'Paris', 'précipitations', 'salade', 'recommande', 'emblématiques', 'région', 'abondantes.', 'suggéré', 'architecture', 'satisfait', 'frites', 'beauté', 'culinaire', 'végétation', 'restaurant,', 'touristes', 'entier', 'Triomphe', 'dessert.', 'résister', 'repas,', 'mesure', 'climat', 'délicieux', 'steak', 'tropical', 'maison,', 'accompagné', 'monuments', 'construite', 'recommander', 'panoramique', 'divine.', 'imprenable', 'tiramisu', 'composée', 'restaurant', 'tropicales', 'vin', 'apprécié', 'espèces', 'mètres', 'luxuriante', 'températures', 'absolument', 'admirer', 'dessert', 'Eiffel,', 'expérience', 'serveur', 'différentes,', 'est', 'célèbres', "n'ai", 'Eiffel', 'restaurant.']

# load the auxiliary dictionary
aux_dict = None
if os.path.exists("production_data/" + src_lang + "_" + target_lang + "_dict.json"):
  aux_dict = get_aux_dict("production_data/" + src_lang + "_" + target_lang + "_dict.json") # load auxiliary dictionary for fast translations
else:
  print("Failed to load auxiliary dictionary. Translations will be much slower since every word has to be google translated (!!)")

def get_target_lang_translation(word, src_lang, target_lang):
  '''
  Args:
  - word (str): a word
  - src_lang (str): the word's current langauge
  - target_lang (str): the language that we want to translate the word into
  - auxilary_dictionary (dict): a dictionary of words that have already been translated.
  '''

  if aux_dict and word in aux_dict:
      # Uncomment this line and your terminal will be flooded with translations. but useful for seeing what kind of words get fast-translated
      #print("Fast-translating the word", word, " because it's in the auxilary dictionary under", aux_dict[word])
      translation = aux_dict[word]
  else:
    translation = GoogleTranslator(source=src_lang, target=target_lang).translate(word)

  if (translation[0:3] == 'to ' and len(translation) > 3):
    translation = translation[3:]

  return translation

def call_gpt(prompt):
  '''
  call the gpt-3.5 turbo model for the beamsearch algorithm
  '''

  pre_prompt = "You are about to receive a sentence in French. Please complete the sentence in that language as coherently as possible."
  if use_seed_words:
    random_sample = sample(seed_words, 2)
    pre_prompt += " Please include the following words in your response: " + random_sample[0] + ", " + random_sample[1] + ". "
    print("Using seed words: ", random_sample)
  pre_prompt += "You may include additional sentences afterward. Please try to generate human-like text. Above all, please do NOT include any English text in your response. \n\n"
  #print("Prompt: ", pre_prompt + prompt)
  response = client.completions.create(model="gpt-3.5-turbo-instruct",
  prompt=pre_prompt + prompt,
  max_tokens=20,
  n=4,
  stop=None,
  temperature=1.1,
  top_p=0.9,
  frequency_penalty=0,
  presence_penalty=0.6)
  return response

def get_cognates(words):
  '''
  Given a list of words, return a set of cognates. A "cognate" is defined as < 40% edit distance between the word and its translation (might make the rule stricter later)
  '''

  cognates = set()
  for w in words:
    w_cleaned = clean_word(w) # remove all punctuation and lowercase the word
    if (src_lang =='fr' and (w[:2].lower() == "l'" or w[:2].lower() == "d'")): # remove l' and d' from any words if french
      w_cleaned = clean_word(w[2:])

    translation = get_target_lang_translation(w_cleaned, src_lang, target_lang)
    synonyms = get_synonyms(translation) # these are English synonyms
    synonyms.append(translation) # add the translation itself to the set of synonyms

    '''
    The below snippet of code checks if the TL word is in the English wordnet. E.g. French "taxi" because it's in English wordnet
    This can help catch some stray cognates
    WARNING: The code is fooled by false cognates
    We check if the word length is >2 because otherwise words like "de" or "a" get flagged as cognates
    '''
    if len(w_cleaned) > 2 and word_in_wordnet(w_cleaned):
      cognates.add(w)
      #print("The french word" + w + " is in English wordnet so it must be a cognate")
      continue
    '''
    Reason why we iterate thru synonyms in addition to the English translation:
    Some words (e.g. "assure" = "ensures") have a very low edit distance with their synonyms
    But these words are clearly cognate. If we compare against the top 10 synonyms, we can catch these cases
    '''
    for s in synonyms:
      edit_ratio = get_edit_ratio(w_cleaned, s)
      if edit_ratio < 0.35:
        cognates.add(w)
        break
  return cognates

def get_score_breakdown(words, cognates):
  '''
  Heuristic function that returns the ratio of cognates to total words in a sentence
  Note that the sentence has to be passed in as a list of words, not a string

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

  # auto-reject sentence that don't have at least three cognates in the first 5 words
  if (len(words) >= 5):
    cntr = 0
    for i in range(0,5):
      if words[i] in cognates:
        cntr += 1
    if (cntr < 3):
      return {"total_score": 0.0}

  ratio = len(cognates) / len(words)

  # some simple rules to throw out obviously good or bad sentences
  if ratio > 0.8:
    return {"total_score": 1.0}
  elif ratio < 0.2:
    return {"total_score": 0.0}

  gap_analysis = gap_heuristic(words, cognates) # get word-gap analysis from the scoring.py file
  biggest_gap = gap_analysis['biggest_gap'] # this metric isnt really used right now
  # avg_gap typically ranges from 1-8. But for scoring purposes we force it to be between 0-1
  # the reason why we use min() is to prevent the score from going negative
  avg_gap_normalized = 1 - (min(gap_analysis['avg_gap'], 6) / 6)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap_normalized > 7 or biggest_gap > 4:
    return {"total_score": 0.0}

  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  breakdown = {
    "cognate_ratio": round(ratio, 2),
    "avg_gap_between_consecutive_cognates": round(gap_analysis['avg_gap'], 2),
    "avg_gap_normalized": round(avg_gap_normalized, 2),
    "biggest_gap": biggest_gap,
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
      #print(f"Choice {i+1}:")
      text = choice.text.strip().replace("\n", " ")
      # Truncate the text to the last space
      # This prevents the model from outputting a half-finished word, 
      # which would then get split in half awkwardly during the next iteration
      last_space_index = text.rfind(' ')
      if last_space_index != -1:
          text = text[:last_space_index]

      #print("   Original text:", currNode.sentence)
      #print("   Newly-added text:", text)

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

  #print("Starting sentence:", first_sentence)
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

  end_of_window = lambda word: word in word_set
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
  # Generate a starting sentence for GPT-3.5 to complete
  sentence_starters = [
    "Le",
    "L'",
    "Les",
    "De",
    "Au",
    "Par",
    "De plus",
    "La",
    "En",
    "Créée par",
    "Cette",
    "Pour",
    "Une",
    "Un climat",
    "Une précaution",
    "Je"
  ]
  #sentence_starters = ["el presidente de Argentina", "en el país de México", "la ciudad de Nueva York", "barcelona es"]
  # if you want to test beam search with a different language, make sure you change target_lang = 'es'
  file_path = "data/" + src_lang + "_to_" + target_lang + "_beam_search_results_3.csv"
  i = 0
  on_good_streak = False

  while True:
    if (on_good_streak and i < 5):
      candidates = run_beam_search(candidates, 3)
      i += 1
    else:
      candidates = init_beam_search(choice(sentence_starters), 3)
      print("Just grabbed the candidates ", candidates)
      i = 0
    on_good_streak = False
    print("\033[1m" + f"Final candidates after iteration {i + 1} of beam search, are:" + "\033[0m")
    for c in candidates:
      if (c.sentence == ""):
        print("WARNING: Empty sentence detected. Skipping.")
        continue
      print("\033[92m", c.sentence, "   [", c.cognates, "]   ",  c.score_breakdown, "\033[0m.")
      print("-" * 50)

      if c.score_breakdown["total_score"] >= 0.40:
        try:
          on_good_streak = True
          print('\033[94m' + "Potential training data sample indicated! Want to print out: " + '\033[0m')
          if (c.score_breakdown["total_score"] == 1.00):
            c.score_breakdown["cognate_ratio"] = -1
            c.score_breakdown["avg_gap_between_consecutive_cognates"] = -1
          message = c.sentence + "\t" + str(c.score_breakdown['cognate_ratio']) + "\t" + str(c.score_breakdown['avg_gap_between_consecutive_cognates']) + "\t" + str(c.score_breakdown['biggest_gap']) + "\t" + str(c.score_breakdown['total_score']) + "\n"
          print(message)
          with open(file_path, "a") as f:
            f.write(message)
        except:
          print("Error printing out data or writing to file")
