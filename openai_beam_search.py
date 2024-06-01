from openai import OpenAI
import os
from utils import get_edit_ratio, get_aux_dict, Node, decompose_sentence, clean_word, get_synonyms, word_in_wordnet
from deep_translator import GoogleTranslator
import re
from random import choice, sample, randint

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection
#model = "gpt-3.5-turbo-instruct"
#model = "gpt-3.5-turbo"
#model = "ft:gpt-3.5-turbo-0125:personal:cognateful1:9SYtWhZp"
model = "gpt-4" # too expensive. and requires different API endpoints. But maybe worth trying in the future

# Seed words to help with cognate generation. These don't have to be used
use_seed_words = True
seed_words = [
    'symbole', 'France', 'unique', 'orchidées', 'caractérisé', 'Paris', 'précipitations',
    'salade', 'recommande', 'emblématique', 'région', 'abondante', 'suggéré', 'architecture',
    'satisfait', 'frites', 'beauté', 'culinaire', 'végétation', 'restaurant,', 'touriste', 'entier',
    'Triomphe', 'dessert', 'résister', 'repas,', 'mesure', 'climat', 'délicieux', 'steak', 'tropical',
    'maison,', 'accompagné', 'monuments', 'construite', 'recommander', 'panoramique', 'divine',
    'imprenable', 'tiramisu', 'composée', 'restaurant', 'tropicale', 'vin', 'apprécié', 'espèce',
    'mètres', 'luxuriante', 'température', 'absolument', 'admirer', 'dessert', 'expérience',
    'serveur', 'différente,', 'célèbre', 'restaurant',
    'artistique', 'bizarre', 'comédie', 'délicatesse', 'éducation', 'félicitations', 'génie', 'harmonie',
    'illusion', 'joie', 'kilogramme', 'lumière', 'musique', 'noble', 'orange', 'poésie', 'qualité',
    'réalité', 'sérieux', 'tempête', 'unique', 'village', 'wagon', 'zoo',
    'architecte', 'banane', 'chocolat', 'décor', 'électricité', 'festival', 'glace', 'hôpital',
    'île', 'journal', 'kiosque', 'livre', 'mélodie', 'naturel', 'océan', 'parfum', 'qualité',
    'résidence', 'supermarché', 'théâtre', 'université', 'volcan', 'week-end', 'zèbre'
]

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

  if (translation == None):
    print("WARNING: Translation failed for word", word)

  if (translation != None and translation[0:3] == 'to ' and len(translation) > 3):
    translation = translation[3:]

  return translation

def make_prompt_for_gpt(seed_words=None):
  pre_prompt = "You are about to receive a sentence in French. Please complete the sentence in French."

  if seed_words:
    pre_prompt += " Include at least one of the following words in your response: " + seed_words[0] + ", " + seed_words[1]+ "."

  pre_prompt += "You may include additional sentences and punctuation. Please try to generate human-like text. Do not repeat the portion of the sentence that is already given to you. The French sentence is:\n\n"
  return pre_prompt

def call_gpt(system_prompt, user_prompt):
  '''
  Call GPT for the beamsearch algorithm
  '''
  # Check if prompt contains INSERT_RANDOM_YEAR
  if "INSERT_RANDOM_YEAR" in system_prompt or "INSERT_RANDOM_YEAR" in user_prompt:
    assert False, "ERROR: Prompt was not pre-processed correctly. Double check the sentence starters array?"

  print("System prompt:", system_prompt)
  print("User prompt:", user_prompt)

  completion = client.chat.completions.create(model = model,
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_prompt},
  ],
  max_tokens=10,
  n=4,
  stop=None,
  temperature=1.3,
  top_p=0.9,
  frequency_penalty=0,
  presence_penalty=0.6
  )
  return [c.message.content for c in completion.choices]

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
  # this heuristic is a little too strict so currently not used
  '''
  if (len(words) >= 5):
    cntr = 0
    for i in range(0,5):
      if words[i] in cognates:
        cntr += 1
      else:
    if (cntr < 3):
      print("Rejecting sentence due to lack of sentence-initial cognate")
      return {"total_score": 0.0}
  '''
  ratio = len(cognates) / len(words)
  early_rejection = False

  if ratio < 0.2:
    print("Rejecting sentence due to low cognate ratio")
    early_rejection = True

  gap_analysis = gap_heuristic(words, cognates) # get word-gap analysis from the scoring.py file
  biggest_gap = gap_analysis['biggest_gap'] # this metric isnt really used right now
  # avg_gap typically ranges from 1-8. But for scoring purposes we force it to be between 0-1
  # the reason why we use min() is to prevent the score from going negative
  avg_gap_normalized = 1 - (min(gap_analysis['avg_gap'], 6) / 6)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap_normalized > 7 or biggest_gap > 4:
    print("Rejecting sentence due to large gap between cognates")
    early_rejection = True

  non_cognates = list(set(words) - cognates)
  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  if len(non_cognates) == 0:
    avg_non_cognate_length = 0
  else:
    avg_non_cognate_length = round(sum([len(w) for w in non_cognates]) / len(non_cognates), 2)

  if early_rejection:
    total_score = 0.0
  else:
    total_score = round(max(0.60 * ratio + 0.40 * avg_gap_normalized, 0.0), 2)
    total_score -= 0.05 * avg_non_cognate_length
  breakdown = {
    "cognate_ratio": round(ratio, 2),
    "avg_gap_between_consecutive_cognates": round(gap_analysis['avg_gap'], 2),
    "avg_gap_normalized": round(avg_gap_normalized, 2),
    "biggest_gap": biggest_gap,
    "avg_non_cognate_length": avg_non_cognate_length,
    "was_sentence_rejected_early": early_rejection,
    "total_score": round(total_score, 2)
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

  if use_seed_words:
    random_sample = sample(seed_words, 2)
    print("Using seed words: ", random_sample)
    prompt = make_prompt_for_gpt(random_sample)
    response = call_gpt(prompt, currNode.sentence)
  else:
    prompt = make_prompt_for_gpt()
    response = call_gpt(prompt, currNode.sentence)

  while min([len(choice) for choice in response]) < 2:
    print("WARNING: Empty response detected\n" * 3)
    response = call_gpt(prompt, currNode.sentence)

  choices = []
  i = 0

  for choice in response:
      text = choice.strip().replace("\n", " ")
      text = text.replace("_", "")

      print(f"[][][]      Choice {i+1}: {text}")
      # Truncate the text to the last space
      # This prevents the model from outputting a half-finished word, 
      # which would then get split in half awkwardly during the next iteration
      last_space_index = text.rfind(' ')
      if last_space_index != -1:
          text = text[:last_space_index]

      print("   Original text:", currNode.sentence)
      print("   Newly-added text:", text)
      print("   Length of Newly-added text:", len(text))

      # We run cognate analysis on just the new part of the sentence, so that we don't
      # have to check the same thing twice
      cognates = get_cognates(decompose_sentence(text))
      cognates.update(currNode.cognates)

      # at this point, we reattach the earlier part of the sentence
      text = currNode.sentence + " " + text

      newNode = Node(text, cognates, get_score_breakdown(decompose_sentence(text), cognates), prompt + currNode.sentence)

      # Check if either of the seed words are in the text
      if use_seed_words:
        if random_sample[0] not in text and random_sample[1] not in text:
          print("Failed to include seed words. Rejecting sentence.")
          continue

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
    "Le",
    "L'",
    "Les",
    "De",
    "Au",
    "Après",
    "Son",
    "Par",
    "De plus",
    "La",
    "La",
    "En",
    "En",
    "En",
    "En_INSERT_RANDOM_YEAR", # see below for explanation (ctrl+f this file for other instances of this string)
    "En_INSERT_RANDOM_YEAR",
    "En_INSERT_RANDOM_YEAR",
    "En_INSERT_RANDOM_YEAR",
    "En_INSERT_RANDOM_YEAR",
    "En_INSERT_RANDOM_YEAR",
    "En septembre",
    "En octobre",
    "En novembre",
    "En décembre",
    "En janvier",
    "En février",
    "En mars",
    "En avril",
    "En mai",
    "En juin",
    "Créée par",
    "Considérée comme",
    "Cette",
    "Avec",
    "Pour",
    "Une",
    "Si",
    "Un",
    "L'actuelle",
    "Une",
    "Je"
    "Vous",
    "Tu",
    "Elle",
    "Nous",
    "Ils",
    "Elles"
  ]
  #sentence_starters = ["el presidente de Argentina", "en el país de México", "la ciudad de Nueva York", "barcelona es"]
  # if you want to test beam search with a different language, make sure you change target_lang = 'es'
  file_path = "data/" + src_lang + "_to_" + target_lang + "_beam_search_round_2_with_seed_word_" + model + ".csv"
  i = 0
  on_good_streak = False
  score_total = 0
  num_sentences_processed = 0

  while num_sentences_processed < 50:
    if (on_good_streak and i < 5):
      candidates = run_beam_search(candidates, 3)
      i += 1
    else:
      sentence_starter = choice(sentence_starters)
      if sentence_starter == "En_INSERT_RANDOM_YEAR":
        sentence_starter = "En " + str(randint(1700, 2050))
      candidates = init_beam_search(sentence_starter, 3)
      i = 0

    on_good_streak = False
    print("\033[1m" + f"Final candidates after iteration {i + 1} of beam search, are:" + "\033[0m")
    for c in candidates:
      if (c.sentence == ""):
        print("WARNING: Empty sentence detected. Skipping.")
        continue
      print("\033[92m", c.sentence, "   [", c.cognates, "]   ",  c.score_breakdown, "   ", c.prompt, "\033[0m.")
      print("-" * 50)
      score_total += c.score_breakdown["total_score"]
      num_sentences_processed += 1
      print("Score_Total = ", score_total)
      print("num_processed = ", num_sentences_processed)
      print("Average score for this model = ", score_total / num_sentences_processed)

      if c.score_breakdown["total_score"] >= 0.35:
        try:
          on_good_streak = True
          print('\033[94m' + "Potential training data sample indicated! Want to print out: " + '\033[0m')
          message = c.sentence
          for j in c.score_breakdown.keys():
            message += "\t" + str(c.score_breakdown[j])
          message += "\t" + str(c.cognates)
          message += "\t" + str(c.prompt)
          message += "\n"
          print(message)
          with open(file_path, "a") as f:
            f.write(message)
        except:
          print("Error printing out data or writing to file")
