from openai import OpenAI
import os
import re
from random import choice, sample, randint
from utils import get_edit_ratio, get_aux_dict, Node, decompose_sentence, clean_word, get_synonyms, word_in_wordnet, get_highlighted
from deep_translator import GoogleTranslator

SENTENCE_SCORING_MODEL = 'gpt-4-turbo'
#SENTENCE_GENERATION_MODEL = 'gpt-3.5-turbo-instruct'
SENTENCE_GENERATION_MODEL = "gpt-4-turbo"
INCORRECT_MORPHEME_DETECTION_MODEL = 'gpt-4-turbo'

src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
use_seed_words = True
use_gpt_scoring = False

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
  print("Auxiliary dictionary loaded successfully")
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

def identify_cognates(words):
  '''
  Given a list of words, return all the words that are cognate. A "cognate" is defined as < 40% edit distance between the word and its translation (might make the rule stricter later)

  TODO: Use GPT-4 judgements to determine whether a word is cognate instead of edit distance comparison
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

def get_sentence_starter(temperature=1.3):
  '''
  When we first init beam search, we need some sentence starters to get the search process started
  TODO: Figure out a way to generate these programmatically
  TODO: Figure out what kind of sentence starters are more likely to lead to cognateful sentences
  There is room for improvement
  '''
  pre_prompt = "You are a helpful AI assistant that speaks both French and English fluently."
  random_sample = sample(seed_words, 2)
  print("Using seedwords ", random_sample)
  prompt = (
    "Please write one (1) sentence in French. "
    #"Please adhere to the following constraints, which will guide you through the generation process:\n\n"
    #"- In your sentence, try to include many cognate words, words that an English speaker can easily identify the meaning of.\n"
    #"- You are encouraged, but not required, to include proper nouns that an English speaker may recognize.\n"
    #"- Please include at least one of the two seed words in your response: " + random_sample[0] + ", " + random_sample[1] + ". Please include the actual word instead of substituting it with underscores. Do not include underscores in your response."
    #"Please include at least one of the two seed words in your response: " + random_sample[0] + ", " + random_sample[1] + ". Please include the actual word instead of substituting it with underscores. Do not include underscores in your response."
   )
  #print("Prompt: ", pre_prompt + "\n" + prompt)


  completion = client.chat.completions.create(model = "gpt-4-turbo",
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'system', 'content': pre_prompt},
    {'role': 'user', 'content': prompt},
  ],
  temperature = temperature,
  max_tokens = 100
  )

  # Extract the actual output
  response_text = completion.choices[0].message.content

  response_text = response_text.replace("\n", "")
  response_text = response_text.replace("\t", "")
  response_text = response_text.replace("\r", "")
  response_text = response_text.replace("  ", "")

  return response_text

def trim_candidates(candidates, beam_size=3):
  '''
  Input: A large list of candidate nodes
  Output: A smaller list of candidate nodes, trimmed down to the beam size

  Run the beam search algorithm for one iteration
  Assumes that a list which contains beam_size nodes ("candidates") has already been generated
  '''

  assert(type(candidates) == list)
  # assert that each eleement of the list is a Node object
  assert(all([type(c) == Node for c in candidates]))

  new_candidates = []
  for c in candidates:
    new_candidates.extend(get_candidates_from_node(c))
  new_candidates = sorted(new_candidates, key = lambda x: x.score, reverse=True)

  candidates = new_candidates[0:beam_size]
  return candidates

def gpt_extend_sentence(sentence, num_choices=6):
    '''
    Ask ChatGPT to output num_choices different options for continuing a sentence
    '''

    system_prompt = "You are a fluent speaker of both French and English. You are about to receive some text in French. Your task is to output one (1) additional French sentence that continues where the provided sentence leaves off. Please keep in mind the following constraints:\n\n- Try to use cognate words, words that an English speaker can easily identify the meaning of.\n- When possible, use proper nouns that an English speaker would be able to recognize.\n- Don't include the existing sentence(s) in your response. Your output should consist of just one (1) French sentence.\n"

    if use_seed_words:
      random_sample = sample(seed_words, 2)
      system_prompt += "- Please include at least one of the following seed words in your response: " + random_sample[0] + ", " + random_sample[1] + ". Please include the actual word instead of substituting it with underscores. "

    sentence = sentence.replace("  ", " ")
    assert SENTENCE_GENERATION_MODEL not in ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct"], "ERROR: GPT-3.5 does not support the completions endpoint"
    possible_extensions = client.chat.completions.create(
      model = SENTENCE_GENERATION_MODEL,
      messages = [ # Change the prompt parameter to the messages parameter
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': sentence},
      ],
      max_tokens=200,
      n=num_choices,
      stop=None,
      temperature=1.4,
      top_p=0.9,
      frequency_penalty=0,
      presence_penalty=0.6
    )

    print(possible_extensions)
    possible_extensions = [i.message.content for i in possible_extensions.choices]
    possible_extensions = [i.strip().replace("\n", " ").replace("_", "").replace("  ", "") for i in possible_extensions]

    '''
    # This code block is used to remove partial words from the end of the sentence,
    # because sometimes the GPT-4 generation halts mid-word

    for text in possible_extensions:
        last_space_index = text.rfind(' ')
        if last_space_index != -1:
            text = text[:last_space_index]
        choices.append(text)
    '''

    #print(f"Sentence {sentence} got the following choices from GPT: {choices}")
    return possible_extensions
# 
def get_candidates_from_node(currNode):
  '''
  Given a sentence starter, finish the sentence using GPT-3.5 or GPT-4 completion
  This method should output more than one option for the completion
  The results should be Node objects, which contain the sentence, cognates,
  and score (includes existing sentence)

  Then later, we use beam search to trim down the result.
  (We pick the sentences which have the highest heuristic score)

  Node that the Node object requires a sentence, a set of cognates, and a score
  '''

  possible_extensions = gpt_extend_sentence(currNode.sentence)
  #print("choices=", choices)

  if min([len(possible_extensions) for choice in possible_extensions]) < 2:
    print("WARNING: Empty response detected\n" * 3)

  i = 0
  choices = []

  for text in possible_extensions:
      assert(type(text) == str)

      text = text.replace(currNode.sentence, "") # remove the old part of the sentence temporarily
      cognates = identify_cognates(decompose_sentence(text)) # we don't want to run cognate analysis on the old part of the sentence bc we've already done that before
      cognates.update(currNode.cognates)

      text = currNode.sentence + " " + text # at this point, we reattach the earlier part of the sentence

      newNode = Node(text, cognates, get_score_breakdown(currNode.sentence, decompose_sentence(text), cognates))
      newNode.parent_sentence = currNode.sentence

      # Check if either of the seed words are in the text
      if use_seed_words:
        if random_sample[0] not in text and random_sample[1] not in text:
          print("Failed to include seed words. Rejecting sentence.")
          continue
        else:
          newNode.seed_words = random_sample

      # if text does not contain any lowercase letters a-z, then we reject it
      # this is important because sometimes the model outputs incoherent text in ALLCAPS or only numbers
      if not re.search("[a-z]", text):
        continue
      else:
        choices.append(newNode)

  if len(choices) == 0:
    print("WARNING: No choices were generated\n" * 3)

  return choices

def gpt_rank(choices):
  '''
  Given a set of n choice sentences, ask GPT 3.5 or 4 to rank them
  '''

  pre_prompt = "You are about to receive a set of " + str(len(choices)) + " sentences in French. Please identify which sentence would be easiest to understand for an English speaker who doesn't know any French.\n"
  prompt = ""

  for i, sentence in enumerate(choices):
    prompt += str(i + 1) + ". " + str(sentence) + "\n"

  pre_prompt += "Please output a number between 1 and " + str(len(choices)) + ". Output nothing else."

  completion = client.chat.completions.create(model = "gpt-3.5-turbo",
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'system', 'content': pre_prompt},
    {'role': 'user', 'content': prompt},
  ],
  temperature = 1.0
  )

  print("ASKING GPT the following prompt: ")
  print("-" * 25)
  print("PROMPT: ", pre_prompt + "\n" + prompt)
  print("-" * 25)

  # Extract the actual output
  response_text = completion.choices[0].message.content

  response_text = response_text.replace("\n", "")
  response_text = response_text.replace("\t", "")
  response_text = response_text.replace("\r", "")
  response_text = response_text.replace(" ", "")

  acceptable_responses = [str(i) for i in range(1, len(choices) + 1)]
  #print("Acceptable responses: ", acceptable_responses)
  if response_text in acceptable_responses:
    return int(response_text)
  else:
    print("ERROR: GPT returned an unexpected response:[" + response_text+']')
    return -1

def evaluate_translation(original_sentence, translated_sentence, lang="French"):
  '''
  Given an original sentence and a translated sentence, evaluate the quality of the translation.
  '''

  prompt_1 = f'I will give you a sentence in {lang} (#1) and a sentence in English (#2). If the English sentence is a correct translation of the {lang} sentence, output "1". Otherwise, output "0". Your task is to assess whether the English sentence is a correct translation; you do not need to assess whether the sentences make sense.'
  
  prompt_2 = (
    'Here are two examples:\n'
    "1. J'ai regardé le livre et \n"
    '2. I looked at the book \n'
    'Reasoning: The English sentence correctly translates the French sentence. Though both sentences are incomplete, the English mirrors the French accurately in the part provided. \n'
    'Final Answer: 1\n\n'
    '1. Je suis triste et\n'
    '2. I am happy\n'
    'Reasoning: The English sentence does not correctly translate the French sentence. "Je suis triste" means "I am sad." Therefore, "I am happy" is not a correct translation. Also, the word "et" is left untranslated. \n'
    'Final Answer: 0'
  )

  system_prompt = f"{prompt_1}\n{prompt_2}"
  user_prompt = f"1. {original_sentence}\n2. {translated_sentence}\nReasoning:"

  print(f"ASKING {SENTENCE_SCORING_MODEL} the following prompt: {system_prompt}\n{user_prompt}")

  completion = client.chat.completions.create(
      model=SENTENCE_SCORING_MODEL,
      messages=[
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_prompt}
      ],
      temperature=0.8
  )

  # Extract the reasoning and final answer from the response
  response_text = completion.choices[0].message.content.strip()
  print("Got a response from chatgpt!", response_text)

  # Separate reasoning and final answer
  try:
      reasoning_part = response_text.split("Final Answer:")[0].strip()
      final_answer_part = response_text.split("Final Answer:")[1].strip()
  except IndexError:
      print(f"ERROR: GPT returned an unexpected response: [{response_text}]")
      return -1

  print(f"Reasoning: {reasoning_part}")
  print(f"Final Answer: {final_answer_part}")

  # Clean and parse the final answer
  final_answer_part = final_answer_part.replace("\n", "").replace("\t", "").replace("\r", "").replace(" ", "")

  if final_answer_part in ["0", "1"]:
      return int(final_answer_part)
  else:
      print(f"ERROR: GPT returned an unexpected response: [{final_answer_part}]")
      return -1

def gpt_scored_rubric(sentence):
    '''
    Given an atbirary French sentence, let GPT-4 score it based on a rubric that assigns points between 0 and 2
    '''
    # TODO: Add third example, "Lors du repas, la famille royale a été accompagnée de musiciens venus de différentes régions pour divertir"
    # This would be helpful bc it would help the model reason about what distinguishes a score of 2 from a score of 3
    # Both of the current examples dont add much because they are quite clear cut
    system_prompt = (
      'You are an expert in French to English translation. I will give you a sentence in French and I want you to assign one of the following scores to it:\n'
      '0 (lowest score): Totally unintelligible to an English speaker\n'
      '1: Contains at least one cognate word. But largely unintelligible to an English speaker\n'
      '2: Contains many cognate words. An English speaker could partially understand the sentence but they may miss a few important words or phrases.\n'
      '3 (highest score): Nearly every word in the sentence is cognate. All non-cognate words can be guessed from context. An English speaker could guess the meaning of the sentence with ease.\n\n'
      'As an example, consider the following sentence:\n'
      '"Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."\n'
      'Reasoning: An English speaker can make out the sentence through the cognate words, and the noncognate words are small enough that they can be ignored.\n'
      'Final Score: 3\n'
      'Another example is the following sentence which would receive a score of 0:\n'
      '"Veux-tu déjeuner avec moi?"\n'
      'Reasoning: The sentence does not contain a single cognate and is totally unintelligible to a monolingual English speaker.\n'
      'Final Score: 0\n'
      'A final example is the following sentence which would receive a score of 2:\n'
      '"Le chef prépare un steak exquis pour le repas."\n'
      'Reasoning: The sentence contains several cognate words (steak, exquis), but the word "repas" is not readily understandable to an English speaker.\n'
      'Final Score: 2\n'
      'Please format your responses like the examples above.'
    )

    #print(f"ASKING {SENTENCE_SCORING_MODEL} the following prompt: {system_prompt}\n{sentence}")

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': sentence}
        ],
        temperature=0.8
    )

    # Extract the reasoning and final answer from the response
    response_text = completion.choices[0].message.content.strip()
    print("Got a response from chatgpt!", response_text)

    # Process the response to extract scores
    score = -1
    lines = response_text.split('\n')

    for line in lines:
        if line.startswith('Final Score:'):
            print("Processing line:", line)
            try:
                score = int(line.split(':')[1].strip())
            except ValueError:
                print(f"Warning: Could not parse score from line: {line}")
    return score

def gpt_scored_rubric_batch(sentences):
    '''
    Given a list of three arbitrary French sentences, let GPT-4 score them based on a rubric that assigns points between 0 and 3
    '''
    if len(sentences) != 3:
        raise ValueError("This function requires exactly three sentences.")

    system_prompt = (
        'You are an expert in French to English translation. I will give you three sentences in French and I want you to assign one of the following scores to each of them:\n'
        '0 (lowest score): Totally unintelligible to an English speaker\n'
        '1: Contains some cognate words, but is largely unintelligible to an English speaker\n'
        '2: Contains many cognate words. An English speaker could partially understand the sentence but they would probably miss a few important words or phrases.\n'
        '3 (highest score): An English speaker can reasonably guess the meaning of the sentence.\n\n'
        'As an example, consider the following set of sentences:\n'
        '1. “Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain.”\n'
        'Reasoning: An English speaker can make out the sentence through the cognate words, and the noncognate words are small enough that they can be ignored.\n'
        'Final Score: 3\n'
        '2. "Veux-tu déjeuner avec moi?"\n'
        'Reasoning: The sentence does not contain a single cognate and is totally unintelligible to a monolingual English speaker.\n'
        'Final Score: 0\n'
        '3. "Lors du repas, la famille royale a été accompagnée de musiciens venus de différentes régions pour divertir."\n'
        'Reasoning: The sentence contains several cognate words and an English speaker can understand the general idea, although some details might be missed.\n'
        'Final Score: 2\n\n'
        'Please format your responses with reasoning and final score for each sentence, like the example above.\n\n'
        'Here are the three sentences:\n'
        f'1. {sentences[0]}\n'
        f'2. {sentences[1]}\n'
        f'3. {sentences[2]}\n'
    )

    print(f"ASKING {SENTENCE_SCORING_MODEL} the following prompt: {system_prompt}")

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt}
        ],
        temperature=0.8
    )

    # Extract the reasoning and final answer from the response
    response_text = completion.choices[0].message.content.strip()
    print("Got a response from chatgpt!", response_text)

    # Process the response to extract scores
    scores = []
    lines = response_text.split('\n')
    for line in lines:
        print("Processing line:", line)
        if line.startswith('Final Score:'):
            try:
                score = int(line.split(':')[1].strip())
                scores.append(score)
            except ValueError:
                print(f"Warning: Could not parse score from line: {line}")
    print("Scores=", scores)
    return scores

def get_wrong_words(original_sentence, translated_sentence):
    '''
    Given a sentence in French and an attempted translation of that sentence in English, return a list of words that were not translated correctly.
    '''

    # Initial prompt with an example
    prompt_1 = """
    You are an expert in professional translation. I will give you a sentence in French (#1) and a sentence in English (#2). The English sentence will be an incorrect attempted translation of the French sentence. Please output, as a Python list, all the words in the French sentence which are not correctly translated. Provide your reasoning step-by-step, and then give the final answer as a Python list.

    For example:
    1. Je suis triste et
    2. I am happy

    Reasoning:
    "triste" means "sad" in English, but it is translated as "happy", which is incorrect.
    "et" means "and" in English, but it is not translated.
    Therefore, the words not correctly translated are ["triste", "et"].
    Final Answer: ["triste", "et"]
    """

    # Create the completion request
    completion = client.chat.completions.create(
        model=INCORRECT_MORPHEME_DETECTION_MODEL,  # Use the appropriate model engine
        messages=[
            {'role': 'system', 'content': prompt_1},
            {'role': 'user', 'content': f"1. {original_sentence}\n2. {translated_sentence}\nReasoning:"}
        ],
        temperature=0.8
    )

    # Extract the reasoning and final answer from the response
    response_text = completion.choices[0].message.content

    print("response received!", response_text)
    # Separate reasoning and final answer
    try:
        reasoning_part = response_text.split("Final Answer:")[0]
        final_answer_part = response_text.split("Final Answer:")[1]
    except IndexError:
        print(f"ERROR: GPT returned an unexpected response: [{response_text}]")
        return []

    print(f"Reasoning: {reasoning_part}")
    print(f"Final Answer: {final_answer_part}")

    # Parse the final answer part into a python list
    try:
        lst = eval(final_answer_part)
        return lst
    except Exception as e:
        print(f"ERROR: GPT returned an unexpected response: [{final_answer_part}]")
        print(e)
        return []

def get_score_breakdown(sentence, words, cognates):
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

  # check if any of the words have underscores in them
  if any(['_' in w for w in words]):
    early_rejection = True

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
    #print("Rejecting sentence due to low cognate ratio")
    early_rejection = True

  gap_analysis = gap_heuristic(words, cognates) # get word-gap analysis from the scoring.py file
  biggest_gap = gap_analysis['biggest_gap'] # this metric isnt really used right now
  # avg_gap typically ranges from 1-8. But for scoring purposes we force it to be between 0-1
  # the reason why we use min() is to prevent the score from going negative
  avg_gap_normalized = 1 - (min(gap_analysis['avg_gap'], 6) / 6)
  # throw out sentences with too many large gaps in between cognates
  if avg_gap_normalized > 7 or biggest_gap > 4:
    #print("Rejecting sentence due to large gap between cognates")
    early_rejection = True

  non_cognates = list(set(words) - cognates)
  # otherwise, do a weighted average: 25% based on cognate ratio, 75% based on gap heuristic
  for word in words:
    if "_" in word:
      early_rejection = True
  if len(non_cognates) == 0:
    avg_non_cognate_length = 0
  else:
    avg_non_cognate_length = round(sum([len(w) for w in non_cognates]) / len(non_cognates), 2)

  if early_rejection:
    total_score = 0.0
  else:
    total_score = round(max(0.60 * ratio + 0.40 * avg_gap_normalized, 0.0), 2)
    total_score -= 0.05 * avg_non_cognate_length


  if use_gpt_scoring:
    gpt_score_normalized = gpt_scored_rubric(sentence) / 3
    breakdown = {
      "cognate_ratio": round(ratio, 2),
      "avg_gap_between_consecutive_cognates": round(gap_analysis['avg_gap'], 2),
      "avg_gap_normalized": round(avg_gap_normalized, 2),
      "biggest_gap": biggest_gap,
      "avg_non_cognate_length": avg_non_cognate_length,
      "was_sentence_rejected_early": early_rejection,
      "gpt_rubric_score_normalized": gpt_score_normalized,
      "total_score": round(0.5 * total_score + 0.5 * gpt_score_normalized, 2)
    }
  else:
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

def make_sentence_object(sentence):
  assert(type(sentence) == str)
  cognates = identify_cognates(decompose_sentence(sentence)) # we don't want to run cognate analysis on the old part of the sentence bc we've already done that before
  node = Node(sentence, cognates, get_score_breakdown(sentence, decompose_sentence(sentence), cognates))
  return node

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

def one_step_forward(curr_node):
    '''
    Input: curr_sentence, which is a Node object
    Output: one additional sentence which is tacked on to the end of the curr_sentence
    '''

    new_candidates = get_candidates_from_node(curr_node)
    new_candidates = sorted(new_candidates, key = lambda x: x.score, reverse=True)

    beam_size = 3
    candidates = new_candidates[0:beam_size]

    best_sentence_index = gpt_rank([i.sentence for i in new_candidates]) - 1
    return new_candidates[best_sentence_index]

def get_sentence_as_json(node_object):
    '''
    Input: A Node object
    Output: A JSON object that contains the sentence, the score, and whether the sentence was pre-generated (always false in this case)
    '''
    highlighted_sentence = get_highlighted(node_object.sentence, node_object.cognates)
    print("highlighted_sentence=", highlighted_sentence)

    output = {
      'unhighlighted_sentence': node_object.sentence,
      'cognates': list(node_object.cognates),
      'score': node_object.score_breakdown['total_score'],
      "is_pregenerated": False
    }

    return output
