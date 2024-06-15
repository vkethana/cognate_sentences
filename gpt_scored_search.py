'''
Use GPT-3.5 or GPT-4 to score sentences instead of a scoring function.
'''
from openai import OpenAI
import os
from utils import get_edit_ratio, get_aux_dict, Node, decompose_sentence, clean_word, get_synonyms, word_in_wordnet
from deep_translator import GoogleTranslator
import re
from random import choice, sample, randint
from openai_beam_search import score_sentence

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection

# model is for the v1/completions endpoint
model = "gpt-3.5-turbo-instruct"
# model_2 is for the v1/chat/completions endpoint
#model_2 = "gpt-3.5-turbo"
model_2 = "gpt-4-turbo"

# Seed words to help with cognate generation. These don't have to be used
use_seed_words = False
seed_words = [
    'symbole', 'Gustave', 'France', 'unique', 'orchidées', 'caractérisé', 'Paris', 'précipitations',
    'salade', 'recommande', 'emblématique', 'région', 'abondante', 'suggéré', 'architecture',
    'satisfait', 'frites', 'beauté', 'culinaire', 'végétation', 'restaurant,', 'touriste', 'entier',
    'Triomphe', 'dessert', 'résister', 'repas,', 'mesure', 'climat', 'délicieux', 'steak', 'tropical',
    'maison,', 'accompagné', 'monuments', 'construite', 'recommander', 'panoramique', 'divine',
    'imprenable', 'tiramisu', 'composée', 'restaurant', 'tropicale', 'vin', 'apprécié', 'espèces',
    'mètres', 'luxuriante', 'température', 'absolument', 'admirer', 'dessert', 'Eiffel,', 'expérience',
    'serveur', 'différentes,', 'célèbres', 'Eiffel', 'restaurant',
    'artistique', 'bizarre', 'comédie', 'délicatesse', 'éducation', 'félicitations', 'génie', 'harmonie',
    'illusion', 'joie', 'kilogramme', 'lumière', 'musique', 'noble', 'orange', 'poésie', 'qualité',
    'réalité', 'sérieux', 'tempête', 'unique', 'village', 'wagon', 'xylophone', 'yoga', 'zoo',
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

def make_prompt_for_gpt(sentence_to_be_extended):
  pre_prompt = "You are about to receive a sentence in French. Please complete the sentence in French as coherently as possible."
  if use_seed_words:
    random_sample = sample(seed_words, 2)
    pre_prompt += " Please include at least one of the following words in your response: " + random_sample[0] + ", " + random_sample[1] + ". Please include the actual word instead of substituting it with underscores. "
    print("Using seed words: ", random_sample)
  pre_prompt += "You may include additional sentences afterward. Please try to generate human-like text.\n\n"
  return pre_prompt + sentence_to_be_extended

def call_gpt(prompt, num_choices=4):
  '''
  Call GPT for the beamsearch algorithm
  '''
  # Check if prompt contains INSERT_RANDOM_YEAR
  if "INSERT_RANDOM_YEAR" in prompt:
    assert False, "ERROR: Prompt was not pre-processed correctly. Double check the sentence starters array?"

  # this API call requires gpt-3.5-turbo (doesn't work with 4)
  # TODO: Refactor this to use the new completions API

  response = client.completions.create(model = model,
  prompt=prompt,
  max_tokens=8,
  n=num_choices,
  stop=None,
  temperature=1.3,
  top_p=0.9,
  frequency_penalty=0,
  presence_penalty=0.6)
  return response

def gpt_rank(choices):
  '''
  Given a set of n choice sentences, ask GPT 3.5 or 4 to rank them
  '''

  pre_prompt = "You are about to receive a set of " + str(len(choices)) + " sentences in French. Please identify which sentence would be easiest to understand for an English speaker who doesn't know any French.\n"
  prompt = ""

  for i, sentence in enumerate(choices):
    prompt += str(i + 1) + ". " + str(sentence) + "\n"

  pre_prompt += "Please output a number between 1 and " + str(len(choices)) + ". Output nothing else."

  completion = client.chat.completions.create(model = model_2,
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'system', 'content': pre_prompt},
    {'role': 'user', 'content': prompt}
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

  full_prompt = f"{prompt_1}\n{prompt_2}\n\n1. {original_sentence}\n2. {translated_sentence}\nReasoning:"

  print(f"ASKING {model_2} the following prompt: {full_prompt}")

  completion = client.chat.completions.create(
      model=model_2,
      messages=[
          {'role': 'system', 'content': full_prompt},
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
        model=model_2,  # Use the appropriate model engine
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

def get_sentence_starter():
  sentence_starters = [
    "Le", "Le", "L'", "Les", "De", "Au", "Après", "Son", "Par", "De plus", "La", "La", "En", "En", "En",
    "En_INSERT_RANDOM_YEAR", # see below for explanation (ctrl+f this file for other instances of this string)
    "En_INSERT_RANDOM_YEAR", "En_INSERT_RANDOM_YEAR", "En_INSERT_RANDOM_YEAR", "En_INSERT_RANDOM_YEAR", "En_INSERT_RANDOM_YEAR", "En septembre", "En octobre", "En novembre", "En décembre", "En janvier", "En février", "En mars", "En avril", "En mai", "En juin", "Créée par", "Considérée comme", "Cette", "Avec", "Pour", "Une", "Si", "Un", "L'actuelle", "Une", "Je", "Vous", "Tu", "Elle", "Nous", "Ils", "Elles" ]

  sentence_starter = choice(sentence_starters)
  if sentence_starter == "En_INSERT_RANDOM_YEAR":
    sentence_starter = "En " + str(randint(1700, 2050))

  return sentence_starter

def gpt_extend_sentence(sentence):
    sentence = sentence.replace("  ", " ")
    possible_extensions = call_gpt(make_prompt_for_gpt(sentence), num_choices=4).choices
    possible_extensions = [sentence + " " + i.text.strip().replace("\n", " ").replace("_", "").replace("  ", "") for i in possible_extensions]

    choices = []
    for text in possible_extensions:
        last_space_index = text.rfind(' ')
        if last_space_index != -1:
            text = text[:last_space_index]
        choices.append(text)

    best_choice_index = gpt_rank(choices) - 1
    #print("Choosing option ", best_choice_index + 1, " with text: ", choices[best_choice_index])
    return choices[best_choice_index]

def gpt_generate_new_sentence():
    sentence = get_sentence_starter()
    return gpt_extend_sentence(sentence)

def run_iteration(sentence):
    print("Starting new iteration with starter " + sentence + "...")
    print("-" * 50)
    curr_score = 1.0
    num_good_sentences = 0

    while len(sentence.split(" ")) < 35 and curr_score > 0.0:
      # no sentences with >35 words
      sentence = sentence.replace("  ", " ")
      possible_extensions = call_gpt(make_prompt_for_gpt(sentence), num_choices=4).choices
      possible_extensions = [sentence + " " + i.text.strip().replace("\n", " ").replace("_", "").replace("  ", "") for i in possible_extensions]

      choices = []
      for text in possible_extensions:
          last_space_index = text.rfind(' ')
          if last_space_index != -1:
              text = text[:last_space_index]
          choices.append(text)

      #print("(" + str(i) + ") Just generated the following choices: ")
      #print(choices, "\n\n")
      best_choice_index = gpt_rank(choices) - 1
      print("Choosing option ", best_choice_index + 1, " with text: ", choices[best_choice_index])

      # We're not tacking on elements to sentence, we just overrwrite it
      old_sentence = sentence # store this in case we need to print out

      sentence = choices[best_choice_index]
      curr_score = score_sentence(sentence)
      print("Sentence = ", sentence)
      print("Current score = ", curr_score)

      if curr_score > 0.25:
        num_good_sentences += 1

        print("\033[92m" + "Good sentence number " + str(num_good_sentences))

        message = ""
        message += sentence + "\t"
        message += str(curr_score) + "\t"
        message += old_sentence + "\t"
        for i in choices:
          message += i + "\t"

        print(message)
        print("\033[0m")

      print('---')
      print()
    return num_good_sentences

if __name__ == "__main__":
  file_path = "data/" + src_lang + "-" + target_lang + "gpt_scorer_results.csv"

  while True:
    sentence = get_sentence_starter()
    print('Got ', run_iteration(sentence), ' good sentences')
