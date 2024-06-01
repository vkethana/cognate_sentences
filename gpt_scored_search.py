'''
Use GPT-3.5 or GPT-4 to score sentences instead of a scoring function.
'''
from openai import OpenAI
import os
from utils import get_edit_ratio, get_aux_dict, Node, decompose_sentence, clean_word, get_synonyms, word_in_wordnet
from deep_translator import GoogleTranslator
import re
from random import choice, sample, randint

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
src_lang = 'fr'    # Language that the model will generate in
target_lang = 'en' # Language that we will translate to for cognate detection
model = "gpt-3.5-turbo-instruct"
#model = "gpt-4" # too expensive. and requires different API endpoints. But maybe worth trying in the future

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

  #print("Prompt: ", pre_prompt + prompt)
  response = client.completions.create(model=model,
  prompt=prompt,
  max_tokens=20,
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
  prompt = "Which of these sentences would be hardest for an English speaker who doesn't know any French to understand?\n"
  for sentence in choices:
    prompt += sentence + "\n"
  prompt += "\nPlease output a number between 1 and " + str(len(choices) + 1) + ". Output nothing else. If one of the sentences is blank, output -1."

  completion = client.chat.completions.create(model = 'gpt-4',
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'user', 'content': prompt},
  ],
  temperature = 1.0
  )
  response_text = completion.choices[0].message.content
  #print("ChatGPT Response:", response_text)

  # TODO: Use a regex for this
  response_text = response_text.replace("\n", "")
  response_text = response_text.replace("\t", "")
  response_text = response_text.replace("\r", "")
  response_text = response_text.replace(" ", "")

  acceptable_responses = [str(i) for i in range(1, len(choices) + 1)]
  print("Acceptable responses: ", acceptable_responses)
  if response_text in acceptable_responses:
    return int(response_text)
  else:
    print("ERROR: GPT returned an unexpected response:[" + response_text+']')
    return -1

def get_sentence_starter():
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

  sentence_starter = choice(sentence_starters)
  if sentence_starter == "En_INSERT_RANDOM_YEAR":
    sentence_starter = "En " + str(randint(1700, 2050))

  return sentence_starter

if __name__ == "__main__":
  file_path = "data/" + src_lang + "-" + target_lang + "gpt_scorer_results.csv"
  sentence_starter = get_sentence_starter()

  response = call_gpt(make_prompt_for_gpt(sentence_starter), 3)
  choices = []

  sentence = sentence_starter

  while True:
    for i, choice in enumerate(response.choices):
        text = choice.text.strip().replace("\n", " ")
        # Truncate the text to the last space
        # This prevents the model from outputting a half-finished word, 
        # which would then get split in half awkwardly during the next iteration
        last_space_index = text.rfind(' ')
        if last_space_index != -1:
            text = text[:last_space_index]

        print("Choice", i, ": ", text)
        choices.append(text)

    best_choice_index = gpt_rank(choices) - 1
    sentence += choices[best_choice_index]
    print("   ", sentence)

  '''
  while True:
    if (on_good_streak and i < 5):
      candidates = run_beam_search(candidates, 3)
      i += 1
    else:
      sentence_starter = choice(sentence_starters)
      if sentence_starter == "En_INSERT_RANDOM_YEAR":
        sentence_starter = "En " + str(randint(1700, 2050))
      candidates = init_beam_search(sentence_starter, 3)
      #print("Just grabbed the candidates ", candidates)
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
    '''
