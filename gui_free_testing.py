import re
from openai_beam_search import run_beam_search, init_beam_search, get_cognates, get_score_breakdown
from utils import decompose_sentence
from random import choice

'''
Test the openai_beam_search module without needing to open up the GUI
'''

lst = [
  "Netflix est une plateforme de streaming vidéo qui offre une large sélection de films, séries télévisées et documentaires",
  "Wikipédia est une encyclopédie et un projet d'encyclopédie collaboratif édité par des bénévoles",
  "L'intellectuel français Voltaire a dit: 'La tolérance est un ingrédient essentiel de la civilisation'",
  "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américaine",
  "Je veux visiter la tour Eiffel parce qu'elle est un symbole important de Paris . J'admire sa beauté architecturale et je veux voir la vue",
  "Une fois arrivé en France, j'ai immédiatement été impressionné par la beauté",
  "De retour à la maison, j'ai préparé un délicieux dîner pour ma famille avec des ingrédients frais du marché.  J'ai utilisé des herbes."
]
#print(sentence_to_word_list(lst[-1]))
for sentence in lst:
  words = decompose_sentence(sentence)
  cognates = get_cognates(words)
  non_cognates = set(words) - set(cognates)
  print("Parsing sentence ", words)
  print("Found cognates ", cognates)
  print("Found non-cognates ", non_cognates)
  print("Running score breakdown")
  print(get_score_breakdown(words, cognates))
  print("\n\n")
