from sentence import Sentence
import json
import random

class Database:
  # If you do give a file path, it will load the data from the file
  def __init__(self, file_path):
    self.data = {}
    with open(file_path, 'r') as f:
      json_data = json.load(f)
    # Iterate thru the dictionary langauge pair
    for language_pair in json_data.keys():
      self.data[language_pair] = []
      # Iterate thru the list of sentences
      for sentence in json_data[language_pair]:
        self.data[language_pair].append(Sentence(sentence['sentence'], sentence['difficulty'], sentence['cognate_percentage'], sentence['cognate_list']))

  def get_sentence(self, src_lang, target_lang, difficulty):
    lang_code = src_lang + "-" + target_lang
    if self.data.get(lang_code) == None:
      print("ERROR: No data for language pair")
    else:
      # Get list of all sentences in the language pair at specified difficulty leve
      sentences = self.data[lang_code]
      # Filter out sentences that are not at the specified difficulty level
      sentences = [sentence for sentence in sentences if sentence.difficulty == difficulty]
      # Return a random sentence from the list
      return random.choice(sentences)

if __name__ == "__main__":
  db = Database("data/small.json")
