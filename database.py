from sentence import Sentence
import json
import random
import dill

class Database:
  default_path = "data/europarl-en-es.pik"
  # If you do give a file path, it will load the data from the file
  def __init__(self, file_path=default_path):
    self.data = {}
    with open(file_path, 'rb') as in_strm:
      datastruct = dill.load(in_strm)
    print(datastruct)
    self.data = datastruct

  def get_sentence(self, src_lang, target_lang, difficulty):
    lang_code = src_lang + "-" + target_lang
    if self.data.get(lang_code) == None:
      print("ERROR: No data for language pair" + lang_code)
    else:
      # Get list of all sentences in the language pair at specified difficulty leve
      sentences = self.data[lang_code]
      # Filter out sentences that are not at the specified difficulty level
      sentences = [sentence for sentence in sentences if sentence.difficulty == difficulty]
      # Return a random sentence from the list
      return random.choice(sentences)

if __name__ == "__main__":
  db = Database()
