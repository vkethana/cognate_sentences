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
    self.data = datastruct

  def get_sentence(self, src_lang, target_lang, difficulty):
    lang_code = src_lang + "-" + target_lang
    if self.data.get(lang_code) == None:
      print("ERROR: No data for language pair" + lang_code)
      return ""
    else:
      # Get list of all sentences in the language pair at specified difficulty level
      sentences = self.data[lang_code]
      print("Accessing database of site" , len(sentences))
      # Filter out sentences that are not at the specified difficulty level
      print(sentences[0].difficulty, difficulty)
      sentences = [sentence for sentence in sentences if abs(float(sentence.difficulty)-difficulty) < 0.20]
      # if size of sentence array is 0
      if len(sentences) == 0:
        print("No sentence found")
        return None
      else:
        # Return a random sentence from the list
        return random.choice(sentences)

if __name__ == "__main__":
  db = Database()
