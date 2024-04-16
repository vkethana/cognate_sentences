from sentence import Sentence

class Database:
  def __init__(self):
    self.data = {
    "en-es": [
        Sentence("Yo soy Vijay y me gusta jugar al golf", "easy", 0.5, ["vijay", "golf"]),
        Sentence("Soy de los estados unidos", "easy", 0.2, ['estados', 'unidos']),
        Sentence("Entender esta oración no es muy difícil", "easy", 0.1, ["dificil"]),
      ]
    }

  def get_sentence(self, src_lang, target_lang, difficulty):
    lang_code = src_lang + "-" + target_lang
    if self.data.get(lang_code) == None:
      print("ERROR: No data for language pair")
    else:
      for item in self.data.get(lang_code):
        if item.difficulty == difficulty:
          return item
      print("ERROR: No sentence at difficulty level")
