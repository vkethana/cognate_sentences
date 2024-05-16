from sentence import Sentence
from database import Database

class Generator:
  # Database of sentences
  # Constructor should take in source and target language codes, 
  # plus difficulty and database type
  def __init__(self, source, target, difficulty, database):
    self.source = source
    self.target = target
    self.difficulty = difficulty
    self.database = database

  # Spit out a sentence in the target language @ the target level of difficulty
  def get_sentence(self):
    s = self.database.get_sentence(self.source, self.target, self.difficulty)
    if s == None:
      s = Sentence("No sentence at specified difficulty level in dataset :(", 0.0, None)
    return s
