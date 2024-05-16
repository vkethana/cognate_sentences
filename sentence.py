import re # need regex for sentence highlighting

class Sentence:
  # Note: cognate_list must be a list of strings, ALL in lowercase
  def __init__(self, sentence, difficulty, cognate_list):
    self.sentence = sentence
    self.difficulty = difficulty
    self.cognate_list = cognate_list

  def __str__(self):
    # Return dictionary representation of all atrributes
    return str(self.__dict__)

  def __repr__(self):
    # Return dictionary representation of all atrributes
    return str(self.__dict__)

  # return formatted version of sentence with the relevant words highlighted
  def get_highlighted(self):
      if (self.sentence == None or self.sentence == ""):
        return ""
      highlighted_sentence = self.sentence
      if (self.cognate_list == None or len(self.cognate_list) == 0):
        return highlighted_sentence
      for word in self.cognate_list:
          highlighted_sentence = re.sub(r'\b({})\b'.format(re.escape(word)), r'<span class="highlight">\1</span>', highlighted_sentence)
      return highlighted_sentence

