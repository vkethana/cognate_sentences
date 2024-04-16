import re # need regex for sentence highlighting

class Sentence:
  # Note: cognate_list must be a list of strings, ALL in lowercase
  def __init__(self, sentence, difficulty, cognate_percentage, cognate_list):
    self.sentence = sentence
    self.cognate_percentage = cognate_percentage
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
      highlighted_sentence = self.sentence
      for word in self.cognate_list:
          # Shamelessly taken from ChatGPT lol
          # Create a regex pattern with the word, but make it case-insensitive
          pattern = re.compile(re.escape(word), re.IGNORECASE)

          # Find all case-insensitive matches of the word in the original sentence
          matches = pattern.findall(highlighted_sentence )

          # Iterate over matches and replace them with the original case in the original sentence
          for match in matches:
              original_case_match = re.search(re.escape(match), highlighted_sentence)
              highlighted_sentence = highlighted_sentence.replace(original_case_match.group(), f'<span class="highlight">{original_case_match.group()}</span>')
      return highlighted_sentence

