class Generator:
  # Constructor should take in source and target language codes, plus difficulty
  def __init__(self, source, target, difficulty):
    self.source = source
    self.target = target
    self.difficulty = difficulty

  # Spit out a sentence in the target language @ the target level of difficulty
  def get_sentence(self):
    # Placeholder for now
    return "This is a placeholder sentence."

  # Print out Set<String> of cognates identified in the sentence (from the target language)
  def get_cognates(self):
    return set('the')
