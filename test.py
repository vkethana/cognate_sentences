import re

sentence = "En entertain endanger en dance cat aenae"
cognate_list = {"en", "cat"}

highlighted_sentence = sentence
for word in cognate_list:
  highlighted_sentence = re.sub(r'\b({})\b'.format(re.escape(word)), r'<span class="highlight">\1</span>', highlighted_sentence)
  print(highlighted_sentence)
