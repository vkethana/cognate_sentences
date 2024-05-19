# Generate new auxiliary dictionary

# Open file french.txt, which contains one French word per line

from deep_translator import GoogleTranslator
import json
words = {}

src_lang = 'fr'
filename = "production_data/" + src_lang + ".txt"

translate_wrapper = lambda x: GoogleTranslator(source=src_lang, target='en').translate(x)
i = 0

with open(filename, "r") as f:
  for line in f:
    word = line.strip().replace(" ","")
    words[word] = translate_wrapper(word)
    print(f"Finished translating {i} words")
    i += 1
    if (i == 10000):
      break

print("Loaded auxiliary dictionary " + filename + " with", len(words), "entries.")
print(words)

# Dump dictionary into JSON
with open("production_data/" + src_lang + "_en_dict2.json", "w") as f:
    json.dump(words, f)
