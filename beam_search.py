from transformers import pipeline
#from make_data import get_ratio_from_sentence

def get_score(samples, **kwargs):
  return [10 * get_ratio_from_sentence(sentence) for sentence in samples]

# Load a pre-trained language model
#generator = pipeline("text2text-generation", model="flax-community/spanish-t5-small", num_beams=4, max_length=50)
#generator = pipeline("text-generation", model="bigscience/bloom")
#generator = pipeline("text-generation", model="DeepESP/gpt2-spanish")

# Partially generated sentence
generator = pipeline("text2text-generation", model="gpt2")  # or another variant
#sentence = "El presidente de Argentina dijó que "
sentence = "The president of Argentina said that "
#outputs = generator(sentence, num_return_sequences=4, return_full_text=False, max_new_tokens=10)
#outputs = generator(sentence)
outputs = generator(sentence, num_beams=4, num_return_sequences=4, max_length=50, early_stopping=True)

for output in outputs:
  print(output)
  #print(f"Sentence: {full_sentence}, Score: {get_ratio_from_sentence(full_sentence, src_lang='es', target_lang='en')}")
  #print(f"Sentence: {full_sentence}, Score: {0}")
'''
# Function to perform beam search for next token prediction
def beam_search_next_token(sentence, beam_size):
  candidates = [(sentence, 1.0)]  # List of (sentence, probability) pairs
  for _ in range(1):  # Change the loop for desired sequence length
    new_candidates = []
    for (prev_sentence, prev_prob) in candidates:
      # Generate next token probabilities
      generated_tokens = generator(prev_sentence, max_length=1, num_return_sequences=beam_size)
      for token in generated_tokens:
        print("Token:")
        print(token)
        print('-' * 50)
        next_sentence = prev_sentence + token['generated_text']
        next_prob = prev_prob * token['logits'][0]
        new_candidates.append((next_sentence, next_prob))
    # Sort and select top beam_size candidates
    sorted_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    candidates = sorted_candidates
  return candidates

# Perform beam search
results = beam_search_next_token(sentence, beam_size)

# Print results
for (sentence, probability) in results:
  print(f"Sentence: {sentence}, Probability: {probability}")
'''
