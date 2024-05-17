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
