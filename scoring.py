# Given a sentence and a list of words, 
# calculate the largest consecutive window of words that are NOT in word_set and return the length of the window
# E.g. "I am a student named Victor and ice cream is my favorite food", {"am", "student", "victor", "ice", "food"} -> 4
# Explanation: The largest window is "a" "named" "and" "is" which has length 4

def sliding_window_helper(sentence, word_set):
  end_of_window = lambda word: word.lower() in word_set
  window_sizes = dict()

  curr_count = 0
  max_count = 0

  for i in range(0, len(sentence)):
    if end_of_window(sentence[i]):
      if curr_count > 0:
        if curr_count in window_sizes:
          window_sizes[curr_count] += 1
        else:
          window_sizes[curr_count] = 1
      max_count = max(max_count, curr_count)
      curr_count = 0

    if not sentence[i][0].isupper():
        # If the word is uppercase, don't include it in the sliding window
        # But also, it shouldn't be a gap-stopper. Basically, just ignore it 
        # Unless its a cognate, then it should count positively toward the scoring function
        curr_count += 1
  if curr_count > 0:
    if curr_count in window_sizes.keys():
      window_sizes[curr_count] += 1
    else:
      window_sizes[curr_count] = 1

  max_count = max(max_count, curr_count)
  #print(window_sizes)
  #print(max_count)
  # Return the value corresponding to the largest key in the dictionary
  return window_sizes, max_count

def gap_heuristic(word_list, word_set):
  '''
  Gap heuristic function that returns the biggest gap between cognates, the number of gaps, and the average gap between cognates.
  Assumes that cognates have already been computed under word_set
  '''

  # Make sure that word_list is a list
  if type(word_list) == str:
    assert False, "word_list should be a list of words, not a string"

  window_sizes, max_count = sliding_window_helper(word_list, word_set)
  num_gaps = 0
  gap_count = 0

  for key in window_sizes.keys():
    num_gaps += window_sizes[key]
    gap_count += key * window_sizes[key]

  avg_gap = 0
  if num_gaps != 0:
    avg_gap = gap_count / num_gaps

  results = {
    #"sentence": sentence,
    #"word_set": word_set,
    "biggest_gap": max_count,
    "num_gaps": num_gaps,
    "avg_gap": round(avg_gap, 1)
  }
  return results

if __name__ == "__main__":
  '''
  sentence = "I am a student named Victor and ice cream is my favorite food".split(" ")
  word_set = {"am", "student", "victor", "ice", "food"}
  print(gap_heuristic(sentence, word_set))
  sentence = ["aa", "cc", "bb", "cc", "cc", "cc", "bb", "aa", "cc", "cc", "cc", "cc", "cc", "cc", "bb", "cc"]
  word_set = {"aa", "bb"}
  print(gap_heuristic(sentence, word_set))
  '''
  sentence = "El presidente de Argentina dijó que el país está 'en la etapa final de un largo ciclo de crecimiento económico'".split(" ")
  print(sentence)
  word_set = {'argentina', 'económico', 'presidente', 'final', 'presidente'}
  print(gap_heuristic(sentence, word_set))
