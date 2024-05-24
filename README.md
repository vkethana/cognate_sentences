# cognate_sentences
Generate cognate sentences for language pairs like Spanish-Portuguese, English-French. Currently, the code translates from Spanish -> English.
# How the interpretability heuristic, `get_score_breakdown` works
It takes into account:
- *cognate_ratio*: The ratio of cognates to total words in a sentence. Any sentence with a low cognate ratio (< 0.20) is automatically given a score of 0.
- *avg_gap_between_consecutive_cognates*: The average length of each gap between clusters of cognates. A "gap" is defined as a stretch two or more consecutive non-cognate words. In a good sentence, the gaps are short, and ideally the largest gap is no more than 3 or 4 words:
![Example of good vs. bad sentence](good_vs_bad_sentence.png)
- *biggest_gap*: Any sentence whose largest gap is larger than 5 words is automatically given a score of 0. (If an English speaker sees 5 words in a row that they don't understand, they may give up.)
- *avg_non_cognate_length*: Sentences are penalized for having non-cognate words which are too long.
- *total_score*: A combination of the above factors

# Known bugs
If the use_seed_words setting is enabled in `openai_beam_search.py`, occaisionally GPT-3.5 will output underscores in place of the actual seed word. Also, it may make grammatical errors. For example, consider the following prompt:
```
You are about to receive a sentence in French. Please complete the sentence in that language as coherently as possible. Please include at least one of the following words in your response: abondantes., caractérisé. You may include additional sentences afterward. Please try to generate human-like text. Above all, please do not write sentences in English (loanwords OK). Avoid including random underscores in your response. The sentence is:

La
```
One incorrect output that the model might give is:
```
La forêt amazonienne est __________ par sa biodiversité abondantes. Les
```
The above sentence has two bugs: the underscores and the possibly incorrect plural in "biodiversité abondantes" (it should be abondante) Interestingly, if we replace the underscores with the provided seed word "caractérisé", the sentence makes sense. "The amazon forest is characterized by its abundants [sic] biodiversity."
