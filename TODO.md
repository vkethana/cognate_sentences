1) Try prompting the model with seed words to see if better sentence generation. 
	 - Seems to be working
2) Get 100 good examples and then fine tune a model on them
	- How to get 100 good examples?
	- Use very strict criteria 
3) Try using RL
  - MIght not yield super impressive results
	- 
4) Try using gpt as a scoring function

2024-06-18
- Get rid of openai_beam_search.py and make a new file called generate_sentences_without_gui.py
- Add more logging capability (not a super urgent feature)
