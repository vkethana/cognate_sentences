from backend import one_step_forward, get_sentence_starter, make_sentence_object, get_sentence_as_json
from collections import defaultdict
import json

'''
{ "1":  { "difficulty": 0.5, "sentences": [
        {
          "raw_text": "Il était une fois un petit village.",
          "score": 0.7,
          "cognates": ["village"]
        },
        {
          "raw_text": "Il était une fois un petit village. Le village était connu pour ses beaux paysages.",
          "score": 0.8,
          "cognates": ["village", "paysages"]
        }
      ]
    },
    "2": {

    },
    "3": { }

}
'''

# Function to write the story_set to a JSON file
def write_story_set_to_json(story_set, filename="data/test_jul5_2.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(story_set, f, ensure_ascii=False, indent=4)

story_set = defaultdict(lambda: {"difficulty": 0.5, "sentences": []})
num_sentences_processed = 0
num_stories_processed = 0
i = 0
curr_node = one_step_forward(make_sentence_object(get_sentence_starter()))

while num_sentences_processed < 500:

    if ((i > 30) or (i > 10 and curr_node.score_breakdown["total_score"] < 0.3)):
        # fail conditions
        num_stories_processed += 1
        curr_node = one_step_forward(make_sentence_object(get_sentence_starter()))
        i = 0
    else:
        curr_node = one_step_forward(curr_node)
        i += 1

    story_set[num_stories_processed]["sentences"].append(get_sentence_as_json(curr_node))
    print("*" * 50)
    print("Story set: \n", story_set)
    print("Number of sentences processed: ", num_sentences_processed)
    print("Number of stories processed: ", num_stories_processed)
    print("*" * 50)
    print("\n\n\n")
    num_sentences_processed += 1

    # Write the story_set to a JSON file
    write_story_set_to_json(story_set)

