import pandas as pd
import json

def save_to_json(conversations, output_file):
    with open(output_file, 'w') as json_file:
        for conversation in conversations:
            json_file.write(json.dumps(conversation) + '\n')

# Open filename as Pandas DF
filename = 'cognateful_4_EVAL'

df = pd.read_csv('raw_data_2/' + filename + '.csv')

print(df.head())

# Iterate thru each row in df
conversations = []
for index, row in df.iterrows():
    sentence = row['sentence']
    parent_sentence = row['parent_sentence']
    seed_phrase = row['seed1'] + ", " + row['seed2']
    conversation = {
        "messages": [
          {"role": "system", "content": "You are about to receive a sentence in French. Please complete the sentence in French. Include at least one of the following phrases in your response: " + seed_phrase
           },
            {"role": "user", "content": parent_sentence},
            {"role": "assistant", "content": sentence}
        ]
    }
    conversations.append(conversation)

save_to_json(conversations, 'json/' + filename + '.jsonl')
