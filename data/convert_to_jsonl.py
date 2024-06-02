'''
def csv_to_json(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        conversations = []
        for row in csv_reader:
            prompt, response = row
            #prompt = prompt.replace("human-like text.", "human-like text.\n\n")
            conversation = {
                "messages": [
                    {"role": "system", "content": "You are a helpful expert in teaching French to English speakers by way of cognates, shared words between the two languages."},
                    {"role": "user", "content": prompt},
                  {"role": "assistant", "content": response}
                ]
            }
            conversations.append(conversation)

    return conversations

# Example usage
csv_file_path = 'good_bad_cognateful.csv'
output_json_path = 'output.jsonl'

conversations = csv_to_json(csv_file_path)
#print(conversations)
save_to_json(conversations, output_json_path)
'''

import pandas as pd
import json

def save_to_json(conversations, output_file):
    with open(output_file, 'w') as json_file:
        for conversation in conversations:
            json_file.write(json.dumps(conversation) + '\n')

# Open filename as Pandas DF
path = 'cognateful_sentences.csv'
df = pd.read_csv(path)

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

save_to_json(conversations, 'finetune_round_two.jsonl')
