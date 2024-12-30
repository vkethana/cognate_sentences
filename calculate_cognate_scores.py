import os
import json
from collections import defaultdict

def calculate_cognate_word_scores(data_dir='data/'):
    # Initialize dictionaries to track total scores and occurrences of each cognate word
    cognate_scores = defaultdict(int)
    cognate_counts = defaultdict(int)

    # Iterate through all JSON files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                story_data = json.load(f)
                for sentence_data in story_data.get('sentences', []):
                    score = sentence_data.get('score', 0)
                    cognate_words = sentence_data.get('cognate_words', [])
                    for word in cognate_words:
                        cognate_scores[word] += score
                        cognate_counts[word] += 1

    # Compute the average score for each cognate word, ignoring those that appear less than twice
    cognate_averages = {
        word: cognate_scores[word] / cognate_counts[word]
        for word in cognate_scores
        if cognate_counts[word] >= 2
    }

    return cognate_averages

def save_cognate_averages(cognate_averages, output_file='cognate_averages.json'):
    # Save the cognate averages to a JSON file
    with open(output_file, 'w') as f:
        json.dump(cognate_averages, f, indent=4)
    print(f"Cognate averages saved to {output_file}")

if __name__ == "__main__":
    # Directory containing the JSON story files
    data_directory = 'data/'
    # Calculate the averages
    averages = calculate_cognate_word_scores(data_directory)
    # Save the results to a JSON file
    save_cognate_averages(averages)

