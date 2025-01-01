import os
import json
from collections import defaultdict

def calculate_cognate_word_scores(data_dir):
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
                    # Updated to use actual_score instead of score
                    score = sentence_data.get('actual_score', 0)
                    # Updated to use actual_cognate_words instead of cognate_words
                    cognate_words = sentence_data.get('actual_cognate_words', [])
                    
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

def print_sorted_cognate_averages(cognate_averages):
    # Sort the words by their average score in descending order and print them
    sorted_cognates = sorted(cognate_averages.items(), key=lambda x: x[1], reverse=True)
    print("\nCognate Words Sorted by Average Score:")
    for word, avg_score in sorted_cognates:
        print(f"{word}: {avg_score:.2f}")

if __name__ == "__main__":
    # Directory containing the JSON story files
    data_directory = 'data/'
    
    # Calculate the averages
    averages = calculate_cognate_word_scores(data_directory)
    
    # Save the results to a JSON file
    save_cognate_averages(averages)
    
    # Print the sorted averages to the terminal
    print_sorted_cognate_averages(averages)
