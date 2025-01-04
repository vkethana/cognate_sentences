import logging
import datetime
import random
import json
import os
from sentence_generator import *

num_stories = 20
num_sentences_per_story = 10
lang_code = 'fr'
num_choices = 3  # Number of choices per generation step

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create log directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger filename includes date
timestamp = int(datetime.datetime.now().timestamp())
log_filename = f'{log_dir}/pregenerate_site_data_{timestamp}.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Second logging handler for stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Generate stories
for _ in range(num_stories):
    sentence_list = []

    # Generate the first sentence
    first_sentence_options = [opt["sentence"] for opt in generate_sentence_no_context(lang_code)]
    logger.info(f"Generated first sentence options: {first_sentence_options}")

    # Score each first sentence individually
    first_sentence_scores = [
        gpt_scored_rubric_individual(sentence) for sentence in first_sentence_options
    ]
    logger.info(f"First sentence scores: {first_sentence_scores}")

    # Select the best first sentence
    max_score = max(score['score'] for score in first_sentence_scores)
    best_first_sentence = random.choice(
        [score for score in first_sentence_scores if score['score'] == max_score]
    )
    logger.info(f"Selected first sentence: {best_first_sentence}")
    sentence_list.append(best_first_sentence)

    # Generate additional sentences
    for __ in range(num_sentences_per_story - 1):
        logger.info(f"Currently on iteration: {__}")
        
        # Generate three candidate sentences
        next_sentence_options = generate_next_sentence(lang_code, [s['sentence'] for s in sentence_list])
        logger.info(f"Generated next sentence options: {next_sentence_options}")

        # Extract the sentences from the options
        candidate_sentences = [opt["sentence"] for opt in next_sentence_options]

        # Score each candidate sentence individually
        next_sentence_scores = [
            gpt_scored_rubric_individual(sentence) for sentence in candidate_sentences
        ]
        logger.info(f"Next sentence scores: {next_sentence_scores}")

        # Select the best sentence based on the highest score
        max_score = max(score['score'] for score in next_sentence_scores)
        best_next_sentence = random.choice(
            [score for score in next_sentence_scores if score['score'] == max_score]
        )
        logger.info(f"Selected next sentence: {best_next_sentence}")

        # Add the best sentence to the sentence list
        sentence_list.append(best_next_sentence)

    # Create story dictionary
    story_dict = {
        'language': lang_code,
        'sentences': [
            {
                'sentence': s['sentence'],
                'score': s['score'],
                'cognate_words': s['cognate_words'],
                'reasoning': s['reasoning']
            }
            for s in sentence_list
        ]
    }

    # Include current date in filename
    data_dir = 'data_hi_variance_fair_scoring'
    os.makedirs(data_dir, exist_ok=True)

    # Get exact UNIX timestamp for the filename
    timestamp = int(datetime.datetime.now().timestamp())
    story_filename = f'{data_dir}/story_{lang_code}_{timestamp}.json'

    with open(story_filename, 'w') as f:
        json.dump(story_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved story to {story_filename}")

