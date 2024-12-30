import logging
import datetime
import random
import json
from sentence_generator import generate_sentence_no_context, generate_next_sentence

num_stories = 100
num_sentences_per_story = 20
lang_code = 'fr'

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create log directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger filename includes date
today = datetime.datetime.now()
today_str = today.strftime('%Y-%m-%d')
log_filename = f'{log_dir}/pregenerate_site_data_{today_str}.log'
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
    first_sentence_options = generate_sentence_no_context(lang_code)
    first_sentence = random.choice(first_sentence_options)["sentence"]
    logger.info(f"Generated first sentence: {first_sentence_options}")
    logger.info(f"Selected first sentence: {first_sentence}")
    sentence_list.append(first_sentence)

    # Generate additional sentences
    for __ in range(num_sentences_per_story - 1):
        next_sentence_options = generate_next_sentence(lang_code, sentence_list)
        next_sentence = random.choice(next_sentence_options)["sentence"]
        logger.info(f"Generated additional sentences: {next_sentence_options}")
        logger.info(f"Selected additional sentence: {next_sentence}")
        sentence_list.append(next_sentence)

    # Create story dictionary
    story_dict = {
        'language': lang_code,
        'sentences': sentence_list
    }

    # Include current date in filename
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    story_filename = f'{data_dir}/story_{lang_code}_{today_str}.json'

    with open(story_filename, 'w') as f:
        json.dump(story_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved story to {story_filename}")

