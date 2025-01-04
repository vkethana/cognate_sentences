def is_valid_json(content):
    """Helper function to validate JSON output"""
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON output: {content}")
        return False

def save_sentences_batch(sentences, output_file):
    """
    Append a batch of sentences to a JSON file, creating the file if it doesn't exist.
    """
    try:
        # Read existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                'sentences': [],
                'metadata': {
                    'total_count': 0,
                    'difficulty_counts': {'0': 0, '1': 0, '2': 0, '3': 0},  # Use strings as keys
                    'last_updated': None
                }
            }
        
        # Add new sentences
        data['sentences'].extend(sentences)
        
        # Update metadata
        data['metadata']['total_count'] = len(data['sentences'])
        data['metadata']['last_updated'] = datetime.datetime.now().isoformat()
        
        # Update difficulty counts - convert score to string for JSON compatibility
        for sentence in sentences:
            score = str(sentence['actual_score'])  # Convert to string
            data['metadata']['difficulty_counts'][score] += 1
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error saving to {output_file}: {str(e)}")  # Print full error message
        # Save to backup file if main save fails
        backup_file = output_file + '.backup'
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

