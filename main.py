import json
import os
import logging
from datetime import datetime

from src.Embeddings import Embedding
from src.database.Settings import Settings

def load_and_validate_json(file_path):
    if not os.path.exists(file_path):
        logging.error(f"settings.json not found at {file_path}.")
        raise FileNotFoundError(f"settings.json not found at {file_path}.")

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if 'config' in data and isinstance(data['config'], dict):
                return data
            else:
                logging.error("Invalid JSON format. 'config' key is missing or not a dictionary.")
                return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return None

def update_settings_from_json(settings, json_data):
    try:
        # Extract configuration and update settings
        config = json_data.get('config', {})
        settings.set('ollama_url', config.get('ollama_url'))
        settings.set('collection_name', config.get('collection_name'))
        settings.set('embedding_model', config.get('embedding_model'))
        settings.set('port', config.get('port'))
        settings.set('sqlite_web_port', config.get('sqlite_web_port'))
        settings.set('flask_host', config.get('flask_host'))

        # Update 'valid' to false and set 'updateAt' with the current timestamp
        json_data['valid'] = False
        json_data['updateAt'] = datetime.now().isoformat()

        # Debugging: log the JSON data before writing
        logging.info("Updating settings.json with the following data:")
        logging.info(json.dumps(json_data, indent=4))

        # Save the updated JSON back to the file
        with open('settings.json', 'w') as file:
            json.dump(json_data, file, indent=4)

        logging.info("settings.json updated successfully.")

    except Exception as e:
        logging.error(f"Error updating settings.json: {e}")
        raise

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize settings
    settings = Settings()

    # Load and validate settings from JSON file
    json_file_path = os.path.join(os.getcwd(), 'settings.json')
    try:
        json_data = load_and_validate_json(json_file_path)
    except FileNotFoundError as e:
        logging.error(e)
        return

    if json_data:
        # Update settings using the JSON data
        update_settings_from_json(settings, json_data)
        logging.info("Settings updated successfully.")
    else:
        logging.error("Failed to update settings due to invalid JSON data.")

    # Load settings
    ollama_url = settings.get('ollama_url')
    collection_name = settings.get('collection_name')
    embedding_model = settings.get('embedding_model')
    port = settings.get('port')
    sqlite_web_port = settings.get('sqlite_web_port')
    flask_host = settings.get('flask_host')

    # Initialize Embedding with settings
    emb = Embedding(ollama_url, embedding_model, collection_name)

    # Embed documents
    emb.embed("books", int(settings.get('chunk_size', '500')))

    # Query embeddings
    data = emb.query_embeddings("AMD RYZEN")
    logging.info("Query results:")
    logging.info(json.dumps({
        'ollama_url': ollama_url,
        'collection_name': collection_name,
        'embedding_model': embedding_model,
        'port': port,
        'sqlite_web_port': sqlite_web_port,
        'flask_host': flask_host,
        'query_result': data
    }, indent=4))

if __name__ == '__main__':
    main()
