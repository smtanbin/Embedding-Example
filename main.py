import json

from src.Embeddings import Embedding
from src.database.Settings import Settings


def main():
    # Initialize settings
    settings = Settings()

    # Check if settings need to be prompted
    if not settings.get('ollama_url') or not settings.get('collection_name') or not settings.get('embedding_model'):
        settings.prompt_for_settings()

    # Load settings
    ollama_url = settings.get('ollama_url')
    collection_name = settings.get('collection_name')
    embedding_model = settings.get('embedding_model')
    port = settings.get('port')
    sqlite_web_port = settings.get('sqlite_web_port')
    flask_host = settings.get('flask_host')

    # Check for JSON input to update settings
    json_input = input("Enter JSON for updating settings (or leave empty to skip): ")
    if json_input:
        settings.update_from_json(json_input)

    # Initialize Embedding with settings
    emb = Embedding(ollama_url, embedding_model, collection_name)

    # Embed documents
    emb.embed("books", int(settings.get('chunk_size', '500')))

    # Query embeddings
    data = emb.query_embeddings("AMD RYZEN")
    print(json.dumps({
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
