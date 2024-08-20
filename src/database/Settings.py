import sqlite3
import os
import json


class Settings:
    def __init__(self):
        # Path to the settings database
        self.db_folder = 'data'
        self.name = 'settings.db'
        self.settings_db_path = os.path.join(self.db_folder, self.name)

        # Ensure the database file exists
        if not os.path.exists(self.settings_db_path):
            self._initialize_db()

        # Load settings from the database
        self.settings = self._load_settings()

    def _initialize_db(self):
        """Create the settings database and table if they do not exist."""
        try:
            conn = sqlite3.connect(self.settings_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            # Insert default values if the table is empty
            cursor.execute('''
                INSERT OR IGNORE INTO settings (key, value) VALUES
                ('ollama_url', 'http://localhost:11434'),
                ('collection_name', 'documents'),
                ('embedding_model', 'snowflake-arctic-embed:latest'),
                ('base_model_name', ''),
                ('port', '3000'),
                ('sqlite_web_port', '9999'),
                ('flask_host', '0.0.0.0')
            ''')
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            raise Exception(f"Error initializing database: {e}")

    def _load_settings(self):
        """Load settings from the database."""
        settings = {}
        try:
            conn = sqlite3.connect(self.settings_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM settings')
            rows = cursor.fetchall()
            for row in rows:
                settings[row[0]] = row[1]
            conn.close()
        except sqlite3.Error as e:
            raise Exception(f"Error loading settings: {e}")
        return settings

    def get(self, key, default=None):
        """Retrieve a setting value by key."""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value."""
        try:
            conn = sqlite3.connect(self.settings_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            ''', (key, value))
            conn.commit()
            conn.close()
            self.settings[key] = value
        except sqlite3.Error as e:
            raise Exception(f"Error setting value: {e}")

    def prompt_for_settings(self):
        """Prompt the user to provide missing settings."""
        print("Please provide the following settings:")
        ollama_url = input(f"Enter Ollama URL (current: {self.get('ollama_url')}): ")
        if ollama_url:
            self.set('ollama_url', ollama_url)
        collection_name = input(f"Enter collection name (current: {self.get('collection_name')}): ")
        if collection_name:
            self.set('collection_name', collection_name)
        embedding_model = input(f"Enter embedding model name (current: {self.get('embedding_model')}): ")
        if embedding_model:
            self.set('embedding_model', embedding_model)
        base_model_name = input(f"Enter base model name (current: {self.get('base_model_name')}): ")
        if base_model_name:
            self.set('base_model_name', base_model_name)
        port = input(f"Enter port (current: {self.get('port')}): ")
        if port:
            self.set('port', port)
        sqlite_web_port = input(f"Enter SQLite-Web port (current: {self.get('sqlite_web_port')}): ")
        if sqlite_web_port:
            self.set('sqlite_web_port', sqlite_web_port)
        flask_host = input(f"Enter Flask host (current: {self.get('flask_host')}): ")
        if flask_host:
            self.set('flask_host', flask_host)

    def update_from_json(self, json_data):
        """Update settings from a JSON object."""
        try:
            data = json.loads(json_data)
            for key, value in data.items():
                self.set(key, value)
        except json.JSONDecodeError as e:
            raise Exception(f"Error decoding JSON: {e}")
