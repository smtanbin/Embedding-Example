import sqlite3
import os
import uuid
import datetime
import json
import logging

import numpy as np


class Database:
    def __init__(self, db_name):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Set the path to the 'data' directory and ensure it exists
        self.db_folder = 'data'
        self.db_name = db_name + ".db"
        self.db_path = os.path.join(self.db_folder, self.db_name)
        self.logger.info(f"Database path: {self.db_path}")

        # Ensure the 'data' directory exists
        if not os.path.exists(self.db_folder):
            try:
                os.makedirs(self.db_folder)
                self.logger.info(f"Created directory: {self.db_folder}")
            except OSError as e:
                self.logger.error(f"Error creating directory {self.db_folder}: {e}")
                raise

        self.conn = None
        self.cursor = None
        self._create_connection()
        self._create_table()

    def _create_connection(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating connection to SQLite database at {self.db_path}: {e}")
            raise

    def _create_table(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    uuid TEXT PRIMARY KEY,
                    document_name TEXT,
                    timestamp TEXT,
                    data TEXT,
                    embeddings TEXT,
                    emb_timestamp TEXT
                )
            ''')
            self.conn.commit()
            self.logger.info("Table 'documents' created or already exists.")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating table: {e}")
            raise

    def add_data(self, document_name, data):
        document_uuid = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        try:
            self.cursor.execute('''
                INSERT INTO documents (uuid, document_name, timestamp, data)
                VALUES (?, ?, ?, ?)
            ''', (document_uuid, document_name, timestamp, data))
            self.conn.commit()
            self.logger.info(f"Data added with UUID: {document_uuid}")
            return document_uuid
        except sqlite3.Error as e:
            self.logger.error(f"Error adding data: {e}")
            raise

    def add_embedding(self, document_uuid, embeddings):
        emb_timestamp = datetime.datetime.now().isoformat()
        # Convert the list of embeddings to a JSON string
        embeddings_str = json.dumps(embeddings)
        try:
            self.cursor.execute('''
                UPDATE documents
                SET embeddings = ?, emb_timestamp = ?
                WHERE uuid = ?
            ''', (embeddings_str, emb_timestamp, document_uuid))
            self.conn.commit()
            self.logger.info(f"Embedding added for UUID: {document_uuid}")
        except sqlite3.Error as e:
            self.logger.error(f"Error adding embedding: {e}")
            raise

    def retrieve_data(self, query):
        try:
            self.cursor.execute('''
                SELECT * FROM documents
                WHERE document_name LIKE ?
            ''', ('%' + query + '%',))
            results = self.cursor.fetchall()
            self.logger.info(f"Data retrieved for query '{query}': {results}")
            return results
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving data: {e}")
            raise

    def drop_all_data(self):
        try:
            self.cursor.execute('DELETE FROM documents')
            self.conn.commit()
            self.logger.info("All data dropped from 'documents' table.")
        except sqlite3.Error as e:
            self.logger.error(f"Error dropping all data: {e}")
            raise

    def get_all_embeddings(self):
        try:
            # Assuming `results` is a list of tuples (uuid, embeddings)
            results = self._fetch_embeddings_from_db()
            parsed_results = []
            for uuid, embeddings in results:
                if embeddings is not None:
                    try:
                        parsed_results.append((uuid, np.array(json.loads(embeddings))))
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON for uuid {uuid}: {e}")
                else:
                    logging.warning(f"Embeddings for uuid {uuid} are None and will be skipped.")
            return parsed_results
        except Exception as e:
            logging.error(f"Error fetching embeddings: {e}")
            raise

    def _fetch_embeddings_from_db(self):
        try:
            self.cursor.execute('''
                SELECT uuid, embeddings FROM documents
            ''')
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            logging.error(f"Error fetching embeddings from database: {e}")
            raise

    def __del__(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Connection to SQLite database closed.")
