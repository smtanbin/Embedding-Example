import logging
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from src.database.Database import Database

class Embedding:
    def __init__(self, ollama_url: str, embedding_ollama_model: str, database_name: str):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.ollama_base_url = ollama_url
        self.emb_model = embedding_ollama_model
        self.db = Database(database_name)

    def embed(self, document_path, chunk_size: int = 500):
        try:
            self.logger.info(f"Starting the embedding process for document at {document_path}.")
            chunks = self.__document_loader(document_path, chunk_size)
            embeddings = self.__emb_invoke(chunks)
            result = self.__load_to_db(chunks, embeddings)
            self.logger.info(f"Embedding process completed successfully for document at {document_path}.")
            return result
        except Exception as e:
            self.logger.error(f"Error during embedding process: {e}")
            raise

    def __document_loader(self, custom_path, chunk_size):
        try:
            self.logger.info(f"Loading document from path: {custom_path}.")
            directory_path = os.path.join('documents', custom_path)

            if not os.path.isdir(directory_path):
                raise FileNotFoundError(f"The directory at {directory_path} does not exist.")

            all_splits = []
            document_id = str(custom_path)  # Use custom path as document ID

            for filename in os.listdir(directory_path):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(directory_path, filename)

                    texts = ""
                    with open(pdf_path, 'rb') as file:
                        reader = PdfReader(file)
                        for page in reader.pages:
                            texts += page.extract_text()

                    document = Document(page_content=texts)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
                    splits = text_splitter.split_documents([document])

                    # Create CustomDocument objects with chunk IDs
                    for i, split in enumerate(splits):
                        chunk_id = f"{document_id}_{i}"
                        all_splits.append(CustomDocument(split, chunk_id))

            self.logger.info(f"Document loaded and split into {len(all_splits)} chunks.")
            return all_splits
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            raise

    def __emb_invoke(self, chunks):
        try:
            self.logger.info(f"Connecting to Ollama for embedding using model {self.emb_model}.")
            print(self.ollama_base_url)
            ollama = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model=self.emb_model
            )
            embeddings = ollama.embed_documents([chunk.page_content for chunk in chunks])
            self.logger.info(f"Embedding completed successfully.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error during embedding with Ollama: {e}")
            raise

    def __load_to_db(self, chunks, embeddings):
        try:
            self.logger.info("Loading embeddings to database.")
            results = []
            for chunk, embedding in zip(chunks, embeddings):
                uuid = self.db.add_data(chunk.page_content, str(embedding))
                self.db.add_embedding(uuid, embedding)
                results.append(uuid)
            self.logger.info("Embeddings successfully loaded into database.")
            return results
        except Exception as e:
            self.logger.error(f"Error loading embeddings to database: {e}")
            raise

    def query_embeddings(self, query):
        self.logger.info("Finding the closest match in the database.")

        # Fetch all embeddings from the database
        all_embeddings = self.db.get_all_embeddings()

        # Convert query to numpy array
        query_embedding = np.array(self.__embed_query(query))

        closest_match = None
        min_distance = float('inf')

        for uuid, db_embedding in all_embeddings:
            # Calculate distance
            try:
                distance = np.linalg.norm(query_embedding - db_embedding)
                if distance < min_distance:
                    min_distance = distance
                    closest_match = [uuid,db_embedding]
            except Exception as e:
                self.logger.error(f"Error calculating distance: {e}")

        self.logger.info(f"Closest match found: {closest_match}")
        return closest_match

    def __embed_query(self, query):
        try:
            self.logger.info(f"Embedding the query: {query}.")
            ollama = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model=self.emb_model
            )
            query_embedding = ollama.embed_documents([query])[0]
            return np.array(query_embedding)
        except Exception as e:
            self.logger.error(f"Error embedding query: {e}")
            raise


class CustomDocument:
    def __init__(self, document: Document, chunk_id: str):
        self.document = document
        self.chunk_id = chunk_id

    @property
    def page_content(self):
        return self.document.page_content