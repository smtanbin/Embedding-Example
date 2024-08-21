# agent.py

import os
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI



class Agent:
    def __init__(self, embedding, ollama_url=None, model_name: str = "llama3.1-8b",
                 system_prompt: str = "You are a helpful assistant."):
        self.embedding = embedding
        self.ollama_url = ollama_url
        self.model = model_name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(__name__)
        self.openai_key = os.environ.get('OpenAIKey')
        self.llm = OpenAI(api_key=self.openai_key, model="text-davinci-003")
        self.prompt_template = PromptTemplate.from_template("""
        {system_prompt}

        User: {input}
        """)

    def input(self, prompt_text):
        try:
            # Retrieve vector data using the embedding instance
            vector_data = self.embedding.query_embeddings(prompt_text)
            self.logger.info(f"Vector data retrieved for prompt: {prompt_text}")

            # Make AI chain chat request using LangChain
            response = self._make_ai_chain_chat_request(vector_data)
            self.logger.info(f"AI chain chat response: {response}")

            return response
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            raise

    def _make_ai_chain_chat_request(self, vector_data):
        try:
            self.logger.info(f"Using {self.llm.__class__.__name__} for AI chain chat request.")

            # Create the LLMChain
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template
            )

            # Run the LLMChain with the vector data and system prompt
            response = llm_chain.run({
                "system_prompt": self.system_prompt,
                "input": vector_data
            })
            return response
        except Exception as e:
            self.logger.error(f"Error during AI chain chat request: {e}")
            raise
