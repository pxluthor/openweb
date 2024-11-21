import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

client = os.getenv("OPENAI_API_KEY")


class Models:
    def __init__(self):
        # ollama pull mxbai-embed-large
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # ollama pull llama3.2
        self.model_ollama = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

        self.embeddings_openai = OpenAIEmbeddings(
            model="text-embedding-ada-002",
        )
        self.model_openai = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=client
        )



        # # Azure OpenAI embeddings
        # self.embeddings_azure_openai = AzureOpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     dimensions=1536,  # Can specify dimensions with new text-embedding-3 models
        #     #azure_endpoint=os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        #     #api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        #     #api_version=os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
        # )

        # # Azure OpenAI chat model
        # self.model_azure_openai = AzureChatOpenAI(
        #     #azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
        #     #api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )