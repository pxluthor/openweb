#from pinecone_upsert import busca_pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.llm import LLMChain

import utils.agents_api as agents_api

from langgraph.prebuilt import create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper

# chat o llm
from langchain_openai import ChatOpenAI 
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# decorator @tool transforma uma função em tool
from langchain.tools import tool 
# realiza o import dos agents
from langchain.agents import AgentExecutor, create_openai_tools_agent

#import mysql.connector
from datetime import datetime
import uuid
import requests
import os
import time
from dotenv import load_dotenv


load_dotenv()
client = os.getenv("OPENAI_API_KEY")
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")