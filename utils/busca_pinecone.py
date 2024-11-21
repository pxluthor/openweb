# testar os retrievers  no pinecone

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore 

load_dotenv()
client = os.getenv("OPENAI_API_KEY")
client_source = os.getenv("TAVILY_API_KEY")
client_pinecone = os.getenv("PINECONE_API_KEY")
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
pc = Pinecone(api_key=client_pinecone)
llm = ChatOpenAI(model="gpt-4o-mini")

def busca_pinecone_planos(question: str )-> str:
    """ Use essa ferramenta para buscar as informações dos planos de internet da empresa Leste Telecom """
    
    template = """
    Baseado no contexto abaixo, responda a pergunta:
    {context}
    Pergunta: 
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    index_name = "lesteplanos"
    index = pc.Index(index_name)
    text_field = "text"  
    vector_store = PineconeVectorStore(  
        index, oembed, text_field  
    )  
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | llm | StrOutputParser()
    response = chain.invoke(question)
    return response

def busca_pinecone_suporte(question: str )-> str:
    """ Use essa ferramenta para fazer um RAG das informações de suporte."""
    
    template = """
    Baseado no contexto abaixo, responda a pergunta:

    {context}

    Pergunta: 

    {question}

    """

    prompt = ChatPromptTemplate.from_template(template)

    index_name = "suporteprocessos"
    index = pc.Index(index_name)
    text_field = "text"  
    vector_store = PineconeVectorStore(  
        index, oembed, text_field  
    )  
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | llm | StrOutputParser()
    response = chain.invoke(question)
    return response

def pinecone_financy(question: str )-> str:
    """ Use essa ferramenta para fazer um RAG das informações de suporte."""
    
    template = """
    Baseado no contexto abaixo, responda a pergunta:

    {context}

    Pergunta: 

    {question}

    """

    prompt = ChatPromptTemplate.from_template(template)

    index_name = "lestefinanceiro"
    index = pc.Index(index_name)
    text_field = "text"  
    vector_store = PineconeVectorStore(  
        index, oembed, text_field  
    )  
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | llm | StrOutputParser()
    response = chain.invoke(question)
    return response





response = busca_pinecone_planos(" quais os planos de internet  para a cidade de niteroi? ")
print(busca_pinecone_planos)