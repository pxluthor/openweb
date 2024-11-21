#from pinecone_upsert import busca_pinecone
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.llm import LLMChain

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
classify_atendimento = []
data_eq=[]


@tool
def verify_client(question: str) -> str:
    '''
    Use esta tool para verificar se o cpf é de um cliente e para pegar os dados do equipamento
    '''
    toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,
)

    tools = toolkit.get_tools()

    api_spec = """
        openapi: 3.0.0
        info:
        title: JSONPlaceholder API
        version: 1.0.0
        servers:
        - url: https://sosbeta.lestetelecom.com.br/api/whatsapp/builder/validacpf
        - parameters: cpfcnpj
        exemplo para o body:
        {
            "cpfcnpj": "11635960789"
        }
        """
    
    api_equipamento = """
        penapi: 3.0.0
        info:
        title: JSONEquipamento
        version: 1.0.0
        servers:
        - url: https://testeserver.pxluthor.com.br/consulta
        - parameters: cpfcnpj
        exemplo para o body:
        {
            "cpfcnpj": "11635960789"
        }

        """

    system_message = f'''
            Você tem acesso a algumas uma APIs para ajudar verificar se o cpf ou cnpj corresponde a um cliente na base de dados da empresa e para pegar os dados do equipamento.
            Aqui está a documentação sobre a API.

            {api_spec}

            <contexto para a resposta do cpf ou cnpj>
            Se retronar o campo: "precadastroid" o cpf/cnpj é de uma pessoa que tem cadastro mas ainda não é cliente.
            Se retornar o campo: "codcli" o cpf/cnpj é de cliente na base de dados.
            Se a resposta for um campo vazio o cpf/cnpj não é cliente e não tem pré-cadastro, portanto é um possivel cliente.

            {api_equipamento}
            <contexto para a resposta dos dados do equipamento>
            sinal: caso o cliente esteja com o sinal < -25 possui degradação, é preciso verificar o cordão optico se não voltar ao normal deve abrir uma visita tecnica.
            status_connection: informa se o cliente está conectado ou não no momento
            temperatura: informa a temperatura do equipamento
            velocidade: teste
            alarme: informa se o equipamento  possui tem algum alarme de erro
            name: informa o nome do titular.
            mode_operation: informa o modo de operação da ONU
            olt: informa a OLT 
            model_onu: informa o modelo da ONU 
            slot: informa o slot,
            pon: informa o pon ,
            vlan: informa a vlan,
            box: informa a caixa de instalação,
            SW: RP2639,
            HW: WKE2.134.321f1G
            atividades tipo: Manutencao informa que tem uma visita em aberto
            criada_em:  informa a data da visita 
        '''

    llm = ChatOpenAI(model="gpt-4o-mini")

    #llm.bind
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    events = agent_executor.invoke({"messages": [("user", question)]})

    response = str(events["messages"][-1])
    resposta = response.split('additional_kwargs')[0]
    data_eq.append(resposta)

    prompt = ChatPromptTemplate.from_template(
        ''' 
            Faça uma revisão do contexto retornando apenas a informação se o cpf ou cnpj é de um cliente Ativo ou um potencial cliente. 
            não passe a informação dos dados do equipamento.

            contexto:
            {resposta}

            '''
    )

    chain = prompt | llm | StrOutputParser()
    response_final = chain.invoke({"resposta": resposta})


    prompt_class = ChatPromptTemplate.from_template(
        ''' 
            retorne a classificação do atendimento que está no contexto.

            contexto:
            {resposta}

            exemplo:
            Classificação do atendimento: Suporte Técnico.

            '''
    )

    chain_class = prompt_class | llm | StrOutputParser()
    response_class = chain_class.invoke({"resposta": resposta})
    classify_atendimento.append(response_class)

    return response_final

