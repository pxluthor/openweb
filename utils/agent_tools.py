import bibliotecas as b
import utils.agents_api as agents_api
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore   

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os


b.load_dotenv()
client = b.os.getenv("OPENAI_API_KEY")
client_pinecone = os.getenv("PINECONE_API_KEY")
oembed = b.OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
pc = Pinecone(api_key=client_pinecone)

def get_response_from_openai(message: str) ->str:
    llm = b.ChatOpenAI(model="gpt-4o-mini", api_key= client, temperature=0.0)
    response = llm.invoke(message)
    return response

llm = b.ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
data_eq = agents_api.data_eq
# AGENTES PARA ATENDIMENTO
@b.tool
def tool_pc_financy(question: str) -> str:
    """ Esta ferramenta serve para buscar informações gerais da empresa Leste telecom 
    como meios de contato: telefone, endereço e-mail do sac etc...
    recebe como input uma pergunta do usuário sobre a empresa Leste telecom e 
    faz uma busca no banco vetorial para encontrar a resposta """

    base = "lestefinanceiro"
    context = b.busca_pinecone(question, base)
    messages = [
        b.SystemMessage(content='''você é um assistente da empresa Leste Telecom (provedor de internet) seu objetivo é responder a pergunta do cliente com base no (contexto)
                      
                      '''),

        b.HumanMessage(content=f"Documentação: {context} \n\n {question}")   
    ]

    response = get_response_from_openai(messages)

    return response

@b.tool
def tool_pc_planos(question: str) -> str:
    """Esta ferramenta recebe como input uma pergunta do usuário sobre os planos de internet e beneficios da empresa leste telecom e 
    faz uma busca no banco vetorial para encontrar a resposta e retorna para o usuário
    Para seguir a consulta é necessário analisar o {texto} e verificar se a cidade do cliente foi informado, caso não tenha sido informado questinar ao cliente.
    """
    base = "lesteplanos"
    context = b.busca_pinecone(question, base)
    messages = [
        b.SystemMessage(content='''você é um assistente da empresa Leste Telecom (provedor de internet) seu objetivo é responder a pergunta do cliente sobre os planos de sua cidade 
                      para oferecer ao cliente a melhor oferta.
                      analise o contexto e identifique se a conversa se trata de uma venda ou uma dúvida.
                      caso o contexto for uma venda, antes de passar o plano deve-se coletar como o cliente vai usar a internet e escolher o plano compatível.
                      caso seja apenas uma dúvida informe os planos 
                      '''),
        b.HumanMessage(content=f"Documentação: {context} \n\n {question}")
       
    ]

    response = get_response_from_openai(messages)

    return response

@b.tool
def tool_pc_suporte_process(question: str, data_eq: list ) -> str:
    """ Use essa ferramenta para atendimento referente a suporte tecnico e consulta ao equipamento """
    
    prompt = b.ChatPromptTemplate.from_template('''
                    Você é especialista em suporte técnico em internet da empresa da empresa Leste Telecom, provedor de internet.
                    Utilize os dados do equipamento para gerar insites que ajudem na resolução do problema do cliente.
                                                
                    Dados do equipamento:
                    {data_eq}

                    Analise a pergunta do cliente correlacionando os dados do equipamento  
                    
                    pergunta do cliente:
                    {question}
                                                                            
                    '''
                )
    chain_equipamento = prompt | llm | b.StrOutputParser()
    response_equipamento = chain_equipamento.invoke({"data_eq": data_eq, "question": question})
    
    base = "suporteprocessos"
    context = b.busca_pinecone(question, base)

    mensagens = b.ChatPromptTemplate.from_template('''
                      Você é especialista em suporte técnico em internet da empresa da empresa Leste Telecom, provedor de internet.

                      Seu objetivo é manter a conversa com base no histórico de interação até a resolução do problema do cliente. 
                      respeitando os procedimentos que estão no contexto: {context}.

                      O contexto retorna uma lista de procedimentos a serem realizados para resolução de acordo com o problema apresentado.
                      Verifique nos dados do equipamento: {data_eq} e os insites {response_equipamento}informações relevantes para o suporte.
                      
                      Compare com base no histórico: {question} o que já foi realizado e continue os passos seguintes. 

                      Retorne ao cliente um passo de cada vez, com muita atenção para não repetir procedimentos na interaçao se baseando no histórico.
                    
                      Responda sempre em pt-BR, 

                    '''
                    )

    chain = mensagens | llm | b.StrOutputParser()
    response = chain.invoke({"context": context, 
                             "data_eq": data_eq, 
                             "question": question, 
                             "response_equipamento": response_equipamento})
    
    base = "atsuporte"
    exemple_response = b.busca_pinecone(question, base)

    messages = b.ChatPromptTemplate.from_template(''' 
                    Com base no contexto: {resposta}
                    formule uma resposta coerente para a pergunta: {question}
                    utilize o exemplo para uma diretriz
                    exemplo:{exemple_response}       
                    retorne uma instrução por vez                       
                '''
                )

    chain_revisao = messages | llm | b.StrOutputParser()
    response_final = chain_revisao.invoke({"question": question, "resposta":response, "exemple_response": exemple_response})

    return response_final

@b.tool
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


@b.tool
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


@b.tool
def busca_pinecone_planos(question: str )-> str:
    """ Use essa ferramenta para fazer um RAG das informações de suporte."""
    
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
