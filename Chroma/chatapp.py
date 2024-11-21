#Funcionou o teste no chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
import os
from dotenv import load_dotenv


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#load_dotenv()
#llm = ChatOpenAI(model="gpt-4o-mini")
#embeddings = OpenAIEmbeddings()


# Initialize the models
models = Models()
embeddings = models.embeddings_openai
llm = models.model_openai



# Initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente útil. Responda à pergunta com base apenas nos dados fornecidos."),
        ("human", "Use a pergunta do usuário {input} para responder à pergunta. Utilize apenas o {context} para responder à pergunta.")
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 4})

#------TESTE---------
# saida usando LCEL - langchain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

### saida normal 
# response = qa_chain.invoke("O que é Cgnat?")
# print(response)
# combine_docs_chain = create_stuff_documents_chain(
#     llm, prompt
# )

# Main loop

def main():
    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
        
        result = qa_chain.invoke(query)
        print("Assistant: ", result, "\n\n")
        #print("Assistant: ", result["answer"], "\n\n")  #==> usado na saida normal
        
# # Run the main loop
if __name__ == "__main__":
    main()