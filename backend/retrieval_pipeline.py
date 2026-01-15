import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, 'chroma_db')

# Load Vector Store
print("Connecting to Vector Store....")
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
)
vectorstore = Chroma(
    persist_directory = PERSIST_DIRECTORY, 
    embedding_function = embeddings
)

# Retriever (top 3 most relevant chunks)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k" : 2, "fetch_k" : 10, "lambda_mult" : 0.5}
)

# Local LLM
llm = ChatOllama(model='llama3.1', temperature=0)

system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Crucially: If a piece of context is irrelevant to the question, ignore it. "
    "Only cite the sources that actually helped you answer the question."
    "If you don't know the answer, just say that you don't know."
    "Context : {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

if __name__ == "__main__":
    query = input("You: ")
    response = rag_chain.invoke({"input" : query})
    
    print(f"Model: {response['answer']}")
    
    print("\nSources used: ")
    for doc in response['context']:
        print(f"'{doc.metadata.get('source')}")