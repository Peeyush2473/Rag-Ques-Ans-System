import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import httpx

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, 'chroma_db')
SENTENCE_TRANSFORMER_MODEL_DIR = os.path.join(BASE_DIR, "backend/models/all-MiniLM-L6-v2")
USE_MODEL = 'deepseek-r1:8b'

# Load Vector Store
print("Connecting to Vector Store....")
embeddings = HuggingFaceEmbeddings(
    model_name=SENTENCE_TRANSFORMER_MODEL_DIR
)
vectorstore = Chroma(
    persist_directory = PERSIST_DIRECTORY, 
    embedding_function = embeddings
)

# Retriever (top 3 most relevant chunks)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k" : 3, "fetch_k" : 10, "lambda_mult" : 0.5}
)

# Local LLM
def load_llm_model(model_name):
    llm = ChatOllama(model=model_name, temperature=0)
    return llm

llm = load_llm_model(USE_MODEL)
    
system_prompt = (
    "You are a precise question-answering assistant. Your task is to answer questions "
    "using only the provided context.\n\n"
    
    "CORE PRINCIPLES:\n"
    "1. Answer directly and concisely - no preambles like 'Based on the context...'\n"
    "2. Use markdown to format the response to highligh the main subject of the answer.\n Eg: Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing. should be returned as <strong>Cloud computing</strong> is the <strong>on-demand delivery<\strong> of IT resources over the Internet with <strong>pay-as-you-go pricing<\strong>."
    "3. Use only information from the context - never use external knowledge\n"
    "4. If the context doesn't contain the answer, respond: 'I don't have enough information to answer this question.'\n"
    "5. Ignore irrelevant context passages completely\n\n"
    
    "CITATION REQUIREMENTS:\n"
    "- Only cite sources that directly support your answer\n\n"
    
    "HANDLING TABLES:\n"
    "- When context contains tabular data, pay careful attention to column headers\n"
    "- Verify values align with the correct columns before citing them\n"
    "- Format numbers and units exactly as they appear\n\n"
    
    "SPECIAL CASES:\n"
    "- Partial information: Provide what you can find, clearly state what's missing\n"
    "- Conflicting information: Present both perspectives with their sources\n"
    "- Ambiguous questions: Answer the most likely interpretation\n\n"
    
    "Context:\n{context}\n\n"
    
    "Remember: Accuracy over completeness. A short correct answer beats a long uncertain one."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

def ask_question(query):
    try:
        response = rag_chain.invoke({"input": query})
        answer = response['answer']
        sources = set()
        
        for doc in response['context']:
            sources.add(doc.metadata.get('source'))
            
        return {"answer" : answer, "sources" : sources}
            
    except httpx.ConnectError:
        print("\n❌ Error: Cannot connect to Ollama.")
        print("💡 Solution: Please open the Ollama app on your Mac or run 'ollama serve' in your terminal.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        
        
if __name__ == "__main__":
    while True:
        query = input("\nYou: ")
        if query.lower() in ['/exit', '/quit']:
            break
        response = ask_question(query)
        print(f"Model: {response['answer']}")
        
        print("\nSources Used: ")
        for source in response['sources']:
            print(f'- {source}')
        