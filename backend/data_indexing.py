import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Configuration - Using absolute path to avoid directory confusion
# This points to a 'documents' folder in the root of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENT_DIRECTORY = os.path.join(BASE_DIR, 'documents')
PERSIST_DIRECTORY = os.path.join(BASE_DIR, 'chroma_db')

# Ensure directories exist
os.makedirs(DOCUMENT_DIRECTORY, exist_ok=True)

# 2. Initialize Components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents(directory):
    """
    Loads documents from the specified directory with support for multiple formats.
    """
    print(f"Scanning directory: {os.path.abspath(directory)}")
    documents = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return documents

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()
            
            try:
                # Handle different file types
                if file_lower.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_lower.endswith(".docx") or file_lower.endswith(".doc"):
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_lower.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_lower.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(file_path)
                else:
                    # Skip unsupported files
                    continue
                
                print(f"Successfully loaded: {file}")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
    return documents

# 3. Main Execution Logic
if __name__ == "__main__":
    # Load
    raw_documents = load_documents(DOCUMENT_DIRECTORY)
    
    if not raw_documents:
        print("\n❌ Error: No documents found! Check if your files are in the 'documents' folder.")
        print(f"Expected path: {os.path.abspath(DOCUMENT_DIRECTORY)}")
    else:
        # Split
        print(f"Splitting {len(raw_documents)} documents...")
        split_docs = text_splitter.split_documents(raw_documents)
        
        # Index
        print(f"Indexing {len(split_docs)} chunks into Chroma...")
        db = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings, 
            persist_directory=PERSIST_DIRECTORY
        )
        
        print(f"✅ Successfully indexed to {PERSIST_DIRECTORY}")