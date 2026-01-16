import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENT_DIRECTORY = os.path.join(BASE_DIR, 'documents')
PERSIST_DIRECTORY = os.path.join(BASE_DIR, 'chroma_db')
SENTENCE_TRANSFORMER_MODEL_DIR = os.path.join(BASE_DIR, "backend/models/all-MiniLM-L6-v2")
# Ensure directories exist
os.makedirs(DOCUMENT_DIRECTORY, exist_ok=True)

# 2. Initialize Components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(
    model_name=SENTENCE_TRANSFORMER_MODEL_DIR
)

# Connect to existing db or create new if it does not exist
db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings, 
    )

def load_documents(directory, existing_db):
    """
    Loads documents from the specified directory with support for multiple formats.
    Skips documents that have already been embedded in vector store.
    """
    print(f"Scanning directory: {os.path.abspath(directory)}")
    documents = []
    
    existing_items = existing_db.get()
    existing_sources = set()
    
    if existing_items and 'metadatas' in existing_items:
        existing_sources = {m['source'] for m in existing_items['metadatas'] if 'source' in m}

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_lower = file.lower()
            
            if file_path in existing_sources:
                print(f"Skipping (already indexed): {file}")
                continue
            
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
                elif file_lower.endswith(".csv"):
                    loader = CSVLoader(file_path)
                elif file_lower.endswith("xlsx") or file_lower.endswith('xls'):
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    extension = os.path.splitext(file)[1]
                    print(f"Could not process {file}. '{extension}' files not supported")
                    continue
                
                print(f"Successfully loaded: {file}")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
    return documents

def process_documents(db):
    raw_documents = load_documents(DOCUMENT_DIRECTORY, db)
    
    if not raw_documents:
        print("\nNo new documents found! Check if your files are in the 'documents' folder.")
        print(f"Expected path: {os.path.abspath(DOCUMENT_DIRECTORY)}")
    else:
        # Split
        print(f"Splitting {len(raw_documents)} new documents...")
        split_docs = text_splitter.split_documents(raw_documents)
        
        # 3. Index with Batching to avoid the 5461 limit
        batch_size = 5000  # Staying safely under the 5461 limit
        print(f"Indexing {len(split_docs)} chunks into Chroma in batches...")
        
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} chunks)...")
            db.add_documents(batch)
        
        print(f"✅ Successfully indexed to Vector Store!!")
# 3. Main Execution Logic

if __name__ == "__main__":
    process_documents(db)