# 📚 RAG Document Q&A System

A powerful, local Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents using local LLMs. Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53.0-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)

## 🚀 Features

- **📄 Multi-Format Support**: Upload and index PDF, TXT, DOCX, PPTX, CSV, and Excel files.
- **🔒 Privacy-First**: Runs entirely locally using **Ollama** (default: `deepseek-r1:8b`) and local HuggingFace embeddings.
- **⚡ Efficient Retrieval**: Uses **ChromaDB** for persistent vector storage and MMR (Maximal Marginal Relevance) for diverse context retrieval.
- **🎨 Modern UI**: Polished, "Gemini-like" interface with floating chat input, dark mode optimized text, and responsive layout.
- **🔍 Transparent Citations**: Every answer provides precise source citations from your documents.
- **🦅 Fast Indexing**: Optimized batch processing for large document sets.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS styling.
- **Orchestration**: [LangChain](https://python.langchain.com/).
- **Vector Database**: [ChromaDB](https://www.trychroma.com/).
- **LLM**: Local models via [Ollama](https://ollama.com/) (Default: DeepSeek-R1).
- **Embeddings**: `all-MiniLM-L6-v2` (via HuggingFace).

## 📋 Prerequisites

1.  **Python 3.10+** installed on your system.
2.  **Ollama** installed and running.
    - [Download Ollama](https://ollama.com/download)
    - Pull the default model:
      ```bash
      ollama pull deepseek-r1:8b
      ```

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Peeyush2473/Rag-Ques-Ans-System.git
    cd Rag-Ques-Ans-System
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Start the Application**
    ```bash
    streamlit run backend/app.py
    ```

2.  **Upload Documents**
    - Open the sidebar using the toggle button.
    - Upload your files (PDF, DOCS, etc.) in the **"Upload Documents"** section.
    - Click **"Process Documents"** to embed and index them into the vector database.

3.  **Chat with your Data**
    - Type your question in the floating input bar at the bottom.
    - The system will retrieve relevant contexts and generate an answer with sources.

## 📂 Project Structure

```
Rag-Ques-Ans-System/
├── backend/
│   ├── app.py                 # Main Streamlit application entry point
│   ├── data_indexing.py       # Logic for loading, splitting, and indexing documents
│   ├── retrieval_pipeline.py  # RAG chain, retrieval logic, and LLM integration
│   └── models/                # Directory for local embedding models
├── documents/                 # Folder where uploaded files are stored
├── chroma_db/                 # Persistent vector database storage
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
