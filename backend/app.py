import streamlit as st
import os
import sys
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing backend modules
from data_indexing import db, process_documents, DOCUMENT_DIRECTORY, PERSIST_DIRECTORY
from retrieval_pipeline import ask_question, vectorstore

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, div {
        color: #000000;
    }
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] * {
        color: white !important;
    }
    
    /* Chat input styling - floating capsule */
    /* Chat input styling - floating capsule */
    [data-testid="stChatInput"] {
        border-radius: 24px;
        border: 2px solid #667eea;
        padding: 8px;
        font-size: 1rem;
        background-color: #262730;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 20px; /* Slight spacing from very bottom edge */
    }

    [data-testid="stChatInput"] div:focus-within {
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stBottomBlockContainer"] {
        padding: 1rem 3.5rem 0;
        background: linear-gradient(135deg, #e1e7f1 0%, #c3cfe2 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        color: black;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
        color: 
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/book.png", width=100)
    st.title("⚙️ Configuration")
    
    # Model info
    st.info("🤖 Using **Local LLM** via Ollama")
    
    st.divider()
    
    # Document Upload Section
    with st.status("📁 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Add files to knowledge base",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc', 'pptx', 'csv', 'xlsx', 'xls'],
            help="Upload documents to add to the knowledge base",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", use_container_width=True, type="primary"):
                with st.spinner("Processing and indexing documents..."):
                    try:
                        # Save uploaded files to documents directory
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(DOCUMENT_DIRECTORY, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Process documents using your data_indexing.py function
                        process_documents(db)
                        
                        st.success("✅ Successfully processed the file(s)!")
                        
                        # Clear the file uploader by rerunning
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
    
    st.divider()
    
    # Database Statistics Section
    with st.status("📊 Database Statistics", expanded=True):
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                collection_count = db._collection.count()
                st.metric("Total Chunks Indexed", collection_count)
            except Exception as e:
                st.warning(f"Could not fetch stats: {e}")
        else:
            st.warning("⚠️ No vector database found!")
            st.info("Upload documents above to create the database")
    
    st.divider()
    
    # Show existing documents
    st.subheader("📄 Indexed Documents")
    if os.path.exists(DOCUMENT_DIRECTORY):
        files = [f for f in os.listdir(DOCUMENT_DIRECTORY) 
                if os.path.isfile(os.path.join(DOCUMENT_DIRECTORY, f)) 
                and not f.startswith('.')]
        
        if files:
            with st.expander(f"View all ({len(files)} files)"):
                for file in sorted(files):
                    st.text(f"📄 {file}")
        else:
            st.info("No documents found in the directory")
    
    st.divider()
    
    # Retrieval Info
    with st.status("🔍 Retrieval Settings", expanded=False):
        st.text("Search Type: MMR")
        st.text("Top Results: 3")
        st.text("Fetch Pool: 10")
        st.text("Diversity: 0.6")
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.divider()
    
    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **How to use:**
        1. Upload documents using the file uploader
        2. Click 'Process Documents' to add them to the knowledge base
        3. Ask questions in the main chat interface
        
        **Supported formats:**
        - PDF, TXT, DOCX, DOC
        - PPTX (PowerPoint)
        - CSV, XLSX, XLS (Excel)
        
        **Tips:**
        - Be specific in your questions
        - Check the sources to verify answers
        - Ensure Ollama app is running
        """)

# Main content
header_col1, header_col2 = st.columns([5,2])

with header_col1:
    st.markdown('<h1 class="main-header">📚 RAG Document Q&A System</h1>', unsafe_allow_html=True)
    st.caption("Ask questions about your knowledge base using a local LLM")

with header_col2:
    try:
        _ = vectorstore._collection.count()
        st.success("🟢 System Online")
    except Exception as e:
        st.error("🔴 System Offline")

st.divider()

# Display chat history using native st.chat_message
if st.session_state.chat_history:
    for idx, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.markdown(chat["question"])
            st.caption(f"🕐 {chat.get('time', '')}")
        
        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(chat["answer"], unsafe_allow_html=True)
            
            # Display sources in an expander
            if chat.get("sources"):
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(chat["sources"], 1):
                        st.text(f"{i}. {os.path.basename(source) if source else 'Unknown'}")
else:
    st.info("💬 No conversations yet. Ask a question below to begin!")
    
    # Show some example questions
    st.markdown("#### 💡 Example Questions:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What are the main topics covered?
        - Summarize the key findings
        - What data is available in the documents?
        """)
    
    with col2:
        st.markdown("""
        - Are there any specific recommendations?
        - What dates or timeframes are mentioned?
        - Explain [specific concept] from the documents
        """)

# Chat input at the bottom (pinned)
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history immediately for display
    st.session_state.chat_history.append({
        "question": prompt,
        "answer": "",  # Placeholder
        "sources": [],
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call your retrieval pipeline
                response = ask_question(prompt)
                
                # Update the last chat entry with the actual response
                st.session_state.chat_history[-1]["answer"] = response['answer']
                st.session_state.chat_history[-1]["sources"] = list(response['sources'])
                
                # Display answer
                st.markdown(response['answer'], unsafe_allow_html=True)
                
                # Display sources
                if response.get('sources'):
                    with st.expander("📎 View Sources"):
                        for i, source in enumerate(response['sources'], 1):
                            st.text(f"{i}. {os.path.basename(source) if source else 'Unknown'}")
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                error_message = str(e)
                # Remove the incomplete chat entry on error
                st.session_state.chat_history.pop()
                
                if "ConnectError" in error_message or "connect" in error_message.lower():
                    st.error("❌ Cannot connect to Ollama. Please make sure the Ollama app is running!")
                    st.info("💡 Open the Ollama app or run 'ollama serve' in your terminal")
                else:
                    st.error(f"❌ Error: {error_message}")
