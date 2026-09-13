"""
Streamlit Community Cloud UI for the Laws of Power RAG Assistant.

Unlike `streamlit_app.py`, this version runs the RAG LCEL chain directly
in-process. It does NOT depend on the FastAPI backend, making it suitable
for free deployment on Streamlit Community Cloud.

Features:
- Connects directly to local ChromaDB index
- Uses @st.cache_resource to cache BGE embedding model & retriever
- Pulls GOOGLE_API_KEY from Streamlit secrets or OS environment
- Fallback chain works seamlessly
"""

import os
import sys

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Laws of Power RAG",
    page_icon="👑",
    layout="centered",
)

# --- Configuration & Secrets ---
# Attempt to load GOOGLE_API_KEY from Streamlit secrets if available,
# otherwise fallback to local .env for testing.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # No secrets file found, fallback to env

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("""
    **Missing Configuration**
    
    Google API key is not configured. 
    Please add `GOOGLE_API_KEY` to Streamlit Secrets (for cloud deployment)
    or to a `.env` file (for local development).
    """)
    st.stop()

# Force Cloud deployment to use the lightweight 80MB model by default
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("CHROMA_DIR", "chroma_db_mini")

# Import RAG components *after* ensuring API key is set,
# so that the LLM clients initialize correctly.
import sys
from pathlib import Path

# Ensure the root of the project is in the Python path so 'src' can be found
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.generation.chain import ask, ask_situation
from src.retrieval.retriever import get_retriever


# --- Caching Expensive Resources ---
@st.cache_resource
def load_retriever(k: int):
    """
    Cache the retriever instance. 
    This prevents the 1.3GB BGE model and ChromaDB connection
    from being reinitialized on every Streamlit rerun.
    """
    try:
        # If the database doesn't exist (because we don't commit it to git), build it!
        db_path = root_dir / os.environ.get("CHROMA_DIR", "chroma_db")
        if not db_path.exists():
            with st.spinner("Building vector database for the first time... (takes ~5 seconds)"):
                import subprocess
                subprocess.run([sys.executable, "-m", "src.ingestion.build_index"], cwd=str(root_dir), check=True)
                
        # Initializing the retriever will load BGE and Chroma DB
        return get_retriever(k=k)
    except Exception as e:
        st.error(f"Error loading Vector Database or Embedding Model: {str(e)}")
        st.stop()

# --- UI Setup ---
st.title("👑 Laws of Power RAG Assistant")
st.markdown("Ask questions about the 28 Laws of Power notes and receive grounded answers with source-law attribution.")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Response Mode",
        options=["Standard", "Situation Advice"],
        help="Standard mode gives general answers. Situation mode provides practical advice for a specific scenario."
    )
    
    k_value = st.slider(
        "Number of laws to retrieve (k)",
        min_value=1,
        max_value=10,
        value=5,
        help="How many laws to fetch from the vector database as context."
    )
    
    st.markdown("---")
    st.markdown("**Architecture:**")
    st.markdown(f"- **Embeddings**: {os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')}")
    st.markdown("- **Vector Store**: ChromaDB")
    st.markdown("- **LLM**: Gemini-3.8-Flash (with fallback cascade)")
    st.markdown("- **Mode**: In-Process (Cloud Optimized)")


# Ensure resources are loaded using the selected k
# The cache handles not doing this redundantly
_ = load_retriever(k_value)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in message["sources"]:
                    st.markdown(f"- **Law {src['law']}**: {src['title']} (`{src['source']}`)")


# Accept user input
if prompt := st.chat_input("Ask about power dynamics..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Consulting the laws... (First request may take longer to load models)"):
            try:
                # Direct in-process call to the LCEL chain
                if mode == "Situation Advice":
                    result = ask_situation(prompt, k=k_value)
                else:
                    result = ask(prompt, k=k_value)
                    
                answer = result["answer"]
                sources = result["sources"]
                
                message_placeholder.markdown(answer)
                
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for src in sources:
                            st.markdown(f"- **Law {src['law']}**: {src['title']} (`{src['source']}`)")
                            
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                # Fallback chain handles 503/429 internally, so errors here are fatal (e.g. auth)
                st.error(f"Error generating response: {str(e)}")
