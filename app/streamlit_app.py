"""
Streamlit UI — clean interface for the Laws of Power RAG Assistant.

Communicates with the FastAPI backend (decoupled architecture).

Run with:
    streamlit run app/streamlit_app.py
"""

import requests
import streamlit as st

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Laws of Power RAG",
    page_icon="👑",
    layout="centered",
)

st.title("👑 Laws of Power RAG Assistant")
st.markdown("Ask a question about the laws, or describe a situation you're facing.")

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
    st.markdown("- **Embeddings**: BAAI/bge-large-en-v1.5")
    st.markdown("- **Vector Store**: ChromaDB")
    st.markdown("- **LLM**: Gemini-3.8-Flash (with fallback cascade)")


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
        
        with st.spinner("Consulting the laws..."):
            try:
                # Call the FastAPI backend
                api_mode = "situation" if mode == "Situation Advice" else "standard"
                payload = {
                    "query": prompt,
                    "mode": api_mode,
                    "k": k_value
                }
                
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
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
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the backend API at {API_URL}. Is it running?")
