import os
import uuid
import logging
from typing import Dict, List, Optional, Any

import streamlit as st
import requests
from requests.exceptions import RequestException, Timeout

# --- Configuration & Setup ---
# Default to localhost if environment variable is not set
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30")) # 30 seconds default timeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="centered")

# --- API Client Abstraction ---
class RAGBackendClient:
    """Handles all HTTP communication with the FastAPI backend."""
    
    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session() # Connection pooling

    def upload_document(self, file_name: str, file_bytes: bytes, session_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/file-upload"
        files = {"file": (file_name, file_bytes, "application/pdf")}
        data = {"session_id": session_id}
        
        response = self.session.post(url, files=files, data=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def ask_question(self, question: str, session_id: str) -> str:
        url = f"{self.base_url}/question"
        payload = {"question": question, "session_id": session_id}
        
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("answer", "Warning: Backend returned a 200 OK but no answer payload.")

    def reset_database(self) -> Dict[str, Any]:
        url = f"{self.base_url}/reset"
        response = self.session.post(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

# Instantiate the client
api_client = RAGBackendClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)

# --- Session State Management ---
def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})

def reset_frontend_session() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

# --- UI Components ---
def render_sidebar() -> None:
    with st.sidebar:
        st.title("System Controls")
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
        
        st.subheader("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF Context", type=["pdf"])
        
        if st.button("Process Document", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please select a file prior to processing.")
                return
                
            with st.spinner("Vectorizing and indexing document..."):
                try:
                    api_client.upload_document(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        session_id=st.session_state.session_id
                    )
                    st.success(f"Successfully indexed: {uploaded_file.name}")
                except Timeout:
                    st.error("Upload failed: Request timed out. File might be too large or backend is unresponsive.")
                except RequestException as e:
                    logger.error(f"Document upload failed: {e}")
                    st.error(f"Upload failed: {str(e)}")

        st.divider()
        st.subheader("System Maintenance")
        if st.button("Purge System Database", type="primary", use_container_width=True):
            with st.spinner("Flushing vector store and resetting state..."):
                try:
                    api_client.reset_database()
                    reset_frontend_session()
                    st.success("System databases successfully purged.")
                except RequestException as e:
                    logger.error(f"Database reset failed: {e}")
                    st.error(f"Failed to reset system: {str(e)}")

def render_chat_interface() -> None:
    st.title("Document QA Interface")
    st.markdown("Query the ingested document context using the integrated RAG pipeline.")

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new input
    if prompt := st.chat_input("Input technical query..."):
        st.chat_message("user").markdown(prompt)
        add_message("user", prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Generating inference..."):
                try:
                    answer = api_client.ask_question(
                        question=prompt, 
                        session_id=st.session_state.session_id
                    )
                    message_placeholder.markdown(answer)
                    add_message("assistant", answer)
                    
                except Timeout:
                    error_msg = "Inference failed: Backend request timed out."
                    message_placeholder.error(error_msg)
                    add_message("assistant", error_msg)
                except RequestException as e:
                    logger.error(f"Inference generation failed: {e}")
                    error_msg = f"System Error: Unable to communicate with inference backend. Details: {str(e)}"
                    message_placeholder.error(error_msg)
                    add_message("assistant", error_msg)

# --- Main Application Execution ---
def main() -> None:
    initialize_session_state()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()