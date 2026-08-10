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
        self.session = requests.Session()  # Connection pooling

    # ---------- Auth ----------
    def signup(self, username: str, password: str) -> Dict[str, Any]:
        url = f"{self.base_url}/signup"
        payload = {"username": username, "password": password}
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """/token uses OAuth2PasswordRequestForm, so this MUST be sent as form data, not JSON."""
        url = f"{self.base_url}/token"
        data = {"username": username, "password": password}
        response = self.session.post(url, data=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()  # {"access_token": ..., "token_type": "bearer"}

    # ---------- Authenticated helpers ----------
    def _auth_headers(self, token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def upload_document(self, file_name: str, file_bytes: bytes, session_id: str, token: str) -> Dict[str, Any]:
        url = f"{self.base_url}/file-upload"
        files = {"file": (file_name, file_bytes, "application/pdf")}
        data = {"session_id": session_id}

        response = self.session.post(
            url, files=files, data=data, headers=self._auth_headers(token), timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def ask_question(self, question: str, session_id: str, token: str) -> str:
        url = f"{self.base_url}/question"
        payload = {"question": question, "session_id": session_id}

        response = self.session.post(
            url, json=payload, headers=self._auth_headers(token), timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("answer", "Warning: Backend returned a 200 OK but no answer payload.")

    def reset_database(self, token: str) -> Dict[str, Any]:
        url = f"{self.base_url}/reset"
        response = self.session.post(url, headers=self._auth_headers(token), timeout=self.timeout)
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
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "username" not in st.session_state:
        st.session_state.username = None


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def reset_frontend_session() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def logout() -> None:
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


# --- Auth UI ---
def render_auth_screen() -> None:
    st.title("🤖 Enterprise RAG Assistant")
    st.caption("Please sign in to continue.")

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log In", use_container_width=True)

        if submitted:
            if not username or not password:
                st.warning("Please enter both a username and password.")
            else:
                try:
                    token_data = api_client.login(username, password)
                    st.session_state.access_token = token_data["access_token"]
                    st.session_state.username = username
                    st.success("Logged in successfully.")
                    st.rerun()
                except Timeout:
                    st.error("Login failed: request timed out.")
                except RequestException as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 401:
                        st.error("Invalid username or password.")
                    else:
                        logger.error(f"Login failed: {e}")
                        st.error(f"Login failed: {str(e)}")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted_signup = st.form_submit_button("Create Account", use_container_width=True)

        if submitted_signup:
            if not new_username or not new_password:
                st.warning("Please fill out all fields.")
            elif new_password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                try:
                    api_client.signup(new_username, new_password)
                    st.success("Account created. You can now log in from the 'Log In' tab.")
                except Timeout:
                    st.error("Sign up failed: request timed out.")
                except RequestException as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 400:
                        st.error("That username is already taken.")
                    else:
                        logger.error(f"Signup failed: {e}")
                        st.error(f"Sign up failed: {str(e)}")


# --- UI Components ---
def render_sidebar() -> None:
    with st.sidebar:
        st.title("System Controls")
        st.caption(f"Logged in as: **{st.session_state.username}**")
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")

        if st.button("Log Out", use_container_width=True):
            logout()
            st.rerun()

        st.divider()
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
                        session_id=st.session_state.session_id,
                        token=st.session_state.access_token,
                    )
                    st.success(f"Successfully indexed: {uploaded_file.name}")
                except Timeout:
                    st.error("Upload failed: Request timed out. File might be too large or backend is unresponsive.")
                except RequestException as e:
                    if getattr(e.response, "status_code", None) == 401:
                        st.error("Session expired. Please log in again.")
                        logout()
                        st.rerun()
                    else:
                        logger.error(f"Document upload failed: {e}")
                        st.error(f"Upload failed: {str(e)}")

        st.divider()
        st.subheader("System Maintenance")
        if st.button("Purge System Database", type="primary", use_container_width=True):
            with st.spinner("Flushing vector store and resetting state..."):
                try:
                    api_client.reset_database(token=st.session_state.access_token)
                    reset_frontend_session()
                    st.success("System databases successfully purged.")
                except RequestException as e:
                    if getattr(e.response, "status_code", None) == 401:
                        st.error("Session expired. Please log in again.")
                        logout()
                        st.rerun()
                    else:
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
                        session_id=st.session_state.session_id,
                        token=st.session_state.access_token,
                    )
                    message_placeholder.markdown(answer)
                    add_message("assistant", answer)

                except Timeout:
                    error_msg = "Inference failed: Backend request timed out."
                    message_placeholder.error(error_msg)
                    add_message("assistant", error_msg)
                except RequestException as e:
                    if getattr(e.response, "status_code", None) == 401:
                        error_msg = "Session expired. Please log in again."
                        message_placeholder.error(error_msg)
                        add_message("assistant", error_msg)
                        logout()
                        st.rerun()
                    else:
                        logger.error(f"Inference generation failed: {e}")
                        error_msg = f"System Error: Unable to communicate with inference backend. Details: {str(e)}"
                        message_placeholder.error(error_msg)
                        add_message("assistant", error_msg)


# --- Main Application Execution ---
def main() -> None:
    initialize_session_state()

    if not is_authenticated():
        render_auth_screen()
        return

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()