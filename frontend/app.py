import streamlit as st
import requests
import uuid # <-- Added for session management

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Shruthi's private assistant", page_icon="🎀", layout="centered")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Generate a unique session ID for this specific browser tab/user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🎀 System Controls")
    
    # Display the session ID just for debugging/visibility (optional)
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if st.button("Upload & Process", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    # CRITICAL FIX: Pass the session_id as form data
                    data = {"session_id": st.session_state.session_id} 
                    
                    response = requests.post(f"{API_BASE_URL}/file-upload", files=files, data=data)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully processed {uploaded_file.name}!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection failed. Is the FastAPI backend running?")
        else:
            st.warning("Please select a file first.")

    st.divider()
    
    st.subheader("Maintenance")
    if st.button("Reset System Database", type="primary", use_container_width=True):
        with st.spinner("Resetting databases..."):
            try:
                response = requests.post(f"{API_BASE_URL}/reset")
                if response.status_code == 200:
                    st.session_state.messages = []  # Clear frontend chat history
                    # Optionally generate a new session ID on reset
                    st.session_state.session_id = str(uuid.uuid4()) 
                    st.success("System fully reset.")
                else:
                    st.error(f"Reset failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection failed.")

# --- Main Chat Interface ---
st.title("🌸 Shruthi's private assistant")
st.markdown("Ask questions based on your uploaded documents.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a technical question..."):
    # Render user message
    st.chat_message("user").markdown(prompt)
    add_message("user", prompt)

    # Fetch and render AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # CRITICAL FIX: Pass the session_id in the JSON payload
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(f"{API_BASE_URL}/question", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided by the backend.")
                    message_placeholder.markdown(answer)
                    add_message("assistant", answer)
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    message_placeholder.error(error_msg)
                    add_message("assistant", error_msg)
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Failed to connect to the backend server. Ensure it is running on port 8000."
                message_placeholder.error(error_msg)
                add_message("assistant", error_msg)