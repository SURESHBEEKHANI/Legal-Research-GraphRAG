import streamlit as st
import requests

# Configure the Streamlit page
st.set_page_config(
    page_title="Graphrag - Legal Research Assistant",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Page title and caption
st.title("⚖️ Legal Research Assistant")
st.caption("🔍 AI-Powered Legal Research & Assistance")

# Custom CSS for styling with explicit !important rules to enforce color settings


# Sidebar with features and footer
with st.sidebar:
    st.divider()
    st.markdown("### Legal Research Features")
    st.markdown("""
    - ⚖️ Comprehensive Legal Analysis
    - 🔍 Case Law Research
    - 📑 Statute and Regulation Lookup
    - 📝 Document Drafting Assistance
    """)
    st.divider()
    st.markdown("Built with llama | LangChain")

# Backend API URL (update as needed)
API_URL = "http://127.0.0.1:9999/chat"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Hello! How can I assist you with your legal research today?"
    }]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Enter your legal query")

if user_query:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Send request to backend
    with st.spinner("Analyzing your query..."):
        payload = {"messages": [user_query]}
        response = requests.post(API_URL, json=payload)

    # Handle response
    if response.status_code == 200:
        response_data = response.json()
        agent_response = response_data.get("response", "No response received.")
        st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
    else:
        st.error("Failed to get a response from the server.")

    # Rerun to display updated chat history
    st.rerun()
