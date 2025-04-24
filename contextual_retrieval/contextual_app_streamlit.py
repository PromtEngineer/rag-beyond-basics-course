"""
Streamlit app for Anthropic's Contextual Retrieval implementation.
"""

import os
import base64
import gc
import tempfile
import uuid

import streamlit as st

from contextual_rag_client import ContextualRagClient
from rag_client import rag_client


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.use_contextual = True

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.title("RAG with Contextual Retrieval")
    
    # Toggle for choosing between standard and contextual RAG
    use_contextual = st.toggle("Use Contextual Retrieval", value=st.session_state.use_contextual)
    st.session_state.use_contextual = use_contextual
    
    if use_contextual:
        st.info("Using Anthropic's Contextual Retrieval approach for enhanced context understanding.")
    else:
        st.info("Using standard RAG approach.")
    
    # Model selection for LLM
    llm_option = st.radio(
        "Select LLM:",
        ["OpenAI GPT", "Anthropic Claude (if API key available)"]
    )
    use_anthropic = "Anthropic" in llm_option
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile() as temp_file, st.status(
            "processing your document", expanded=False, state="running"
        ):
            with open(temp_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Create a unique key based on the file and RAG method
            file_key = f"{session_id}-{uploaded_file.name}-{use_contextual}-{use_anthropic}"
            
            st.write("indexing in progress...")
            if file_key not in st.session_state.file_cache:
                if use_contextual:
                    client = ContextualRagClient(
                        files=temp_file.name,
                        use_anthropic=use_anthropic
                    )
                else:
                    client = rag_client(files=temp_file.name)
                
                st.session_state.file_cache[file_key] = client
            else:
                client = st.session_state.file_cache[file_key]
            
            st.write("processing complete, ask your questions...")

        display_pdf(uploaded_file)


col1, col2 = st.columns([6, 1])

with col1:
    if st.session_state.use_contextual:
        st.header("Chat with PDF using Contextual Retrieval")
    else:
        st.header("Chat with PDF using Standard RAG")

with col2:
    st.button("Clear ↺", on_click=reset_chat)


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's on your mind?"):
    if uploaded_file is None:
        st.exception(FileNotFoundError("Please upload a document first!"))
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in client.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
