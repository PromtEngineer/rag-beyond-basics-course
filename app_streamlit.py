# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import tempfile
import uuid

import streamlit as st

from rag_client import rag_client


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

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
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile() as temp_file, st.status(
            "processing your document", expanded=False, state="running"
        ):
            with open(temp_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_key = f"{session_id}-{uploaded_file.name}"
            st.write("indexing in progress...")
            if file_key not in st.session_state.file_cache:
                client = rag_client(files=temp_file.name)
                st.session_state.file_cache[file_key] = client
            else:
                client = st.session_state.file_cache[file_key]
            st.write("processing complete, ask your questions...")

        display_pdf(uploaded_file)


col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with PDF")

with col2:
    st.button("Clear â†º", on_click=reset_chat)


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
        # context = st.session_state.context

        # Simulate stream of response with milliseconds delay
        for chunk in client.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "â–Œ")

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})