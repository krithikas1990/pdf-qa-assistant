"""
streamlit_app.py — Simple web UI for the PDF Q&A RAG assistant.

Run with:
    streamlit run streamlit_app.py

Make sure ANTHROPIC_API_KEY is set as an environment variable first.
Install extra dependency: pip install streamlit
"""

import streamlit as st
import tempfile
import os
from rag_core import build_index_from_pdf, ask_claude

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 Document Q&A Assistant (RAG + Claude)")
st.caption("Upload a PDF, then ask questions about its content.")

if "store" not in st.session_state:
    st.session_state.store = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.store is None:
    with st.spinner("Reading and indexing the document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.session_state.store = build_index_from_pdf(tmp_path)
        os.unlink(tmp_path)
    st.success("Document indexed. Ask away below.")

if st.session_state.store is not None:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            chunks = st.session_state.store.retrieve(question)
            answer = ask_claude(question, chunks)
        st.session_state.history.append((question, answer))

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        st.divider()

if uploaded_file is None:
    st.info("Upload a PDF above to get started.")
