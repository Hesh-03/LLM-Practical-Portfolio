"""
app.py - Streamlit chat window for the University AI Assistant.

* answers questions from every indexed document (several PDFs at once)
* users can upload their own PDF / DOCX / TXT files and ask about them straight away
* shows which document and page each answer came from

Run:  streamlit run app.py
"""
import re
from pathlib import Path

import streamlit as st

from ingest import (DOCS_DIR, SUPPORTED, UPLOAD_DIR, check_ollama, clear_store, get_vectorstore,
                    ingest_file, ingest_folder, list_indexed, remove_source)
from rag_pipeline import TOP_K, answer_stream, build_chain, retrieve, source_labels

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")


@st.cache_resource(show_spinner=False)
def get_store():
    return get_vectorstore()


@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()


ok, status = check_ollama()
if not ok:
    st.error(status)
    st.stop()

vs = get_store()
indexed = list_indexed(vs)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Knowledge base")
    st.success(status)

    if indexed:
        for name, n in indexed.items():
            st.write(f"📄 **{name}**: {n} chunks")
    else:
        st.info("No documents indexed yet. Upload a file below, or put PDFs in the documents/ folder and click 'Index documents folder'.")

    selected = st.multiselect("Search only in", options=list(indexed), default=[],
                              help="Leave empty to search all documents.")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, TOP_K)

    st.subheader("Upload your own documents")
    uploads = st.file_uploader("PDF, DOCX or TXT", type=[e.strip(".") for e in sorted(SUPPORTED)],
                               accept_multiple_files=True)
    if st.button("Add to knowledge base", disabled=not uploads):
        UPLOAD_DIR.mkdir(exist_ok=True)
        for f in uploads:
            safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(f.name).name)
            target = UPLOAD_DIR / safe_name
            target.write_bytes(f.getbuffer())
            with st.spinner(f"Reading and indexing {safe_name} ..."):
                n = ingest_file(target, vs)
            if n:
                st.success(f"{safe_name}: {n} chunks added")
            else:
                st.warning(f"{safe_name}: no readable text found (a scanned PDF?)")
        st.rerun()

    if st.button("Index documents folder"):
        DOCS_DIR.mkdir(exist_ok=True)
        with st.spinner("Indexing the documents folder (this can take a few minutes) ..."):
            results = ingest_folder(DOCS_DIR, vs, progress=lambda m: None)
        st.success(f"Indexed {len(results)} file(s)")
        st.rerun()

    if indexed:
        remove = st.selectbox("Remove a document", options=[""] + list(indexed))
        if remove and st.button(f"Remove {remove}"):
            remove_source(vs, remove)
            st.rerun()
        if st.button("Clear everything"):
            clear_store(vs)
            st.session_state.pop("messages", None)
            st.rerun()

# ----------------------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for line in message["sources"]:
                    st.markdown(f"- {line}")

question = st.chat_input("Ask your university question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not indexed:
            reply, sources = "The knowledge base is empty. Upload a document first.", []
            st.markdown(reply)
        else:
            results = retrieve(vs, question, k=top_k, sources=selected or None)
            reply = st.write_stream(answer_stream(get_chain(), question, results))
            sources = [f"{doc.metadata.get('source')}, page {doc.metadata.get('page')}: "
                       f"{doc.page_content[:200].strip()}... (distance {score:.2f})"
                       for doc, score in results]
            with st.expander("Sources"):
                for line in sources:
                    st.markdown(f"- {line}")
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
