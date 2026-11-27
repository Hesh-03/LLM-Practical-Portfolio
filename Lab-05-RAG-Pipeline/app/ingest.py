"""
ingest.py - Step 1 of the RAG pipeline: load documents, split them into chunks,
embed the chunks and store them in ChromaDB.

Supports several PDF files at once (and also .docx and .txt). Files can come from the
`documents/` folder (run this script) or from the Streamlit app (user uploads).

Usage:
    python ingest.py            # index every file in the documents/ folder
    python ingest.py --reset    # empty the knowledge base first, then index
    python ingest.py --list     # show what is currently indexed
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"        # put your PDFs here
UPLOAD_DIR = BASE_DIR / "uploads"        # files uploaded through the web app are saved here
CHROMA_DIR = BASE_DIR / "chroma_db"      # the vector database is stored here
COLLECTION = "university_docs"

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # characters shared between neighbouring chunks
SUPPORTED = {".pdf", ".docx", ".txt"}


# ----------------------------------------------------------------------------- Ollama
def check_ollama():
    """Return (ok, message). Checks that Ollama is running and both models are pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return False, ("Cannot reach Ollama at " + OLLAMA_URL + ". Start the Ollama app "
                       "(or run `ollama serve`) and try again.")
    def is_pulled(wanted):
        # "llama3.2:3b" must match exactly; a name without a tag means ":latest"
        return wanted in models or (":" not in wanted and wanted + ":latest" in models)
    missing = [m for m in (LLM_MODEL, EMBED_MODEL) if not is_pulled(m)]
    if missing:
        return False, "Missing Ollama model(s): " + ", ".join(missing) + ". Run `ollama pull <name>` for each."
    return True, f"Ollama is running with {LLM_MODEL} and {EMBED_MODEL}."


# ----------------------------------------------------------------------------- vector store
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def list_indexed(vs):
    """Return {file name: number of chunks} for everything in the knowledge base."""
    metadatas = vs.get(include=["metadatas"])["metadatas"] or []
    counts = {}
    for m in metadatas:
        name = (m or {}).get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def remove_source(vs, name):
    """Delete all chunks that came from one file."""
    ids = vs.get(where={"source": name})["ids"]
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def clear_store(vs):
    """Delete everything from the knowledge base."""
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)
    return len(ids)


# ----------------------------------------------------------------------------- loading
def load_file(path):
    """Read one file and return a list of Documents (one per PDF page / per file)."""
    path = Path(path)
    ext = path.suffix.lower()
    docs = []
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(str(path)) as pdf:
            for number, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": number}))
    elif ext == ".docx":
        import docx
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(parts).strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": 1}))
    elif ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": 1}))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = number
    return chunks


# ----------------------------------------------------------------------------- ingestion
def ingest_file(path, vs):
    """Index one file. Re-indexing a file with the same name replaces its old chunks.
    Returns the number of chunks stored (0 if the file has no readable text, e.g. a scanned PDF)."""
    path = Path(path)
    docs = load_file(path)
    if not docs:
        return 0
    chunks = split_documents(docs)
    remove_source(vs, path.name)
    ids = [hashlib.md5(f"{path.name}-{c.metadata['chunk']}-{c.page_content}".encode("utf-8")).hexdigest()
           for c in chunks]
    vs.add_documents(chunks, ids=ids)
    return len(chunks)


def ingest_folder(folder, vs, progress=print):
    """Index every supported file in a folder. Returns {file name: chunks}."""
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in SUPPORTED)
    results = {}
    for path in files:
        progress(f"Indexing {path.name} ...")
        results[path.name] = ingest_file(path, vs)
        progress(f"   -> {results[path.name]} chunks")
    return results


# ----------------------------------------------------------------------------- command line
def main():
    parser = argparse.ArgumentParser(description="Index documents into the ChromaDB knowledge base.")
    parser.add_argument("--reset", action="store_true", help="empty the knowledge base before indexing")
    parser.add_argument("--list", action="store_true", help="only list what is currently indexed")
    args = parser.parse_args()

    ok, message = check_ollama()
    print(message)
    if not ok:
        sys.exit(1)

    vs = get_vectorstore()
    if args.list:
        for name, n in list_indexed(vs).items():
            print(f"{name}: {n} chunks")
        return
    if args.reset:
        print(f"Removed {clear_store(vs)} existing chunks.")

    DOCS_DIR.mkdir(exist_ok=True)
    if not any(p.suffix.lower() in SUPPORTED for p in DOCS_DIR.glob("*")):
        print(f"No documents found. Put PDF files in {DOCS_DIR} and run this script again.")
        return
    results = ingest_folder(DOCS_DIR, vs)
    print("\nDone. Knowledge base now contains:")
    for name, n in list_indexed(vs).items():
        print(f"  {name}: {n} chunks")
    skipped = [n for n, c in results.items() if c == 0]
    if skipped:
        print("\nNo text could be read from (scanned or image-only?):", ", ".join(skipped))


if __name__ == "__main__":
    main()
