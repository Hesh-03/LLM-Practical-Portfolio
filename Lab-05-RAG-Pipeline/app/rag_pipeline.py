"""
rag_pipeline.py - Step 2 of the RAG pipeline: retrieve relevant chunks from ChromaDB,
put them into a prompt and let the LLM (Llama 3.2 3B via Ollama) answer from them.

Usage (command-line chat):
    python rag_pipeline.py
"""
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ingest import LLM_MODEL, OLLAMA_URL, check_ollama, get_vectorstore, list_indexed

TOP_K = 4

SYSTEM_PROMPT = (
    "You are a university assistant. Answer the question using ONLY the context taken from the "
    "university documents below. If the answer is not in the context, reply exactly: "
    "'I could not find this in the provided documents.' Do not guess or use outside knowledge. "
    "Be concise, use a bullet list for lists, and mention the document name when it helps."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def get_llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)


def build_chain():
    return PROMPT | get_llm() | StrOutputParser()


def retrieve(vs, question, k=TOP_K, sources=None):
    """Find the k most similar chunks. `sources` optionally limits the search to some files.
    Returns a list of (Document, distance); a smaller distance means a closer match."""
    flt = None
    if sources:
        flt = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
    return vs.similarity_search_with_score(question, k=k, filter=flt)


def format_context(results):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')}, page {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc, _ in results
    )


def source_labels(results):
    """Unique 'file (page N)' labels for the retrieved chunks, in order."""
    seen, labels = set(), []
    for doc, _ in results:
        label = f"{doc.metadata.get('source')} (page {doc.metadata.get('page')})"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def answer_stream(chain, question, results):
    """Yield the answer piece by piece (so a chat window can show it as it is written)."""
    yield from chain.stream({"context": format_context(results), "question": question})


def answer(chain, question, results):
    return chain.invoke({"context": format_context(results), "question": question})


def main():
    ok, message = check_ollama()
    print(message)
    if not ok:
        sys.exit(1)
    vs = get_vectorstore()
    indexed = list_indexed(vs)
    if not indexed:
        print("The knowledge base is empty. Run `python ingest.py` first.")
        sys.exit(1)
    print(f"Knowledge base: {len(indexed)} document(s), {sum(indexed.values())} chunks")

    chain = build_chain()
    while True:
        question = input("\nAsk a university question (or exit): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        results = retrieve(vs, question)
        print("\nAnswer: ", end="", flush=True)
        for piece in answer_stream(chain, question, results):
            print(piece, end="", flush=True)
        print("\n\nSources: " + "; ".join(source_labels(results)))


if __name__ == "__main__":
    main()
