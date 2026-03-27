from src.loader import load_all_pdfs
from src.chunker import chunk_pages
from src.vectorstore import build_index, load_index
from src.retriever import hybrid_retrieval
from src.generator import generate
from src.hallucination import check_groundedness, format_sources
from src.config import PDF_FOLDER, CHUNKING_STRATEGY


def index_documents(strategy=CHUNKING_STRATEGY):
    print("[pipeline] loading PDFs from " + PDF_FOLDER + "...")
    pages = load_all_pdfs(PDF_FOLDER)
    if not pages:
        print("[pipeline] no PDFs found. Add PDF files to " + PDF_FOLDER + " and retry.")
        return None, None
    print("[pipeline] " + str(len(pages)) + " pages loaded. chunking with strategy=" + strategy + "...")
    chunks = chunk_pages(pages, strategy=strategy)
    print("[pipeline] " + str(len(chunks)) + " chunks created. building FAISS index...")
    index, chunks = build_index(chunks)
    print("[pipeline] index ready.")
    return index, chunks


def query(question, index, chunks, api_key=None, top_k=5):
    results = hybrid_retrieval(question, index, chunks, top_k=top_k)
    answer = generate(question, results, api_key=api_key)
    groundedness = check_groundedness(answer, results)
    sources = format_sources(results)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "groundedness": groundedness,
        "retrieved": results
    }
