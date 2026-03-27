# rag-from-scratch

RAG system built from scratch — PDF chunking, FAISS vector search, BM25 hybrid retrieval and LLM generation without LangChain
Ask questions on PDF documents. Every component implemented manually.

Generation backend: Groq API (free, fast, no local GPU needed).

## Architecture

```
PDF files (data/pdfs/)
        |
    loader.py           <- extract text page by page with pypdf
        |
    chunker.py          <- split into chunks (paragraph / sentence / fixed)
        |
    embedder.py         <- dense embeddings via SentenceTransformers
        |
    vectorstore.py      <- FAISS flat index (cosine similarity)
        |
    bm25.py             <- BM25 lexical index (implemented from scratch)
        |
    retriever.py        <- hybrid search: 60% semantic + 40% BM25
        |
    generator.py        <- answer generation via Groq API (llama3-8b-8192)
        |
    hallucination.py    <- groundedness check on generated answer
        |
    pipeline.py         <- end-to-end orchestration
```

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at https://console.groq.com.
Paste it in `run_query.py` and `run_index.py`.

## Usage

**Step 1 — drop PDFs into data/pdfs/**

**Step 2 — index:**
```bash
python run_index.py
```

**Step 3 — query:**
```bash
python run_query.py
```

Or use the CLI:
```bash
python cli.py --index
python cli.py --query "What are the main conclusions?" --api-key YOUR_KEY
python cli.py --interactive --api-key YOUR_KEY
```

## Design decisions

**Why no LangChain?** LangChain abstracts chunking, embedding, retrieval and prompting
into single-line calls. Building each component manually forces understanding of failure
modes: why a chunk boundary breaks an answer, why BM25 outperforms semantic search on rare
proper nouns, why temperature affects hallucination rate.

**Why hybrid retrieval?** Pure semantic search fails on exact terminology and acronyms.
Pure BM25 fails on paraphrased queries. The weighted hybrid consistently outperforms both.

**Why paragraph chunking?** Sentence chunks lose context. Fixed chunks break mid-sentence.
Paragraph chunks preserve the natural semantic unit the embedding model was trained on.

**Why Groq?** Free, no local GPU required, sub-second latency, OpenAI-compatible API.
Runs llama3-8b-8192 which is strong enough for document Q&A tasks.

## Stack

Python · pypdf · SentenceTransformers · FAISS · BM25 (from scratch) · Groq API
