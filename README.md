# rag-from-scratch

Added semantic embedding and FAISS vector storage.

## Install

```bash
pip install -r requirements.txt
```

## Components

- `src/loader.py` — PDF text extraction
- `src/chunker.py` — paragraph / sentence / fixed-size chunking
- `src/embedder.py` — dense embeddings via SentenceTransformers (all-MiniLM-L6-v2)
- `src/vectorstore.py` — FAISS flat inner-product index

## Why all-MiniLM-L6-v2

Fast, lightweight (22M params), strong on semantic similarity benchmarks.
Good tradeoff between quality and inference speed for a local pipeline.

## Usage

```python
from src.loader import load_all_pdfs
from src.chunker import chunk_pages
from src.vectorstore import build_index, search
from src.embedder import embed_query

pages = load_all_pdfs("data/pdfs")
chunks = chunk_pages(pages)
index, chunks = build_index(chunks)

q = embed_query("What is the main topic?")
results = search(q, index, chunks, top_k=5)
for r in results:
    print(str(r["score"]) + " | " + r["chunk"]["text"][:100])
```
