# rag-from-scratch

Building a Retrieval-Augmented Generation system from scratch without LangChain.

Every component is implemented manually: chunking, embedding, vector storage,
hybrid retrieval, generation and hallucination checking.

## Install

```bash
pip install -r requirements.txt
```

## Components so far

- `src/loader.py` — extracts text from PDF files page by page
- `src/chunker.py` — splits text using three strategies: paragraph, sentence, fixed-size

## Usage

```python
from src.loader import load_all_pdfs
from src.chunker import chunk_pages

pages = load_all_pdfs("data/pdfs")
chunks = chunk_pages(pages, strategy="paragraph")
print(str(len(chunks)) + " chunks from " + str(len(pages)) + " pages")
```

Drop PDF files into `data/pdfs/` before running.
