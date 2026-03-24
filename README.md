# rag-from-scratch

Added BM25 (from scratch) and hybrid retrieval combining semantic + keyword matching.

## Install

```bash
pip install -r requirements.txt
```

## Retrieval strategies

**Semantic** — FAISS inner-product search on normalized embeddings (cosine similarity).
Good at finding conceptually related chunks even when exact words differ.

**BM25** — TF-IDF variant with length normalization, implemented from scratch.
Good at exact keyword matches, fast, no GPU needed.

**Hybrid** — 60% semantic + 40% BM25, scores normalized to [0,1] before combining.
Consistently outperforms either method alone on domain-specific documents.

## Chunking strategy comparison

| Strategy | Pros | Cons |
|---|---|---|
| Paragraph | Preserves semantic units | Variable chunk size |
| Sentence | Fine-grained, good recall | Loses context |
| Fixed-size | Predictable | Cuts mid-sentence |

Paragraph is the default because it keeps the natural semantic unit intact,
which leads to higher quality embeddings and more coherent retrieved context.
