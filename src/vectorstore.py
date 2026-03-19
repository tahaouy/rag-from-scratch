import os
import json
import faiss
from src.config import FAISS_INDEX_PATH, METADATA_PATH
from src.embedder import embed_texts


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)
    print("[vectorstore] indexed " + str(len(chunks)) + " chunks, dim=" + str(dim))
    return index, chunks


def load_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Run build_index first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        chunks = json.load(f)
    return index, chunks


def search(query_embedding, index, chunks, top_k=5):
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({"chunk": chunks[idx], "score": float(score)})
    return results
