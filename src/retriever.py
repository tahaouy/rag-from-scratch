from src.embedder import embed_query
from src.vectorstore import search as faiss_search
from src.bm25 import build_bm25
from src.config import TOP_K, SIMILARITY_THRESHOLD, BM25_WEIGHT, SEMANTIC_WEIGHT


def normalize_scores(scores):
    if not scores:
        return scores
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def semantic_retrieval(query, index, chunks, top_k=TOP_K):
    q_emb = embed_query(query)
    results = faiss_search(q_emb, index, chunks, top_k=top_k * 2)
    filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
    return filtered[:top_k]


def hybrid_retrieval(query, index, chunks, top_k=TOP_K):
    q_emb = embed_query(query)
    sem_raw = faiss_search(q_emb, index, chunks, top_k=top_k * 3)

    bm25 = build_bm25(chunks)
    bm25_raw = bm25.search(query, top_k=top_k * 3)

    sem_scores = {r["chunk"]["chunk_id"]: r["score"] for r in sem_raw}
    bm25_scores_raw = {chunks[i]["chunk_id"]: s for i, s in bm25_raw}

    all_ids = list(set(sem_scores.keys()) | set(bm25_scores_raw.keys()))

    sem_norm = normalize_scores([sem_scores.get(cid, 0.0) for cid in all_ids])
    bm25_norm = normalize_scores([bm25_scores_raw.get(cid, 0.0) for cid in all_ids])

    combined = {}
    for i, cid in enumerate(all_ids):
        combined[cid] = SEMANTIC_WEIGHT * sem_norm[i] + BM25_WEIGHT * bm25_norm[i]

    sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    chunk_map = {c["chunk_id"]: c for c in chunks}
    results = []
    for cid, score in sorted_ids:
        if cid in chunk_map:
            results.append({"chunk": chunk_map[cid], "score": score})
    return results
