import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def embed_query(query):
    model = get_model()
    emb = model.encode([query], normalize_embeddings=True)
    return emb.astype(np.float32)
