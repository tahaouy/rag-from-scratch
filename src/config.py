import os

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNKING_STRATEGY = "paragraph"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
SIMILARITY_THRESHOLD = 0.1
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
FAISS_INDEX_PATH = "data/faiss_index"
METADATA_PATH = "data/metadata.json"
PDF_FOLDER = "data/pdfs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_TOKENS = 512
TEMPERATURE = 0.2
