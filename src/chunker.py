import re
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY


def chunk_by_paragraph(text, source, page):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        if len(para) > 30:
            chunks.append({
                "text": para,
                "source": source,
                "page": page,
                "chunk_id": source + "_p" + str(page) + "_c" + str(i),
                "strategy": "paragraph"
            })
    return chunks


def chunk_by_sentence(text, source, page):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    idx = 0
    for sent in sentences:
        if len(current) + len(sent) <= CHUNK_SIZE:
            current += " " + sent
        else:
            if current.strip():
                chunks.append({
                    "text": current.strip(),
                    "source": source,
                    "page": page,
                    "chunk_id": source + "_p" + str(page) + "_s" + str(idx),
                    "strategy": "sentence"
                })
                idx += 1
            current = sent
    if current.strip():
        chunks.append({
            "text": current.strip(),
            "source": source,
            "page": page,
            "chunk_id": source + "_p" + str(page) + "_s" + str(idx),
            "strategy": "sentence"
        })
    return chunks


def chunk_by_fixed(text, source, page):
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_id": source + "_p" + str(page) + "_f" + str(idx),
                "strategy": "fixed"
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


def chunk_pages(pages, strategy=None):
    strategy = strategy or CHUNKING_STRATEGY
    all_chunks = []
    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]
        if strategy == "paragraph":
            chunks = chunk_by_paragraph(text, source, page_num)
        elif strategy == "sentence":
            chunks = chunk_by_sentence(text, source, page_num)
        else:
            chunks = chunk_by_fixed(text, source, page_num)
        all_chunks.extend(chunks)
    return all_chunks
