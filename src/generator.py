import requests
from src.config import GROQ_API_KEY, GROQ_MODEL, GROQ_URL, MAX_TOKENS, TEMPERATURE

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer ONLY based on the context provided. "
    "If the answer is not in the context, say: "
    "'I could not find this information in the provided documents.' "
    "Do not speculate or use outside knowledge."
)


def build_prompt(query, retrieved_chunks, max_context_chars=6000):
    context_parts = []
    total = 0
    for i, r in enumerate(retrieved_chunks):
        chunk = r["chunk"]
        text = chunk["text"][:1200]
        header = "[Source " + str(i + 1) + ": " + chunk["source"] + ", page " + str(chunk["page"]) + "]"
        entry = header + "\n" + text
        if total + len(entry) > max_context_chars:
            break
        context_parts.append(entry)
        total += len(entry)
    context = "\n\n".join(context_parts)
    return "Context:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer based strictly on the context above:"


def generate(query, retrieved_chunks, api_key=None):
    if not retrieved_chunks:
        return "No relevant context found in the documents for this query."
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("GROQ_API_KEY not set.")
    prompt = build_prompt(query, retrieved_chunks)
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("Groq error: " + response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()