import re


def check_groundedness(answer, retrieved_chunks):
    if "could not find" in answer.lower():
        return {"grounded": True, "warning": None}

    answer_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    context_text = " ".join(r["chunk"]["text"].lower() for r in retrieved_chunks)
    context_words = set(re.findall(r"\b\w{4,}\b", context_text))

    ungrounded = []
    for sent in answer_sentences:
        sent_words = set(re.findall(r"\b\w{4,}\b", sent.lower()))
        if sent_words and len(sent_words & context_words) / len(sent_words) < 0.2:
            ungrounded.append(sent)

    if ungrounded:
        return {
            "grounded": False,
            "warning": str(len(ungrounded)) + " sentence(s) may not be grounded in retrieved context.",
            "flagged": ungrounded
        }
    return {"grounded": True, "warning": None}


def format_sources(retrieved_chunks):
    seen = set()
    sources = []
    for r in retrieved_chunks:
        key = (r["chunk"]["source"], r["chunk"]["page"])
        if key not in seen:
            seen.add(key)
            sources.append(r["chunk"]["source"] + " (page " + str(r["chunk"]["page"]) + ")")
    return sources
