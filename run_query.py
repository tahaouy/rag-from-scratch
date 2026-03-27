from src.pipeline import query
from src.vectorstore import load_index

GROQ_API_KEY = "key"

index, chunks = load_index()

question = "What is the role of the bias in a perceptron?"
result = query(question, index, chunks, api_key=GROQ_API_KEY)

print("\nAnswer:\n" + result["answer"])
print("\nSources: " + ", ".join(result["sources"]))
if result["groundedness"]["warning"]:
    print("[WARNING] " + result["groundedness"]["warning"])





