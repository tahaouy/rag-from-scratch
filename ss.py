from src.vectorstore import load_index

index, chunks = load_index()

print("Total chunks: " + str(len(chunks)))
for i, c in enumerate(chunks):
    print("[" + str(i) + "] page=" + str(c["page"]) + " | " + c["text"][:100])
    print()