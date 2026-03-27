import argparse
import os
from src.pipeline import index_documents, query
from src.vectorstore import load_index


def main():
    parser = argparse.ArgumentParser(description="RAG from scratch")
    parser.add_argument("--index", action="store_true", help="Index PDF documents")
    parser.add_argument("--strategy", default="paragraph", choices=["paragraph", "sentence", "fixed"])
    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A session")
    parser.add_argument("--api-key", type=str, default=None, help="Groq API key")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GROQ_API_KEY", "")

    if args.index:
        index_documents(strategy=args.strategy)
        return

    index, chunks = load_index()

    if args.query:
        result = query(args.query, index, chunks, api_key=api_key, top_k=args.top_k)
        print("\nAnswer:\n" + result["answer"])
        if result["groundedness"]["warning"]:
            print("\n[WARNING] " + result["groundedness"]["warning"])
        print("\nSources: " + ", ".join(result["sources"]))
        return

    if args.interactive:
        print("RAG Q&A — type exit to quit\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            result = query(question, index, chunks, api_key=api_key, top_k=args.top_k)
            print("\nAnswer: " + result["answer"])
            if result["groundedness"]["warning"]:
                print("[WARNING] " + result["groundedness"]["warning"])
            print("Sources: " + ", ".join(result["sources"]) + "\n")


if __name__ == "__main__":
    main()
