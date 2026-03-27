from src.pipeline import query
from src.vectorstore import load_index


def evaluate(qa_pairs, index, chunks, api_key=None):
    results = []
    for pair in qa_pairs:
        question = pair["question"]
        expected = pair.get("expected_keywords", [])
        result = query(question, index, chunks, api_key=api_key)
        answer = result["answer"].lower()
        keyword_hits = [kw for kw in expected if kw.lower() in answer]
        recall = len(keyword_hits) / len(expected) if expected else None
        results.append({
            "question": question,
            "answer": result["answer"],
            "keyword_recall": recall,
            "grounded": result["groundedness"]["grounded"],
            "sources": result["sources"]
        })
    return results


def print_report(results):
    print("\n--- Evaluation Report ---")
    grounded_count = sum(1 for r in results if r["grounded"])
    recalls = [r["keyword_recall"] for r in results if r["keyword_recall"] is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print("Questions evaluated : " + str(len(results)))
    print("Grounded answers    : " + str(grounded_count) + "/" + str(len(results)))
    print("Avg keyword recall  : " + str(round(avg_recall * 100, 1)) + "%")
    print()
    for r in results:
        tag = "OK" if r["grounded"] else "WARN"
        recall_str = str(round(r["keyword_recall"] * 100)) + "%" if r["keyword_recall"] is not None else "n/a"
        print("[" + tag + "] recall=" + recall_str + " | " + r["question"][:60])
