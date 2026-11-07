"""Compare the chatbot with RAG vs a naive persona baseline.

Run from the project root:  python evaluation/run_eval.py [--judge]
Fill "expected_source" in questions.json with text that should appear in the right passage
(only questions with a value are scored for retrieval hit).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from chatbot import abstained, answer, call_llm, format_context, load_prompt  # noqa: E402
from evaluation.metrics import hit, summarize, valid_citations  # noqa: E402
from rag import Retriever  # noqa: E402


def judge(question: str, text: str, passages) -> bool | None:
    prompt = f"Passages:\n{format_context(passages)}\n\nQuestion: {question}\n\nAnswer:\n{text}"
    raw = call_llm(load_prompt("judge_prompt.txt"), [{"role": "user", "content": prompt}], temperature=0)
    try:
        return bool(json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["supported"])
    except Exception:
        return None


def score(item, text, passages, mode, retriever, use_judge) -> dict:
    ab = abstained(text)
    row = {"id": item["id"], "type": item["type"], "mode": mode, "abstained": ab, "answer": text}
    row["correct_abstain"] = ab if item["type"] == "out_of_scope" else not ab
    if mode == "rag":
        row["hit"] = hit(passages, item.get("expected_source", ""))
        row["citations_ok"] = True if ab else valid_citations(text, len(passages))
    if use_judge and not ab:
        # the baseline is judged against what retrieval would have found for the same question
        row["supported"] = judge(item["question"], text, passages or retriever.search(item["question"]))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="also check faithfulness with an LLM judge")
    args = parser.parse_args()

    retriever = Retriever()
    if not retriever.chunks:
        sys.exit("No documents in data/raw/. Add documents or run fetch_open_corpus.py first.")
    questions = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for item in questions:
        for mode in ("rag",):
            text, passages = answer(item["question"], retriever, use_rag=(mode == "rag"))
            rows.append(score(item, text, passages, mode, retriever, args.judge))
            print(f"[{mode}] Q{item['id']} done")
            time.sleep(2.5)  # stay under free-tier rate limits

    summary = summarize(rows)
    (ROOT / "evaluation" / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    keys = list(next(iter(summary.values())))
    print("\n| metric | " + " | ".join(summary) + " |\n|---|" + "---|" * len(summary))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(summary[m][k]) for m in summary) + " |")
