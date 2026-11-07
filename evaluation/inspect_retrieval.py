"""Show what retrieval finds for each question, to help you fill expected_source.

Run from the project root:  python evaluation/inspect_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import Retriever  # noqa: E402

retriever = Retriever()
questions = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
for item in questions:
    if item["type"] == "out_of_scope":
        continue
    print(f"\n=== Q{item['id']} ({item['type']}): {item['question']}")
    print(f"expected_source now: {item['expected_source']!r}")
    for i, p in enumerate(retriever.search(item["question"], 4), start=1):
        print(f"  [{i}] {p.title}: {p.text[:160]}...")
