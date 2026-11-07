import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.run_eval import judge
from rag import Retriever

path = Path("evaluation/results.json")
data = json.load(open(path, encoding="utf-8"))
questions = {q["id"]: q["question"] for q in json.load(open(next(Path("evaluation").glob("*questions*.json")), encoding="utf-8"))} if list(Path("evaluation").glob("*questions*.json")) else {}
retriever = Retriever()
for row in data["rows"]:
    if row["mode"] == "rag" and not row["abstained"] and row["type"] != "out_of_scope":
        q = questions.get(row["id"], "")
        row["supported"] = judge(q, row["answer"], retriever.search(q))
        print(row["id"], row["supported"], flush=True)
        time.sleep(5)
json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
