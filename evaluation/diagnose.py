import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import Retriever

def norm(s):
    s = s.lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return re.sub(r"\s+", " ", s)

rows = json.load(open("evaluation/results.json", encoding="utf-8"))["rows"]
qs = {q["id"]: q["question"] for q in json.load(open("evaluation/questions.json", encoding="utf-8"))}
corpus = norm(" ".join(p.read_text(encoding="utf-8", errors="ignore") for p in Path("data/raw").glob("*.txt")))
r = Retriever()

def claims(answer):
    items = []
    for q in re.findall(r"[\u201c\"]([^\u201c\u201d\"]{12,})[\u201d\"]", answer):
        for seg in re.split(r"\u2026|\.\.\.", q):
            if len(seg.split()) >= 4:
                items.append(("quote", norm(seg).strip(" ,.;")))
    for n in re.findall(r"\b\d[\d,]*\d\b", answer):
        items.append(("number", n))
    return items

tally = {}
for row in rows:
    if row["mode"] != "rag" or row["abstained"] or row["type"] == "out_of_scope":
        continue
    q = qs[row["id"]]
    ps = norm(" ".join(p.text for p in r.search(q, 4)))
    cl = claims(row["answer"])
    inp = [c for c in cl if c[1] in ps]
    corp = [c for c in cl if c[1] not in ps and c[1] in corpus]
    nowhere = [c for c in cl if c[1] not in corpus]
    if nowhere: diag = "NOT in corpus (model's own knowledge)"
    elif corp: diag = "in corpus but NOT retrieved (retrieval miss)"
    elif cl: diag = "all checkable claims are in retrieved passages (inference problem)"
    else: diag = "no quotes/numbers to check"
    tally.setdefault(diag, []).append((row["id"], row.get("supported")))
    print(f"Q{row['id']:>2} supported={str(row.get('supported')):5} checked={len(cl)} in_passages={len(inp)} corpus_only={len(corp)} nowhere={len(nowhere)} -> {diag}")
    for kind, text in nowhere + corp:
        print("      ", kind, "|", text[:90])
print()
for d, v in tally.items():
    print(d, "->", v)
if "show" in sys.argv:
    for i in (1, 9):
        print("\n##### Q", i, qs[i])
        for n, p in enumerate(r.search(qs[i], 4), 1):
            print(f"-- S{n} {p.title}\n{p.text[:500]}")