import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import Retriever

STOP = {"which","their","there","these","those","would","could","about","after","other","being","where","while","under","through","between","within","because","should","without","singapore","government"}
CITE = re.compile(r"[\[\u3010]S(\d+)[\]\u3011]")
def words(s):
    return {w[:6] for w in re.findall(r"[a-z]{5,}", s.lower()) if w not in STOP}

rows = json.load(open("evaluation/results.json", encoding="utf-8"))["rows"]
qs = {q["id"]: q["question"] for q in json.load(open("evaluation/questions.json", encoding="utf-8"))}
r = Retriever()
tot = bad_total = miscited = nowhere = 0
per_row = []
for row in rows:
    if row["mode"] != "rag" or row["abstained"] or row["type"] == "out_of_scope":
        continue
    ps = r.search(qs[row["id"]], 4)
    sents = []
    for f in re.split(r"(?<=[.!?])\s+", row["answer"].replace("\n", " ")):
        if sents and re.fullmatch(r"(?:[\[\u3010]S\d+[\]\u3011][,\s]*)+", f.strip()):
            sents[-1] += " " + f
        else:
            sents.append(f)
    n = nb = 0
    for s in sents:
        ids = [int(x) for x in CITE.findall(s)]
        if not ids:
            continue
        w = words(CITE.sub("", s))
        if len(w) < 4:
            continue
        n += 1
        cov = [len(w & words(ps[i-1].text)) / len(w) if 1 <= i <= len(ps) else 0 for i in ids]
        anyc = len(w & words(" ".join(p.text for p in ps))) / len(w)
        if max(cov) < 0.4:
            nb += 1
            kind = "MISCITED (support is in another passage)" if anyc >= 0.5 else "LOW SUPPORT IN ALL 4 PASSAGES"
            if anyc >= 0.5: miscited += 1
            else: nowhere += 1
            print(f"Q{row['id']:>2} cited={ids} cov={[round(c,2) for c in cov]} any={anyc:.2f} {kind}\n      {s.strip()[:110]}")
    tot += n; bad_total += nb
    per_row.append((row["id"], row.get("supported"), nb, n))
print(f"\ncited sentences checked: {tot}, weakly supported by their cited passage: {bad_total} (miscited: {miscited}, low support everywhere: {nowhere})")
for i, sup, nb, n in per_row:
    print(f"Q{i:>2} judge_supported={str(sup):5} weak_sentences={nb}/{n}")