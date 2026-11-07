import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import abstained, answer, cited_ids, strip_marker
from evaluation.metrics import hit, summarize, valid_citations
from fetch_open_corpus import clean_wikitext
from rag import Retriever, chunk_text, parse_doc


def make_retriever(tmp_path):
    (tmp_path / "water.txt").write_text(
        "title: Water policy\nsource: Test A\ndate: 1965\n---\nWe must secure water supply through reservoirs and recycling.",
        encoding="utf-8",
    )
    (tmp_path / "media.txt").write_text(
        "title: Mass media\nsource: Test B\n---\nThe mass media should reinforce cultural values taught in schools.",
        encoding="utf-8",
    )
    (tmp_path / "housing.txt").write_text("Public housing gives citizens a stake in the nation.", encoding="utf-8")
    return Retriever(tmp_path)


def test_parse_doc_header_and_default_title(tmp_path):
    r = make_retriever(tmp_path)
    meta, body = parse_doc(tmp_path / "water.txt")
    assert meta["title"] == "Water policy" and meta["date"] == "1965" and body.startswith("We must")
    assert parse_doc(tmp_path / "housing.txt")[0]["title"] == "housing"


def test_chunk_text_overlaps_without_tiny_tail():
    assert chunk_text("a b c d e f g h i j", size=4, overlap=1) == ["a b c d", "d e f g", "g h i j"]
    assert chunk_text("") == []


def test_retriever_ranks_relevant_first_and_ignores_unrelated(tmp_path):
    r = make_retriever(tmp_path)
    assert r.search("how do we secure water supply")[0].title == "Water policy"
    assert r.search("cryptocurrency bitcoin") == []


def test_answer_puts_numbered_context_in_prompt(tmp_path):
    r = make_retriever(tmp_path)
    seen = {}

    def fake_llm(system, messages):
        seen["system"], seen["user"] = system, messages[-1]["content"]
        return "Water matters. [S1]"

    text, passages = answer("Why does water matter?", r, llm=fake_llm)
    assert "[S1] Water policy" in seen["user"] and "AI simulation" in seen["system"]
    assert passages and text == "Water matters. [S1]"


def test_baseline_has_no_context(tmp_path):
    r = make_retriever(tmp_path)
    seen = {}
    answer("Why does water matter?", r, llm=lambda s, m: seen.update(user=m[-1]["content"]) or "x", use_rag=False)
    assert seen["user"] == "Why does water matter?"


def test_markers_and_citations():
    assert abstained("  [NO SOURCE] Not covered.") and strip_marker("[NO SOURCE] Not covered.") == "Not covered."
    assert cited_ids("see [S1] and [S3]") == {1, 3}
    assert valid_citations("ok [S2]", 2) and not valid_citations("bad [S5]", 2) and not valid_citations("none", 2)


def test_hit_and_summary(tmp_path):
    r = make_retriever(tmp_path)
    assert hit(r.search("water supply"), "reservoirs") is True and hit([], "") is None
    rows = [
        {"mode": "rag", "type": "out_of_scope", "correct_abstain": True},
        {"mode": "rag", "type": "opinion", "correct_abstain": True, "hit": False},
        {"mode": "baseline", "type": "out_of_scope", "correct_abstain": False},
    ]
    s = summarize(rows)
    assert s["rag"]["refuses_out_of_scope_%"] == 100 and s["baseline"]["refuses_out_of_scope_%"] == 0
    assert s["rag"]["retrieval_hit_%"] == 0


def test_clean_wikitext():
    assert clean_wikitext("'''Lee''' said [[Singapore|the island]]{{cite web|x}}") == "Lee said the island"
