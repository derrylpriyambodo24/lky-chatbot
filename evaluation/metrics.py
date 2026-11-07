"""Scoring helpers for the evaluation."""
from chatbot import cited_ids


def hit(passages, expected: str):
    """Did retrieval return a passage containing the expected text? None if no expectation was set."""
    if not expected:
        return None
    return any(expected.lower() in f"{p.title} {p.source} {p.text}".lower() for p in passages)


def valid_citations(text: str, n_passages: int) -> bool:
    ids = cited_ids(text)
    return bool(ids) and all(1 <= i <= n_passages for i in ids)


def rate(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(100 * sum(values) / len(values)) if values else None


def summarize(rows) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        scope = [r for r in rs if r["type"] != "out_of_scope"]
        trap = [r for r in rs if r["type"] == "out_of_scope"]
        out[mode] = {
            "questions": len(rs),
            "answers_in_scope_%": rate(scope, "correct_abstain"),
            "refuses_out_of_scope_%": rate(trap, "correct_abstain"),
            "retrieval_hit_%": rate(rs, "hit"),
            "valid_citations_%": rate(rs, "citations_ok"),
            "supported_by_sources_%": rate(rs, "supported"),
        }
    return out
