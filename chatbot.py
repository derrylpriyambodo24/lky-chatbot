"""LLM call, persona prompt assembly, and answer markers."""
import os
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
NO_SOURCE = "[NO SOURCE]"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def call_llm(system: str, messages: list[dict], temperature: float = 0.3) -> str:
    """Any OpenAI-compatible API (Groq by default). Configure in .env."""
    from openai import APIStatusError, OpenAI

    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is missing. Copy .env.example to .env and add your key.")
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    for attempt in range(8):
        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
                messages=[{"role": "system", "content": system}] + messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except APIStatusError as err:
            if err.status_code == 429 and attempt < 7:
                time.sleep(15)
                continue
            raise RuntimeError(f"API error {err.status_code}: {err.response.text}") from err


def format_context(passages) -> str:
    if not passages:
        return "(no passages found)"
    blocks = []
    for i, p in enumerate(passages, start=1):
        where = ", ".join(x for x in (p.source, p.date) if x)
        blocks.append(f"[S{i}] {p.title} ({where})\n{p.text}")
    return "\n\n".join(blocks)


def answer(question, retriever, llm=call_llm, history=None, use_rag=True, k=4):
    """Return (answer_text, passages). use_rag=False is the naive persona baseline for evaluation."""
    history = (history or [])[-6:]
    if use_rag:
        passages = retriever.search(question, k)
        system = load_prompt("system_prompt.txt")
        user = f"Context passages:\n{format_context(passages)}\n\nQuestion: {question}"
    else:
        passages = []
        system = load_prompt("baseline_prompt.txt")
        user = question
    return llm(system, history + [{"role": "user", "content": user}]).strip(), passages


def abstained(text: str) -> bool:
    return text.lstrip().startswith(NO_SOURCE)


def strip_marker(text: str) -> str:
    return text.replace(NO_SOURCE, "", 1).strip() if abstained(text) else text


def cited_ids(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"[\[【]S(\d+)[\]】]", text)}