"""Corpus loading, chunking, and BM25 retrieval."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(os.getenv("LKY_DATA_DIR", Path(__file__).parent / "data" / "raw"))
STOPWORDS = set(
    "a an and are as at be but by for from has have he i in is it its of on or that the their this "
    "to was were what when where which who why will with you your do does did how".split()
)
WORD = re.compile(r"[a-z0-9']+")


@dataclass
class Chunk:
    id: int
    title: str
    source: str
    date: str
    text: str


def tokenize(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def parse_doc(path: Path) -> tuple[dict, str]:
    """Optional header of 'key: value' lines (title, source, date) ended by a line with '---'."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta = {"title": path.stem.replace("_", " "), "source": "", "date": ""}
    start = 0
    for i, line in enumerate(lines[:12]):
        if line.strip() == "---":
            for header in lines[:i]:
                key, _, value = header.partition(":")
                if key.strip().lower() in meta and value.strip():
                    meta[key.strip().lower()] = value.strip()
            start = i + 1
            break
    return meta, "\n".join(lines[start:]).strip()


def chunk_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
    return chunks


class Retriever:
    def __init__(self, data_dir=DATA_DIR, size: int = 300, overlap: int = 50):
        self.chunks: list[Chunk] = []
        folder = Path(data_dir)
        for path in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md")):
            meta, body = parse_doc(path)
            for piece in chunk_text(body, size, overlap):
                self.chunks.append(Chunk(len(self.chunks) + 1, meta["title"], meta["source"], meta["date"], piece))
        self.tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokens) if self.chunks else None

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        """Top-k chunks that share at least one meaningful word with the query, best BM25 first."""
        query_tokens = tokenize(query)
        if not self.bm25 or not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        wanted = set(query_tokens)
        ranked = sorted(
            (i for i in range(len(self.chunks)) if wanted & set(self.tokens[i])),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [self.chunks[i] for i in ranked[:k]]
