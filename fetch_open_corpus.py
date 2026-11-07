"""Download a small open-licensed (CC BY-SA 4.0) corpus from Wikipedia and Wikiquote into data/raw/.

Run:  python fetch_open_corpus.py
Wikiquote has sourced, dated quotes by Lee Kuan Yew; Wikipedia gives third-person background.
Keep the attribution header in each file. Add your own legally usable documents next to them.
"""
import re
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "lky-rag-student-project/0.1 (educational use)"}
OUT = Path(__file__).parent / "data" / "raw"
WIKIPEDIA = [
    "Lee Kuan Yew",
    "History of the Republic of Singapore",
    "Separation of Singapore from Malaysia",
    "People's Action Party",
    "Politics of Singapore",
    "Economy of Singapore",
    "Education in Singapore",
]
WIKIQUOTE = ["Lee Kuan Yew"]


def clean_wikitext(text: str) -> str:
    for _ in range(5):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^[*:#]+\s*", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_wikipedia(title: str) -> str:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title},
        headers=HEADERS,
        timeout=30,
    )
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def fetch_wikiquote(title: str) -> str:
    r = requests.get(
        "https://en.wikiquote.org/w/api.php",
        params={"action": "parse", "page": title, "prop": "wikitext", "redirects": 1, "format": "json"},
        headers=HEADERS,
        timeout=30,
    )
    return clean_wikitext(r.json().get("parse", {}).get("wikitext", {}).get("*", ""))


def save(site: str, title: str, text: str) -> None:
    url = f"https://en.{site}.org/wiki/{title.replace(' ', '_')}"
    name = re.sub(r"\W+", "_", f"{site}_{title}").strip("_") + ".txt"
    header = f"title: {title} ({site.capitalize()})\nsource: {url} (CC BY-SA 4.0)\n---\n"
    (OUT / name).write_text(header + text, encoding="utf-8")
    print(f"saved {name} ({len(text.split())} words)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for site, titles, fetch in (("wikiquote", WIKIQUOTE, fetch_wikiquote), ("wikipedia", WIKIPEDIA, fetch_wikipedia)):
        for title in titles:
            text = fetch(title)
            if text:
                save(site, title, text)
            else:
                print(f"WARNING: no text for {site}: {title}")
            time.sleep(1)
