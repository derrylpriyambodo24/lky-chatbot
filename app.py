"""What Would Lee Kuan Yew Do? - Streamlit chat UI."""
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chatbot import abstained, answer, strip_marker  # noqa: E402
from rag import Retriever  # noqa: E402

MAX_QUESTIONS = 20  # per session, protects the free API quota of a public demo

EXAMPLES = [
    "Is Singapore an open or a closed society? How do you see open debate?",
    "How did your government tackle the housing crisis?",
    "What did you mean by \"the cure for all this talk is a good dose of incompetent government\"?",
    "How does Singapore's education system treat madrasahs?",
]

st.set_page_config(page_title="What Would Lee Kuan Yew Do?", page_icon="🇸🇬", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
        background: linear-gradient(135deg, #FF5A5F 0%, #FF8A5B 60%, #FFB86B 100%);
        color: #fff;
        box-shadow: 0 6px 18px rgba(255, 90, 95, .25);
    }
    .hero h1 {margin: 0; font-size: 1.9rem; line-height: 1.2; color: #fff;}
    .hero p {margin: .4rem 0 0 0; opacity: .92; font-size: .98rem;}
    .badge-nosource {
        display: inline-block; padding: .25rem .7rem; border-radius: 999px;
        background: #FFF4D6; color: #8A5A00; border: 1px solid #F2C766;
        font-size: .82rem; margin-bottom: .5rem;
    }
    .src-card {
        border-left: 3px solid #FF5A5F; padding: .45rem .8rem; margin: .5rem 0;
        background: #FFF6EE; border-radius: 0 8px 8px 0;
    }
    .src-card .t {font-weight: 600;}
    .src-card .m {opacity: .65; font-size: .82rem;}
    .src-card .s {opacity: .85; font-size: .88rem; margin-top: .25rem;}
    [data-testid="stChatMessage"] {border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>🇸🇬 What Would Lee Kuan Yew Do?</h1>
      <p>Ask a question and get an answer built only from retrieved passages, with sources you can check.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


retriever = get_retriever()
if not retriever.chunks:
    st.error("No documents found in data/raw/. Add .txt files or run `python fetch_open_corpus.py`.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Your question is matched against the corpus.\n"
        "2. The top passages are given to the model as **[S1], [S2]...**\n"
        "3. The answer may only use those passages and must cite them.\n"
        "4. If nothing relevant is found, the bot says so instead of guessing."
    )
    st.caption(
        f"Corpus: {len({c.title for c in retriever.chunks})} documents, {len(retriever.chunks)} passages"
    )
    st.divider()
    st.subheader("Try a question")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

st.warning(
    "This is an AI simulation, not the real Lee Kuan Yew. Answers are generated from the retrieved "
    "passages shown under each reply and can still be wrong. Check the sources.",
    icon="⚠️",
)


def pretty_citations(text: str) -> str:
    """Show [S1] / 【S1】 as small code chips; the stored text stays untouched."""
    return re.sub(r"[\[【]S(\d+)[\]】]", r"`S\1`", text)


def show(msg: dict) -> None:
    avatar = "🇸🇬" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("no_source"):
            st.markdown(
                '<span class="badge-nosource">No matching source</span>', unsafe_allow_html=True
            )
            st.info("No source in my material covers this, so I will not guess.")
        st.markdown(pretty_citations(msg["content"]))
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, p in enumerate(msg["sources"], start=1):
                    where = " · ".join(x for x in (p["source"], p["date"]) if x)
                    st.markdown(
                        f'<div class="src-card"><div class="t">S{i} · {p["title"]}</div>'
                        f'<div class="m">{where}</div><div class="s">{p["snippet"]}</div></div>',
                        unsafe_allow_html=True,
                    )


if not st.session_state.messages:
    st.markdown("##### Not sure where to start? Pick an example in the sidebar, or type your own question below.")

for m in st.session_state.messages:
    show(m)

typed = st.chat_input("Ask Lee Kuan Yew anything...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    if sum(m["role"] == "user" for m in st.session_state.messages) >= MAX_QUESTIONS:
        st.warning(f"Demo limit reached ({MAX_QUESTIONS} questions per session). Refresh to start again.")
    else:
        user_msg = {"role": "user", "content": question}
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append(user_msg)
        show(user_msg)
        with st.spinner("Searching the sources and drafting an answer..."):
            try:
                text, passages = answer(question, retriever, history=history)
            except Exception as err:
                st.error(f"Could not get an answer: {err}")
                st.stop()
        reply = {
            "role": "assistant",
            "content": strip_marker(text),
            "no_source": abstained(text),
            "sources": [
                {"title": p.title, "source": p.source, "date": p.date, "snippet": p.text[:200] + "..."}
                for p in passages
            ],
        }
        st.session_state.messages.append(reply)
        show(reply)
