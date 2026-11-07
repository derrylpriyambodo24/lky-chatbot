# What Would Lee Kuan Yew Do? — RAG Chatbot

A retrieval-augmented chatbot that answers questions in the voice of Lee Kuan Yew, grounded only in a corpus of his own statements. Every claim is meant to carry a citation to a retrieved passage, and the bot abstains (`[NO SOURCE]`) when the material does not cover the question instead of guessing.

Built with Streamlit and a free Groq-hosted LLM (any OpenAI-compatible API works).

## How it works

1. **Retrieve** — `rag.py` searches the corpus and returns the top-k passages (k=4 by default) for the question.
2. **Prompt** — `chatbot.py` numbers the passages `[S1]..[Sn]` and sends them to the LLM with the system prompt in `prompts/system_prompt.txt`.
3. **Answer or abstain** — the model answers in the first person using only those passages, citing `[Sx]` after each claim. If nothing relevant is retrieved, it must start with `[NO SOURCE]`, and the UI shows that no source covers the question.
4. **Baseline** — `prompts/baseline_prompt.txt` is a naive persona prompt with no retrieval, used only for evaluation.

```
app.py            Streamlit UI
chatbot.py        LLM call, prompt assembly, citation / abstain markers
rag.py            Retriever over the corpus
prompts/          system_prompt.txt, baseline_prompt.txt, judge_prompt.txt
evaluation/       questions.json, run_eval.py, rejudge.py, metrics.py, results*.json
```

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # then fill in LLM_API_KEY
python fetch_open_corpus.py   # downloads the corpus into data/raw/
python -m streamlit run app.py
```

Environment variables (`.env`):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | — | API key (Groq free tier works) |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model name |

## Evaluation

21 questions in `evaluation/questions.json`: 12 in-corpus, 3 opinion, 6 out-of-scope. Each is run through the RAG pipeline and the no-retrieval baseline.

```powershell
python -m evaluation.run_eval --judge
```

Metrics:

- **Answers in scope** — share of in-corpus and opinion questions that were answered rather than refused.
- **Refuses out-of-scope** — share of out-of-scope questions where the bot abstained.
- **Retrieval hit** — whether the expected source appears in the retrieved passages (in-corpus only).
- **Valid citations** — every `[Sx]` in the answer points to a passage that was actually retrieved.
- **Supported by sources** — an LLM judge (`prompts/judge_prompt.txt`) checks whether the answer is supported by the retrieved passages.

### Results

| Metric | Baseline (no retrieval) | RAG |
|---|---|---|
| Answers in scope | 100% | 87% (13/15) |
| Refuses out-of-scope | 0% (0/6) | **100% (6/6)** |
| Retrieval hit | n/a | 83% (10/12) |
| Valid citations | n/a | **100%** |
| Supported by sources (judge) | not judged | 62% overall (8/13); 55% in-corpus (6/11) |

The baseline never abstains and cannot cite, so it answers every out-of-scope question in character with no grounding. RAG refuses all six, which is the main behavioural gain.

### Findings and fixes

- **Citation checker bug.** The first run showed only about half of answers with valid citations. The cause was not the model: it wrote citations as `【S2】`, while `cited_ids` only recognised `[S2]`. The regex now accepts both forms and the prompt asks for plain brackets. Valid citations went to 100%.
- **Unsupported claims.** Early answers added closing summaries and outside commentary (for example "observers note…") and attached citations to them. I appended grounding rules to the system prompt: use only the passages, cite per claim, add no conclusions, answer partially when the sources cover only part of the question.
- **Remaining weakness: faithfulness.** About 45% of in-corpus answers are still judged unsupported. Inspected examples show the model drawing a conclusion the passages do not state (question 1 asserts Singapore is "a closed society" from passages that only define open and closed societies) or adding specific figures and dates that may not be in the retrieved text (question 9). A citation being present does not mean the claim is in the passage.

### Known limitations

- **Small sample.** 21 questions, 13 judged. One answer changes the percentages by 7 to 9 points, so the before/after difference in "supported" is within noise.
- **Model mismatch with earlier runs.** The daily token quota for `gpt-oss-120b` on Groq ran out during the rerun, so the final RAG and baseline runs and the judge used `gpt-oss-20b`. Numbers from the earlier `gpt-oss-120b` run (`evaluation/results_before_fix.json`) are not strictly comparable.
- **LLM judge.** The faithfulness judge is itself a small model and can be too strict or too lenient. Results were not hand-verified for every row.
- **Question 7** still abstains even though the right passage is retrieved, so the abstain rule is too conservative on partially covered questions.
- **Opinion questions (13-15)** are unstable between runs: the same question abstained in one run and was answered in another.
- **Retrieval hit** for questions 4 and 5 is marked as a miss while the answers were judged supported and correctly cited. The expected-source label is probably too narrow, so retrieval quality may be underestimated.

### Next steps

- Sentence-level citation verification: check each cited sentence against its passage, not just the citation ID.
- A stricter quote-or-close-paraphrase prompt, and optionally a second pass that removes unsupported sentences.
- Larger evaluation set, hand-checked labels for expected sources, and a stronger judge model.
- Tune the abstain threshold so partially covered questions get a partial answer instead of a refusal.

## Reproducing the evaluation

```powershell
python -m evaluation.run_eval --judge     # runs RAG and baseline, writes evaluation/results.json
python evaluation\rejudge.py              # re-judge saved RAG answers without regenerating them
```

If you hit Groq rate limits, set `LLM_MODEL` to another model for the session (`$env:LLM_MODEL="openai/gpt-oss-20b"`). `call_llm` retries on HTTP 429 with a 15-second wait.
