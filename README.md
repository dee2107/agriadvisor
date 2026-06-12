# AgriAdvisor

LLM Crop Advisory Agent inspired by a B.Tech capstone project. AgriAdvisor helps smallholder farmers ask crop-management questions and receive grounded advice from a local agricultural-extension knowledge base.

## Features

- Local retrieval over extension-style crop notes with a lightweight TF-IDF vector index.
- Streamlit interface for farmer questions, retrieved source snippets, and grounded advisory output.
- Optional OpenAI generation when `OPENAI_API_KEY` is available; deterministic local fallback when it is not.
- Evaluation harness that tracks retrieval accuracy, grounding rate, and a simple hallucination proxy across test cases.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional LLM mode:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run streamlit_app.py
```

## Project Structure

```text
agriadvisor/
  app/
    advisor.py       # answer generation and optional OpenAI path
    corpus.py        # seed extension documents
    evaluation.py    # retrieval and grounding test harness
    retrieval.py     # lightweight TF-IDF vector search
  tests/
    test_retrieval.py
  streamlit_app.py
```

## Evaluation

```bash
pytest
```

The included tests cover retrieval quality, grounded answer formatting, and evaluation metrics.

