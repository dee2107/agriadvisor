from __future__ import annotations

import streamlit as st

from app.advisor import answer_question
from app.corpus import DOCUMENTS
from app.evaluation import run_evaluation

st.set_page_config(page_title="AgriAdvisor", page_icon="A", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(120deg, rgba(247,244,236,.96), rgba(240,248,236,.92)),
            url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1180px; }
    .hero {
        padding: 1rem 0 1.25rem;
        max-width: 860px;
    }
    .hero h1 {
        color: #173d2a;
        font-size: 3rem;
        line-height: 1.05;
        margin-bottom: .4rem;
    }
    .hero p {
        color: #273b30;
        font-size: 1.05rem;
        max-width: 760px;
    }
    .source-card {
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(35,86,55,.18);
        border-radius: 8px;
        padding: .9rem 1rem;
        margin-bottom: .65rem;
    }
    .source-title {
        color: #173d2a;
        font-weight: 700;
        margin-bottom: .25rem;
    }
    .source-meta {
        color: #65756a;
        font-size: .85rem;
        margin-bottom: .4rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(35,86,55,.18);
        border-radius: 8px;
        padding: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AgriAdvisor</h1>
        <p>LLM crop advisory agent that retrieves extension guidance, grounds each answer in source notes, and tracks retrieval quality through an evaluation harness.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.subheader("Ask a crop-management question")
    question = st.text_area(
        "Question",
        value="My rice leaves have diamond-shaped spots after humid nights. What should I do?",
        label_visibility="collapsed",
        height=120,
    )
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)
    ask = st.button("Generate advisory", type="primary", use_container_width=True)

    if ask or question:
        answer, results, used_llm = answer_question(question, top_k=top_k)
        st.markdown("#### Grounded advisory")
        st.info("OpenAI generation enabled" if used_llm else "Local grounded fallback enabled")
        st.markdown(answer)

        st.markdown("#### Retrieved sources")
        for result in results:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{result.title}</div>
                    <div class="source-meta">{result.crop.title()} | {result.region} | score {result.score:.3f}</div>
                    <div>{result.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    st.subheader("Knowledge base")
    st.metric("Extension notes", len(DOCUMENTS))
    st.metric("Mode", "RAG + citations")
    st.metric("Optional LLM", "OPENAI_API_KEY")

    with st.expander("Run evaluation harness", expanded=True):
        report = run_evaluation()
        a, b, c = st.columns(3)
        a.metric("Test cases", report["total"])
        b.metric("Retrieval", f"{report['retrieval_accuracy']:.0%}")
        c.metric("Grounding", f"{report['grounding_rate']:.0%}")
        st.dataframe(report["rows"], use_container_width=True, hide_index=True)
