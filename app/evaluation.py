"""Simple retrieval and grounding evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass

from app.advisor import answer_question, retrieve_context


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_doc: str
    required_terms: tuple[str, ...]


EVAL_CASES = [
    EvalCase(
        question="My rice has diamond shaped leaf spots after humid nights. What should I do?",
        expected_doc="rice-blast-integrated-management",
        required_terms=("nitrogen", "fungicide", "resistant"),
    ),
    EvalCase(
        question="How do I control yellow stripes and rust pustules in wheat?",
        expected_doc="wheat-rust-advisory",
        required_terms=("resistant", "thresholds", "upper leaves"),
    ),
    EvalCase(
        question="Tomato leaves are water soaked before rain. Is this late blight?",
        expected_doc="tomato-late-blight",
        required_terms=("debris", "airflow", "fungicides"),
    ),
    EvalCase(
        question="Cotton bolls have holes and trap catches are rising.",
        expected_doc="cotton-pink-bollworm-ipm",
        required_terms=("pheromone", "residue", "thresholds"),
    ),
    EvalCase(
        question="Maize whorls have fresh ragged holes from armyworm.",
        expected_doc="maize-fall-armyworm",
        required_terms=("scout", "whorl", "threshold"),
    ),
    EvalCase(
        question="How can I reduce aflatoxin risk in groundnut after drought?",
        expected_doc="groundnut-moisture-stress",
        required_terms=("dry", "store", "sort"),
    ),
]


def run_evaluation() -> dict[str, object]:
    rows = []
    retrieval_hits = 0
    grounded_hits = 0

    for case in EVAL_CASES:
        results = retrieve_context(case.question, top_k=3)
        answer, _, _ = answer_question(case.question, top_k=3)
        top_ids = [result.id for result in results]
        retrieved = case.expected_doc in top_ids
        grounded = all(term.lower() in answer.lower() for term in case.required_terms[:2])
        retrieval_hits += int(retrieved)
        grounded_hits += int(grounded)
        rows.append(
            {
                "question": case.question,
                "expected_source": case.expected_doc,
                "top_sources": ", ".join(top_ids),
                "retrieval_hit": retrieved,
                "grounding_hit": grounded,
            }
        )

    total = len(EVAL_CASES)
    return {
        "total": total,
        "retrieval_accuracy": retrieval_hits / total,
        "grounding_rate": grounded_hits / total,
        "hallucination_proxy": 1 - (grounded_hits / total),
        "rows": rows,
    }

