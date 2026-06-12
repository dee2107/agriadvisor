from app.advisor import answer_question
from app.evaluation import run_evaluation
from app.retrieval import LocalVectorIndex
from app.corpus import DOCUMENTS


def test_retrieves_rice_blast_document():
    index = LocalVectorIndex(DOCUMENTS)
    results = index.search("rice diamond leaf spots humid nights", top_k=2)
    assert results
    assert results[0].id == "rice-blast-integrated-management"


def test_answer_includes_sources():
    answer, results, used_llm = answer_question("fall armyworm holes in maize whorl", top_k=2)
    assert results[0].id == "maize-fall-armyworm"
    assert "Sources used" in answer
    assert used_llm is False


def test_evaluation_reports_quality_metrics():
    report = run_evaluation()
    assert report["total"] >= 5
    assert report["retrieval_accuracy"] >= 0.8
    assert 0 <= report["hallucination_proxy"] <= 1

