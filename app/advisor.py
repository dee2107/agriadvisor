"""Answer generation for AgriAdvisor."""

from __future__ import annotations

import os
from textwrap import shorten

from app.corpus import DOCUMENTS
from app.retrieval import LocalVectorIndex, SearchResult

SYSTEM_PROMPT = (
    "You are AgriAdvisor, a careful crop-management assistant. Give localized, "
    "actionable advice grounded only in the supplied extension notes. Mention when "
    "the farmer should verify chemical use with a local extension officer or label."
)

INDEX = LocalVectorIndex(DOCUMENTS)


def retrieve_context(question: str, top_k: int = 3) -> list[SearchResult]:
    return INDEX.search(question, top_k=top_k)


def build_grounded_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return (
            "I could not find enough matching extension guidance in the local knowledge "
            "base. Please add the crop, symptoms, location, crop stage, and recent weather "
            "so the advisory can be grounded."
        )

    lead = results[0]
    actions = _extract_action_sentences(results)
    bullets = "\n".join(f"- {sentence}" for sentence in actions[:6])
    source_list = ", ".join(f"{item.title} ({item.region})" for item in results)
    return (
        f"Based on the strongest match, this looks closest to guidance for "
        f"{lead.crop} in {lead.region}.\n\n"
        f"{bullets}\n\n"
        "Use any pesticide or fungicide only if it is registered for your crop and "
        "location, and follow the product label plus local extension thresholds.\n\n"
        f"Sources used: {source_list}."
    )


def answer_question(question: str, top_k: int = 3) -> tuple[str, list[SearchResult], bool]:
    results = retrieve_context(question, top_k=top_k)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return build_grounded_answer(question, results), results, False

    try:
        from openai import OpenAI

        context = "\n\n".join(
            f"Source: {result.title} | Region: {result.region} | Crop: {result.crop}\n"
            f"{result.text}"
            for result in results
        )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Extension notes:\n{context}\n\n"
                        "Answer in 5-8 concise bullets and cite source titles."
                    ),
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or build_grounded_answer(question, results), results, True
    except Exception:
        return build_grounded_answer(question, results), results, False


def _extract_action_sentences(results: list[SearchResult]) -> list[str]:
    sentences: list[str] = []
    for result in results:
        for sentence in result.text.split(". "):
            clean = sentence.strip().rstrip(".")
            if any(
                word in clean.lower()
                for word in [
                    "scout",
                    "inspect",
                    "remove",
                    "apply",
                    "avoid",
                    "use",
                    "treat",
                    "harvest",
                    "check",
                    "sample",
                    "split",
                ]
            ):
                sentences.append(shorten(clean + ".", width=190, placeholder="..."))
    return sentences or [shorten(results[0].text, width=190, placeholder="...")]

