"""Retrieval evaluation: does the right passage come back, per chunking strategy?

A question is a hit at k when a top-k chunk contains its evidence phrase word for
word (ignoring case, spacing and punctuation). No LLM is involved.

    uv run python evals/run.py        add -v to list the questions each one misses
"""

import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pypdfium2

from structure_aware_rag.chunking import chunk_document
from structure_aware_rag.config import SAMPLE_SOURCE, Settings
from structure_aware_rag.embed import Embedder
from structure_aware_rag.parse import parse_document

FLAT_SIZE, FLAT_OVERLAP = 96, 16  # tokens; close to the structural chunks' average


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def flat_chunks(source: str, embedder: Embedder) -> list[str]:
    """Baseline: plain PDF text cut into fixed-size overlapping token windows."""
    pdf = pypdfium2.PdfDocument(urlopen(source).read())
    text = " ".join(" ".join(p.get_textpage().get_text_bounded() for p in pdf).split())
    offsets = embedder.model.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )["offset_mapping"]
    step = FLAT_SIZE - FLAT_OVERLAP
    windows = [offsets[i : i + FLAT_SIZE] for i in range(0, len(offsets) - FLAT_OVERLAP, step)]
    return [text[w[0][0] : w[-1][1]] for w in windows]


def rank_evidence(
    embed_texts: list[str], contents: list[str], questions: list[dict], embedder: Embedder
) -> list[int | None]:
    """Rank of the first chunk holding each question's evidence, or None if no
    chunk holds it at all."""
    chunks = np.array(embedder.encode(embed_texts))
    queries = np.array(embedder.encode([q["q"] for q in questions]))
    order = np.argsort(-(queries @ chunks.T), axis=1)  # MiniLM vectors are unit length
    haystack = [normalise(c) for c in contents]
    return [
        next((r for r, i in enumerate(row, 1) if normalise(q["evidence"]) in haystack[i]), None)
        for q, row in zip(questions, order)
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    verbose = "-v" in sys.argv
    data = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    questions = data["answerable"]
    embedder = Embedder(Settings().embed_model)

    raw = chunk_document(parse_document(SAMPLE_SOURCE, rescue=False)[0])
    chunks = chunk_document(
        parse_document(SAMPLE_SOURCE)[0], embedder.count_tokens, embedder.max_tokens
    )
    flat = flat_chunks(SAMPLE_SOURCE, embedder)
    contents = [c.content for c in chunks]
    with_path = [c.embed_text for c in chunks]

    runs = {
        f"fixed-size windows ({FLAT_SIZE}/{FLAT_OVERLAP})": (flat, flat),
        "structure, Docling as is": ([c.embed_text for c in raw], [c.content for c in raw]),
        "structure + rescue, content only": (contents, contents),
        "structure + rescue + heading path": (with_path, contents),
    }

    print(f"{len(questions)} questions\n")
    print(f"{'strategy':36}{'chunks':>7}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}{'lost':>6}")
    for name, (texts, bodies) in runs.items():
        ranks = rank_evidence(texts, bodies, questions, embedder)
        hits = [sum(1 for r in ranks if r and r <= k) for k in (1, 3, 5)]
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        lost = ranks.count(None)
        print(
            f"{name:36}{len(bodies):7}{hits[0]:7}{hits[1]:7}{hits[2]:7}{mrr:8.3f}{lost:6}"
        )
        if verbose:
            for q, r in zip(questions, ranks):
                if r != 1:
                    print(f"    rank {r}: {q['q']}")

    vectors = np.array(embedder.encode(with_path))

    def top_scores(texts: list[str]) -> list[tuple[float, str]]:
        best = (np.array(embedder.encode(texts)) @ vectors.T).max(axis=1)
        return sorted(zip(best.tolist(), texts))

    print("\ntop-1 similarity, five lowest answerable:")
    for score, text in top_scores([q["q"] for q in questions])[:5]:
        print(f"    {score:.3f}  {text}")
    print("top-1 similarity, out of scope:")
    for score, text in reversed(top_scores(data["out_of_scope"])):
        print(f"    {score:.3f}  {text}")


if __name__ == "__main__":
    main()
