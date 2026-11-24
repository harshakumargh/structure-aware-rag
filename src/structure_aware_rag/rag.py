"""Grounded generation with section-level citations."""

import re
from dataclasses import dataclass, field

from structure_aware_rag.config import Settings
from structure_aware_rag.store import ChunkStore, Hit

FALLBACK = "I don't know. The indexed document doesn't cover this question."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Each passage starts with its number and section. Do not use outside knowledge. "
    "Cite the passages you used by number, like [1] or [2]. "
    f'If the context does not contain the answer, reply exactly: "{FALLBACK}"'
)

# gpt-oss models sometimes cite as 【1】 or 【1†L1-L3】 instead of [1].
CITATION = re.compile(r"[\[【](\d+)[^\]】]*[\]】]")


@dataclass(frozen=True)
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)

    @property
    def citations(self) -> list[tuple[int, str]]:
        """(number, section path) for each passage the answer cites. The path is
        looked up from the retrieved chunk, so the model can't invent one."""
        numbers = dict.fromkeys(int(n) for n in CITATION.findall(self.text))
        return [
            (n, self.hits[n - 1].chunk.section)
            for n in numbers
            if 1 <= n <= len(self.hits)
        ]


def build_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {hit.chunk.section}\n{hit.chunk.content}" for i, hit in enumerate(hits, 1)
    )


def answer(
    question: str, store: ChunkStore, llm, settings: Settings, section: str | None = None
) -> Answer:
    """Retrieve, then generate. With nothing above the threshold, returns the
    fallback without calling the LLM."""
    hits = store.search(question, settings.top_k, settings.score_threshold, section)
    if not hits:
        return Answer(FALLBACK)

    user_message = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    return Answer(response.choices[0].message.content.strip(), hits)
