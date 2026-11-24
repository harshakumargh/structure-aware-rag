"""Chunking: split along document structure, keeping each chunk's heading path."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    headings: tuple[str, ...]
    content: str

    @property
    def section(self) -> str:
        return " > ".join(self.headings)

    @property
    def embed_text(self) -> str:
        """Heading path + content: what gets embedded."""
        return f"{self.section}\n\n{self.content}" if self.headings else self.content


def chunk_document(
    doc: DoclingDocument,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = 256,
) -> list[Chunk]:
    chunks = [
        Chunk(tuple(c.meta.headings or ()), c.text.strip())
        for c in HierarchicalChunker().chunk(doc)
        if c.text.strip()
    ]
    chunks = stitch(chunks)
    if count_tokens:
        chunks = [p for c in chunks for p in split_oversized(c, count_tokens, max_tokens)]
    return chunks


def stitch(chunks: list[Chunk]) -> list[Chunk]:
    """Rejoin a sentence cut in two by a page break: same section, no closing
    punctuation before the cut, and a lowercase start after it."""
    out: list[Chunk] = []
    for chunk in chunks:
        prev = out[-1] if out else None
        if (
            prev
            and prev.headings == chunk.headings
            and prev.content[-1] not in ".!?:;"
            and chunk.content[0].islower()
        ):
            out[-1] = Chunk(prev.headings, f"{prev.content} {chunk.content}")
        else:
            out.append(chunk)
    return out


def split_oversized(
    chunk: Chunk, count_tokens: Callable[[str], int], max_tokens: int
) -> list[Chunk]:
    """Split on sentence boundaries when a chunk is longer than the embedding
    model reads. A structural chunk has no size limit of its own, and the model
    would silently ignore the tail. Every part keeps the heading path."""
    if count_tokens(chunk.embed_text) <= max_tokens:
        return [chunk]

    parts: list[Chunk] = []
    current = ""
    for sentence in SENTENCE_END.split(chunk.content):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(Chunk(chunk.headings, candidate).embed_text) > max_tokens:
            parts.append(Chunk(chunk.headings, current))
            current = sentence
        else:
            current = candidate
    parts.append(Chunk(chunk.headings, current))
    return parts
