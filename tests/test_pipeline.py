"""Offline tests: hand-built Docling documents, a fake embedder, in-memory Qdrant
and a fake LLM, so no model download, network access or API key is needed."""

from types import SimpleNamespace

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument

from structure_aware_rag.chunking import Chunk, chunk_document, split_oversized, stitch
from structure_aware_rag.config import Settings
from structure_aware_rag.parse import rescue_furniture
from structure_aware_rag.rag import FALLBACK, Answer, answer, build_context
from structure_aware_rag.store import ChunkStore, Hit, connect

PROSE = (
    "Upon receiving a complaint, the committee will acknowledge it within 3 working "
    "days and complete the inquiry within 60"
)


def add_footer(doc: DoclingDocument, text: str) -> None:
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text, content_layer=ContentLayer.FURNITURE)


@pytest.fixture
def split_paragraph_doc() -> DoclingDocument:
    """A paragraph cut by a page break whose first half is labelled a footer."""
    doc = DoclingDocument(name="test")
    doc.add_heading("Handbook", level=1)
    doc.add_heading("Complaints", level=2)
    add_footer(doc, PROSE)
    doc.add_text(label=DocItemLabel.TEXT, text="days. Both sides will be heard.")
    return doc


def test_mislabelled_prose_is_lost_without_rescue(split_paragraph_doc):
    chunks = chunk_document(split_paragraph_doc)
    assert [c.content for c in chunks] == ["days. Both sides will be heard."]


def test_rescue_restores_prose_and_rejoins_the_sentence(split_paragraph_doc):
    assert rescue_furniture(split_paragraph_doc) == [PROSE]

    chunks = chunk_document(split_paragraph_doc)
    assert chunks == [
        Chunk(("Handbook", "Complaints"), f"{PROSE} days. Both sides will be heard.")
    ]


def test_rescue_leaves_real_footers_alone():
    doc = DoclingDocument(name="test")
    add_footer(doc, "Page 1 of 2")
    for page in (1, 2):
        add_footer(
            doc,
            f"Page {page} This handbook is confidential and intended only for employees "
            "of the company and must not be shared",
        )
    assert rescue_furniture(doc) == []
    assert all(t.content_layer == ContentLayer.FURNITURE for t in doc.texts)


def test_chunks_carry_their_heading_path():
    doc = DoclingDocument(name="test")
    doc.add_heading("Handbook", level=1)
    doc.add_heading("Leave", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Leave is approved by managers.")
    doc.add_heading("Pay", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Salaries are paid monthly.")

    chunks = chunk_document(doc)

    assert [c.section for c in chunks] == ["Handbook > Leave", "Handbook > Pay"]
    assert chunks[0].embed_text == "Handbook > Leave\n\nLeave is approved by managers."


def test_stitch_keeps_separate_paragraphs_separate():
    finished = [Chunk(("A",), "First sentence."), Chunk(("A",), "second paragraph.")]
    other_section = [Chunk(("A",), "no full stop"), Chunk(("B",), "lowercase start.")]
    assert stitch(finished) == finished
    assert stitch(other_section) == other_section


def test_split_oversized_respects_limit_and_keeps_headings_and_text():
    def count_words(text: str) -> int:
        return len(text.split())

    sentences = [f"Sentence number {i} has exactly seven words." for i in range(20)]
    chunk = Chunk(("Handbook", "Long"), " ".join(sentences))

    parts = split_oversized(chunk, count_words, max_tokens=30)

    assert len(parts) > 1
    assert all(count_words(p.embed_text) <= 30 for p in parts)
    assert all(p.headings == chunk.headings for p in parts)
    assert " ".join(p.content for p in parts) == chunk.content


class FakeEmbedder:
    """One dimension per keyword, plus a constant so no vector is all zeros."""

    KEYWORDS = ("leave", "salary", "notice")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(k in t.lower()) for k in self.KEYWORDS] + [0.1] for t in texts]


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


CHUNKS = [
    Chunk(("Handbook", "Leave Policy"), "Employees get 12 days of leave."),
    Chunk(("Handbook", "Pay"), "The salary is paid monthly. No leave is deducted."),
    Chunk(("Handbook", "Exit"), "The notice period is 30 days."),
]


@pytest.fixture
def store():
    client = connect(Settings(qdrant_path=":memory:"))
    store = ChunkStore(client, FakeEmbedder(), "test")
    store.index(CHUNKS)
    yield store
    client.close()


def test_reindexing_replaces_instead_of_duplicating(store):
    store.index(CHUNKS[:2])
    assert store.chunks() == CHUNKS[:2]


def test_threshold_filters_irrelevant_chunks(store):
    assert len(store.search("leave", top_k=3)) == 3
    assert len(store.search("leave", top_k=3, score_threshold=0.5)) == 2
    assert store.search("bananas", top_k=3, score_threshold=0.5) == []


def test_section_filter_restricts_search(store):
    assert store.search("leave", top_k=1)[0].chunk == CHUNKS[0]
    hits = store.search("leave", top_k=3, section="Pay")
    assert [h.chunk for h in hits] == [CHUNKS[1]]


def test_resolve_section_ignores_case(store):
    assert store.resolve_section("leave policy") == "Leave Policy"
    assert store.resolve_section("missing") is None


def test_context_includes_section_names():
    context = build_context([Hit(CHUNKS[0], 0.9), Hit(CHUNKS[1], 0.5)])
    assert context.startswith("[1] Handbook > Leave Policy\nEmployees get 12 days")
    assert "\n\n---\n\n[2] Handbook > Pay\n" in context


def test_citations_resolve_to_sections_and_drop_unknown_numbers():
    hits = [Hit(CHUNKS[0], 0.9), Hit(CHUNKS[1], 0.5)]
    result = Answer("12 days [1]. Paid monthly 【2†L1-L2】【1】. Made up [7].", hits)
    assert result.citations == [(1, "Handbook > Leave Policy"), (2, "Handbook > Pay")]


def test_answer_falls_back_without_calling_llm(store):
    llm = FakeLLM("should not be used")
    result = answer("bananas", store, llm, Settings(score_threshold=0.5))
    assert result == Answer(FALLBACK)
    assert llm.requests == []


def test_answer_sends_context_and_returns_cited_sections(store):
    llm = FakeLLM("You get 12 days [1].")
    result = answer("How much leave?", store, llm, Settings(score_threshold=0.5))

    assert result.text == "You get 12 days [1]."
    assert result.citations == [(1, "Handbook > Leave Policy")]
    user_message = llm.requests[0]["messages"][1]["content"]
    assert "[1] Handbook > Leave Policy" in user_message
    assert user_message.endswith("Question: How much leave?")
