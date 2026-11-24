"""Vector store: index chunks in Qdrant, search with threshold and section filter."""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from structure_aware_rag.chunking import Chunk
from structure_aware_rag.config import Settings


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def connect(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    if settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=settings.qdrant_path)


class ChunkStore:
    def __init__(self, client: QdrantClient, embedder, collection: str):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def index(self, chunks: list[Chunk]) -> None:
        """Embed chunks and rebuild the collection, so re-ingesting a document
        never leaves stale or duplicate points behind."""
        vectors = self.embedder.encode([c.embed_text for c in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={"headings": list(chunk.headings), "content": chunk.content},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
            wait=True,
        )

    def search(
        self,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
        section: str | None = None,
    ) -> list[Hit]:
        """Top-k chunks above the threshold. `section` restricts the search to
        chunks that have that heading anywhere in their path."""
        query_filter = None
        if section:
            # MatchValue on a list field matches any element of the list.
            condition = FieldCondition(key="headings", match=MatchValue(value=section))
            query_filter = Filter(must=[condition])

        response = self.client.query_points(
            self.collection,
            query=self.embedder.encode([query])[0],
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        return [Hit(_to_chunk(p.payload), p.score) for p in response.points]

    def chunks(self) -> list[Chunk]:
        """Every indexed chunk, in document order."""
        points, offset = [], None
        while True:
            page, offset = self.client.scroll(self.collection, limit=256, offset=offset)
            points.extend(page)
            if offset is None:
                break
        return [_to_chunk(p.payload) for p in sorted(points, key=lambda p: p.id)]

    def resolve_section(self, name: str) -> str | None:
        """Exact heading for a case-insensitive name, or None if not indexed."""
        for chunk in self.chunks():
            for heading in chunk.headings:
                if heading.lower() == name.lower():
                    return heading
        return None


def _to_chunk(payload: dict) -> Chunk:
    return Chunk(tuple(payload["headings"]), payload["content"])
