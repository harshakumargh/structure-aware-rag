"""Runtime settings, read from environment variables or a local .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

SAMPLE_SOURCE = (
    "https://raw.githubusercontent.com/tnahddisttud/sample-doc/"
    "refs/heads/main/AtliqAI_HR_Policies.pdf"
)


@dataclass(frozen=True)
class Settings:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    # Qdrant: a server URL if given, otherwise embedded local mode at qdrant_path.
    qdrant_url: str | None = None
    qdrant_path: str = "./qdrant_data"
    collection: str = "documents"
    top_k: int = 5
    # Minimum cosine similarity for a chunk to count as relevant.
    score_threshold: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ
        return cls(
            embed_model=env.get("EMBED_MODEL", cls.embed_model),
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            qdrant_url=env.get("QDRANT_URL") or None,
            qdrant_path=env.get("QDRANT_PATH", cls.qdrant_path),
            collection=env.get("QDRANT_COLLECTION", cls.collection),
            top_k=int(env.get("TOP_K", cls.top_k)),
            score_threshold=float(env.get("SCORE_THRESHOLD", cls.score_threshold)),
        )
