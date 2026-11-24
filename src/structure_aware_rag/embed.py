"""Embeddings: a thin wrapper over a local sentence-transformers model."""


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text)["input_ids"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
