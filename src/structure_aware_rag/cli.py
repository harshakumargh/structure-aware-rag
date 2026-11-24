"""Command-line entry point: ingest, sections, search, ask."""

import argparse
import sys
from contextlib import closing

from structure_aware_rag.config import SAMPLE_SOURCE, Settings
from structure_aware_rag.store import ChunkStore, connect


def open_store(client, settings: Settings) -> ChunkStore:
    from structure_aware_rag.embed import Embedder

    return ChunkStore(client, Embedder(settings.embed_model), settings.collection)


def pick_section(store: ChunkStore, name: str | None) -> str | None:
    if name is None:
        return None
    section = store.resolve_section(name)
    if section is None:
        sys.exit(f"No section named '{name}'. Run `structure-rag sections` to list them.")
    return section


def cmd_ingest(args: argparse.Namespace, client, settings: Settings) -> None:
    from structure_aware_rag.chunking import chunk_document
    from structure_aware_rag.parse import parse_document

    doc, rescued = parse_document(args.source)
    store = open_store(client, settings)
    chunks = chunk_document(doc, store.embedder.count_tokens, store.embedder.max_tokens)
    if not chunks:
        sys.exit(f"No text found in {args.source}.")
    store.index(chunks)

    print(f"Indexed {len(chunks)} chunks from {args.source} into '{settings.collection}'.")
    for text in rescued:
        print(f"Rescued from page furniture: {text[:90]}...")


def cmd_sections(args: argparse.Namespace, client, settings: Settings) -> None:
    # Reads payloads only, so no embedding model is needed.
    seen: set[tuple[str, ...]] = set()
    for chunk in ChunkStore(client, None, settings.collection).chunks():
        for depth in range(1, len(chunk.headings) + 1):
            path = chunk.headings[:depth]
            if path not in seen:
                seen.add(path)
                print("  " * (depth - 1) + path[-1])


def cmd_search(args: argparse.Namespace, client, settings: Settings) -> None:
    """Show raw similarity scores, useful for picking SCORE_THRESHOLD."""
    store = open_store(client, settings)
    section = pick_section(store, args.section)
    for hit in store.search(args.question, args.k, section=section):
        mark = "+" if hit.score >= settings.score_threshold else "-"
        print(f"{mark} {hit.score:.3f}  {hit.chunk.section}")
    print(f"\n(+ passes the {settings.score_threshold} threshold, - is filtered out)")


def cmd_ask(args: argparse.Namespace, client, settings: Settings) -> None:
    from groq import Groq

    from structure_aware_rag.rag import answer

    store = open_store(client, settings)
    section = pick_section(store, args.section)
    result = answer(args.question, store, Groq(), settings, section)

    print(result.text)
    if result.citations:
        print("\nSources:")
        for number, path in result.citations:
            print(f"  [{number}] {path}")
    if args.show_context:
        for i, hit in enumerate(result.hits, 1):
            print(f"\n[{i}] {hit.score:.3f}  {hit.chunk.section}\n{hit.chunk.content}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="structure-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse, chunk, embed and index a document")
    ingest.add_argument("--source", default=SAMPLE_SOURCE, help="file path or URL")
    ingest.set_defaults(func=cmd_ingest)

    sections = sub.add_parser("sections", help="print the indexed heading tree")
    sections.set_defaults(func=cmd_sections)

    search = sub.add_parser("search", help="show retrieval scores without calling the LLM")
    search.add_argument("question")
    search.add_argument("--section", help="only search under this heading")
    search.add_argument("-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="answer a question, citing sections")
    ask.add_argument("question")
    ask.add_argument("--section", help="only search under this heading")
    ask.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    settings = Settings.from_env()
    with closing(connect(settings)) as client:
        args.func(args, client, settings)


if __name__ == "__main__":
    main()
