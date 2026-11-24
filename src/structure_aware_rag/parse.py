"""Parsing: source file or URL -> DoclingDocument with usable heading levels."""

import re
from collections import Counter

from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument

FURNITURE = (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER)
MIN_PROSE_WORDS = 15


def parse_document(source: str, rescue: bool = True) -> tuple[DoclingDocument, list[str]]:
    """Parse a document. Returns it with the text of any rescued blocks."""
    # Imported here because docling pulls in torch.
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    from hierarchical.postprocessor import ResultPostprocessor

    result = DocumentConverter().convert(source)
    # Before heading recovery, so rescued text is nested under its heading too.
    rescued = rescue_furniture(result.document) if rescue else []

    if result.input.format == InputFormat.PDF:
        # PDFs come back with flat headings. Passing the source lets the
        # postprocessor read levels from the PDF bookmarks; without it, it can't
        # reopen a URL input and falls back to numbering and font style.
        ResultPostprocessor(result, source=source).process()
    return result.document, rescued


def rescue_furniture(doc: DoclingDocument) -> list[str]:
    """Move prose mislabelled as a page header/footer back into the body.

    The layout model can tag a paragraph next to a page break as furniture, and
    chunkers skip furniture, so the text never reaches the index. Real running
    headers and footers are short or repeat across pages; prose is neither.
    """
    blocks = [t for t in doc.texts if t.label in FURNITURE]
    counts = Counter(_fingerprint(t.text) for t in blocks)

    rescued = []
    for item in blocks:
        is_prose = len(item.text.split()) >= MIN_PROSE_WORDS
        if is_prose and counts[_fingerprint(item.text)] == 1:
            item.label = DocItemLabel.TEXT
            item.content_layer = ContentLayer.BODY
            rescued.append(item.text)
    return rescued


def _fingerprint(text: str) -> str:
    """Letters only, so 'Page 1 of 5' and 'Page 2 of 5' count as a repeat."""
    return re.sub(r"[\W\d_]", "", text.lower())
