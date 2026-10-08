"""Split an article into overlapping chunks, each starting with a header.

Why chunk at all: an article is ~12,500 characters. Embedding it as one vector
blurs a dozen topics together, and sending it whole to the LLM wastes context.
Chunks of ~1,000 characters keep each vector about one thing.

Why a header on every chunk: questions in this dataset name the outlet and date
("...according to TechCrunch..."). The body text rarely repeats its own source,
so without the header a chunk cannot match those words. The header is part of
the text that is embedded, searched, shown to the LLM, and cited.

These are pure functions (text in, text out) so they are easy to unit test.
"""

from datetime import datetime

MAX_CHARS = 1000  # upper bound for one chunk's body text
OVERLAP = 150  # characters repeated between neighbouring chunks


def make_header(source: str, published_at: datetime, title: str) -> str:
    """e.g. 'Source: Mashable | Date: 2023-11-27 | Title: 200+ of the best deals ...'"""
    return f"Source: {source} | Date: {published_at.date().isoformat()} | Title: {title}"


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split `text` into pieces of at most `max_chars` characters.

    - Prefers to end a chunk at a sentence boundary (". "), so sentences stay whole.
    - Neighbouring chunks share up to `overlap` characters, so a fact that sits on
      a boundary still appears complete in at least one chunk.
    - Each chunk starts at the beginning of a word, never mid-word.
    """
    # Each step must move forward by more than `overlap`, or the loop never ends.
    # A chunk is at least max_chars // 2 long, so this condition guarantees it.
    if not 0 <= overlap < max_chars // 2:
        raise ValueError("overlap must be >= 0 and smaller than half of max_chars")

    text = " ".join(text.split())  # collapse newlines / repeated spaces into one space
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Look back for the last sentence end inside this window; use it only
            # if it keeps the chunk at least half full, otherwise cut at the limit.
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + max_chars // 2:
                end = sentence_end + 1  # keep the period

        chunks.append(text[start:end].strip())
        if end == len(text):
            break

        # Step back by `overlap`, then forward to the next word boundary.
        start = end - overlap
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1

    return chunks


def chunk_article(source: str, published_at: datetime, title: str, body: str) -> list[str]:
    """Header + newline + chunk text, for every chunk of the article body."""
    header = make_header(source, published_at, title)
    return [f"{header}\n{chunk}" for chunk in chunk_text(body)]
