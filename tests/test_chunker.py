"""Unit tests for the chunker — pure functions, no database needed."""

from datetime import UTC, datetime

import pytest

from app.services.chunker import chunk_article, chunk_text, make_header

# ~4,000 characters made of numbered sentences, so every boundary is visible.
LONG_TEXT = " ".join(f"Sentence number {i} talks about the news." for i in range(100))


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("A short article.") == ["A short article."]


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_chunks_respect_max_length_and_are_never_empty() -> None:
    chunks = chunk_text(LONG_TEXT, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 300 for chunk in chunks)


def test_chunks_end_at_sentence_boundaries() -> None:
    chunks = chunk_text(LONG_TEXT, max_chars=300, overlap=50)
    # Every chunk except the last should end on a full sentence.
    assert all(chunk.endswith(".") for chunk in chunks[:-1])


def test_neighbouring_chunks_overlap() -> None:
    chunks = chunk_text(LONG_TEXT, max_chars=300, overlap=50)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        # The start of each chunk repeats text from the end of the previous one.
        assert current[:20] in previous


def test_chunks_start_on_a_word_boundary() -> None:
    chunks = chunk_text(LONG_TEXT, max_chars=300, overlap=50)
    words = set(LONG_TEXT.split())
    assert all(chunk.split()[0] in words for chunk in chunks)


def test_whitespace_is_collapsed() -> None:
    assert chunk_text("Line one.\n\n   Line   two.") == ["Line one. Line two."]


def test_all_text_is_covered() -> None:
    chunks = chunk_text(LONG_TEXT, max_chars=300, overlap=50)
    assert chunks[0].startswith("Sentence number 0 ")
    assert chunks[-1].endswith("Sentence number 99 talks about the news.")


def test_overlap_too_large_is_rejected() -> None:
    # Would make the loop step backwards forever.
    with pytest.raises(ValueError):
        chunk_text(LONG_TEXT, max_chars=300, overlap=150)


def test_header_format() -> None:
    published = datetime(2023, 11, 27, 8, 45, 59, tzinfo=UTC)
    header = make_header("Mashable", published, "Cyber Monday deals")
    assert header == "Source: Mashable | Date: 2023-11-27 | Title: Cyber Monday deals"


def test_every_article_chunk_starts_with_the_header() -> None:
    published = datetime(2023, 11, 27, tzinfo=UTC)
    contents = chunk_article("TechCrunch", published, "FTX trial", LONG_TEXT)
    assert len(contents) > 1
    assert all(
        c.startswith("Source: TechCrunch | Date: 2023-11-27 | Title: FTX trial\n") for c in contents
    )
