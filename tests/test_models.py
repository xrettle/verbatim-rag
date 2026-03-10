"""Tests for verbatim_core.models pydantic models."""

import pytest
from pydantic import ValidationError
from verbatim_core.models import (
    Citation,
    DocumentWithHighlights,
    Highlight,
    QueryResponse,
    StructuredAnswer,
)


class TestHighlight:
    def test_valid_highlight(self):
        h = Highlight(text="hello", start=0, end=5)
        assert h.text == "hello"
        assert h.start == 0
        assert h.end == 5

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError):
            Highlight(text="hello", start=5, end=5)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            Highlight(text="hello", start=5, end=3)

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            Highlight(text="hello", start=-1, end=5)

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            Highlight(text="", start=0, end=5)


class TestCitation:
    def test_valid_citation(self):
        c = Citation(text="span", doc_index=0, highlight_index=0, number=1, type="display")
        assert c.number == 1
        assert c.type == "display"

    def test_optional_fields(self):
        c = Citation(text="span", doc_index=0, highlight_index=0)
        assert c.number is None
        assert c.type is None


class TestDocumentWithHighlights:
    def test_defaults(self):
        d = DocumentWithHighlights(content="Some text")
        assert d.highlights == []
        assert d.title == ""
        assert d.metadata == {}

    def test_with_highlights(self):
        h = Highlight(text="Some", start=0, end=4)
        d = DocumentWithHighlights(content="Some text", highlights=[h])
        assert len(d.highlights) == 1


class TestQueryResponse:
    def test_full_construction(self):
        sa = StructuredAnswer(text="Answer text", citations=[])
        doc = DocumentWithHighlights(content="Doc content")
        qr = QueryResponse(
            question="What?",
            answer="Answer text",
            structured_answer=sa,
            documents=[doc],
        )
        assert qr.question == "What?"
        assert len(qr.documents) == 1

    def test_serialization_roundtrip(self):
        sa = StructuredAnswer(text="Answer", citations=[])
        doc = DocumentWithHighlights(content="Content")
        qr = QueryResponse(question="Q", answer="Answer", structured_answer=sa, documents=[doc])
        data = qr.model_dump()
        restored = QueryResponse.model_validate(data)
        assert restored.question == "Q"
