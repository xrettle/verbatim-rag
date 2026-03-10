"""Tests for verbatim_core.transform."""

from unittest.mock import MagicMock

import pytest

from verbatim_core.transform import VerbatimTransform, _coerce_context_to_results
from verbatim_core.universal_document import UniversalDocument


class TestCoerceContextToResults:
    def test_dict_with_content_key(self):
        context = [{"content": "Hello world", "title": "Doc 1"}]
        results = _coerce_context_to_results(context)
        assert len(results) == 1
        assert results[0].text == "Hello world"
        assert results[0].metadata["title"] == "Doc 1"

    def test_dict_with_text_key(self):
        context = [{"text": "Hello world"}]
        results = _coerce_context_to_results(context)
        assert results[0].text == "Hello world"

    def test_object_with_text_attribute(self):
        # UniversalDocument has .content, not .text — test with a mock
        obj = MagicMock()
        obj.text = "Hello from object"
        obj.metadata = {"key": "val"}
        results = _coerce_context_to_results([obj])
        assert results[0].text == "Hello from object"

    def test_bad_input_raises(self):
        with pytest.raises(TypeError):
            _coerce_context_to_results([42])

    def test_missing_content_raises(self):
        with pytest.raises(ValueError):
            _coerce_context_to_results([{"title": "no content"}])

    def test_multiple_items(self):
        context = [
            {"content": "First", "title": "A"},
            {"content": "Second", "title": "B"},
        ]
        results = _coerce_context_to_results(context)
        assert len(results) == 2
        assert results[0].id == "ctx_0"
        assert results[1].id == "ctx_1"


class TestVerbatimTransform:
    def test_transform_with_mocked_extractor(self):
        mock_extractor = MagicMock()
        mock_extractor.extract_spans.return_value = {"The study found X.": ["found X"]}

        from verbatim_core.templates.manager import TemplateManager

        tm = TemplateManager(llm_client=None, default_mode="static")

        vt = VerbatimTransform(
            llm_client=MagicMock(),
            extractor=mock_extractor,
            template_manager=tm,
        )

        response = vt.transform(
            question="What was found?",
            context=[{"content": "The study found X."}],
        )

        assert response.question == "What was found?"
        assert "found X" in response.answer

    def test_transform_empty_context(self):
        mock_extractor = MagicMock()
        mock_extractor.extract_spans.return_value = {}

        from verbatim_core.templates.manager import TemplateManager

        tm = TemplateManager(llm_client=None, default_mode="static")

        vt = VerbatimTransform(
            llm_client=MagicMock(),
            extractor=mock_extractor,
            template_manager=tm,
        )

        response = vt.transform(
            question="What?",
            context=[{"content": "No relevant info here."}],
        )

        assert response.question == "What?"
        assert "No relevant information" in response.answer


class TestUniversalDocument:
    def test_from_text(self):
        doc = UniversalDocument.from_text("Hello", title="T", source="S")
        assert doc.content == "Hello"
        assert doc.title == "T"

    def test_from_dict_content_key(self):
        doc = UniversalDocument.from_dict({"content": "Hello", "title": "T"})
        assert doc.content == "Hello"

    def test_from_dict_text_key(self):
        doc = UniversalDocument.from_dict({"text": "Hello"})
        assert doc.content == "Hello"

    def test_from_dict_missing_content_raises(self):
        with pytest.raises(ValueError):
            UniversalDocument.from_dict({"title": "no content"})

    def test_from_dict_bad_type_raises(self):
        with pytest.raises(TypeError):
            UniversalDocument.from_dict("not a dict")

    def test_to_context_roundtrip(self):
        doc = UniversalDocument(content="Hello", title="T", source="S", metadata={"k": "v"})
        ctx = doc.to_context()
        assert ctx["content"] == "Hello"
        doc2 = UniversalDocument.from_dict(ctx)
        assert doc2.content == doc.content
