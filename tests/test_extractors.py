"""Tests for verbatim_core.extractors (LLMSpanExtractor only)."""

from unittest.mock import MagicMock

from verbatim_core.extractors import LLMSpanExtractor


class TestVerifySpans:
    def setup_method(self):
        self.extractor = LLMSpanExtractor(llm_client=MagicMock())

    def test_keeps_verbatim_spans(self):
        result = self.extractor._verify_spans(["cat", "mat"], "The cat sat on the mat.")
        assert result == ["cat", "mat"]

    def test_filters_non_verbatim_spans(self):
        result = self.extractor._verify_spans(["cat", "dog"], "The cat sat on the mat.")
        assert result == ["cat"]

    def test_strips_whitespace(self):
        result = self.extractor._verify_spans(["  cat  "], "The cat sat.")
        assert result == ["cat"]

    def test_empty_span_filtered(self):
        result = self.extractor._verify_spans(["", "  "], "Some text.")
        assert result == []


class TestExtractSpans:
    def test_empty_results(self):
        extractor = LLMSpanExtractor(llm_client=MagicMock())
        result = extractor.extract_spans("What?", [])
        assert result == {}

    def test_batch_mode(self):
        mock_client = MagicMock()
        mock_client.extract_spans.return_value = {
            "doc_0": ["cat sat on the mat"],
        }

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="batch", batch_size=5)

        result_obj = MagicMock()
        result_obj.text = "The cat sat on the mat."

        result = extractor.extract_spans("What animal?", [result_obj])
        assert "The cat sat on the mat." in result
        assert result["The cat sat on the mat."] == ["cat sat on the mat"]

    def test_individual_mode(self):
        mock_client = MagicMock()
        mock_client.extract_relevant_spans.return_value = ["The cat"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="individual")

        result_obj = MagicMock()
        result_obj.text = "The cat sat."

        result = extractor.extract_spans("What?", [result_obj])
        assert result["The cat sat."] == ["The cat"]

    def test_auto_mode_selects_batch_for_small_input(self):
        mock_client = MagicMock()
        mock_client.extract_spans.return_value = {"doc_0": ["span"]}

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="auto", batch_size=5)

        result_obj = MagicMock()
        result_obj.text = "Some text with span inside."

        extractor.extract_spans("Q?", [result_obj])
        mock_client.extract_spans.assert_called_once()

    def test_auto_mode_selects_individual_for_large_input(self):
        mock_client = MagicMock()
        mock_client.extract_relevant_spans.return_value = ["span"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="auto", batch_size=2)

        results = []
        for i in range(5):
            r = MagicMock()
            r.text = f"Document {i} with span content."
            results.append(r)

        extractor.extract_spans("Q?", results)
        assert mock_client.extract_relevant_spans.call_count == 5

    def test_batch_fallback_on_error(self):
        mock_client = MagicMock()
        mock_client.extract_spans.side_effect = Exception("API error")
        mock_client.extract_relevant_spans.return_value = ["fallback span"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="batch")

        result_obj = MagicMock()
        result_obj.text = "Text with fallback span."

        result = extractor.extract_spans("Q?", [result_obj])
        assert result["Text with fallback span."] == ["fallback span"]
