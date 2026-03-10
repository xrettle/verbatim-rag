"""Tests for verbatim_core.response_builder."""

from unittest.mock import MagicMock

from verbatim_core.response_builder import ResponseBuilder


class TestCreateHighlights:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_single_span(self):
        highlights = self.builder._create_highlights("The cat sat on the mat.", ["cat"])
        assert len(highlights) == 1
        assert highlights[0].text == "cat"
        assert highlights[0].start == 4
        assert highlights[0].end == 7

    def test_multiple_non_overlapping_spans(self):
        highlights = self.builder._create_highlights("The cat sat on the mat.", ["cat", "mat"])
        assert len(highlights) == 2
        texts = {h.text for h in highlights}
        assert texts == {"cat", "mat"}

    def test_span_not_found(self):
        highlights = self.builder._create_highlights("The cat sat.", ["dog"])
        assert len(highlights) == 0

    def test_overlapping_spans_prevented(self):
        text = "The big cat sat."
        highlights = self.builder._create_highlights(text, ["big cat", "cat"])
        # "cat" overlaps with "big cat", so only one should be created
        assert len(highlights) == 1
        assert highlights[0].text == "big cat"


class TestHasOverlap:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_no_overlap(self):
        assert not self.builder._has_overlap(0, 5, {(10, 15)})

    def test_overlap(self):
        assert self.builder._has_overlap(3, 8, {(5, 10)})

    def test_contained(self):
        assert self.builder._has_overlap(6, 8, {(5, 10)})

    def test_empty_regions(self):
        assert not self.builder._has_overlap(0, 5, set())


class TestCleanAnswer:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_removes_surrounding_double_quotes(self):
        assert self.builder.clean_answer('"Hello world"') == "Hello world"

    def test_removes_surrounding_single_quotes(self):
        assert self.builder.clean_answer("'Hello world'") == "Hello world"

    def test_converts_literal_newlines(self):
        assert "line1\nline2" == self.builder.clean_answer("line1\\nline2")

    def test_collapses_multiple_spaces(self):
        assert "a b" == self.builder.clean_answer("a   b")

    def test_collapses_many_newlines(self):
        result = self.builder.clean_answer("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_empty_string(self):
        assert self.builder.clean_answer("") == ""


class TestBuildResponse:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def _make_result(self, text, title="", source=""):
        r = MagicMock()
        r.text = text
        r.metadata = {"title": title, "source": source}
        r.title = title
        r.source = source
        return r

    def test_basic_response(self):
        results = [self._make_result("The cat sat on the mat.")]
        spans = {"The cat sat on the mat.": ["cat"]}

        response = self.builder.build_response(
            question="What animal?",
            answer="A cat.",
            search_results=results,
            relevant_spans=spans,
        )

        assert response.question == "What animal?"
        assert response.answer == "A cat."
        assert len(response.documents) == 1
        assert len(response.documents[0].highlights) == 1
        assert response.structured_answer.citations[0].text == "cat"

    def test_citation_numbering(self):
        results = [
            self._make_result("Doc one has alpha and beta."),
            self._make_result("Doc two has gamma."),
        ]
        spans = {
            "Doc one has alpha and beta.": ["alpha", "beta"],
            "Doc two has gamma.": ["gamma"],
        }

        response = self.builder.build_response(
            question="Q",
            answer="A",
            search_results=results,
            relevant_spans=spans,
            display_span_count=2,
        )

        citations = response.structured_answer.citations
        assert len(citations) == 3
        assert citations[0].number == 1
        assert citations[0].type == "display"
        assert citations[1].number == 2
        assert citations[1].type == "display"
        assert citations[2].number == 3
        assert citations[2].type == "reference"

    def test_no_spans(self):
        results = [self._make_result("Some text.")]
        spans = {"Some text.": []}

        response = self.builder.build_response(
            question="Q", answer="A", search_results=results, relevant_spans=spans
        )

        assert len(response.documents) == 1
        assert len(response.documents[0].highlights) == 0
        assert len(response.structured_answer.citations) == 0
