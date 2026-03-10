"""Tests for verbatim_core.templates (static mode + filler)."""

import pytest

from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.templates.manager import TemplateManager
from verbatim_core.templates.static import StaticTemplate


class TestTemplateFiller:
    def setup_method(self):
        self.filler = TemplateFiller(citation_mode="inline")

    def test_aggregate_placeholder(self):
        template = "Here are findings:\n\n[DISPLAY_SPANS]"
        spans = [{"text": "Span one."}, {"text": "Span two."}]
        result = self.filler.fill(template, spans, [])
        assert "[1] Span one." in result
        assert "[2] Span two." in result

    def test_per_fact_placeholders(self):
        template = "Finding: [FACT_1]\nAlso: [FACT_2]"
        spans = [{"text": "Alpha."}, {"text": "Beta."}]
        result = self.filler.fill(template, spans, [])
        assert "[1] Alpha." in result
        assert "[2] Beta." in result

    def test_citation_refs(self):
        template = "[DISPLAY_SPANS]\n\nRefs: [CITATION_REFS]"
        display = [{"text": "Main point."}]
        citation = [{"text": "Extra ref."}]
        result = self.filler.fill(template, display, citation)
        assert "[2]" in result

    def test_hidden_citation_mode(self):
        filler = TemplateFiller(citation_mode="hidden")
        template = "[DISPLAY_SPANS]"
        spans = [{"text": "No numbers here."}]
        result = filler.fill(template, spans, [])
        assert "[1]" not in result
        assert "No numbers here." in result

    def test_empty_template(self):
        assert self.filler.fill("", [], []) == ""

    def test_no_spans(self):
        result = self.filler.fill("[DISPLAY_SPANS]", [], [])
        assert "No relevant information" in result

    def test_invalid_citation_mode(self):
        with pytest.raises(ValueError):
            TemplateFiller(citation_mode="bogus")


class TestIsTable:
    def test_table_detected(self):
        text = "| Col A | Col B |\n|-------|-------|\n| val1 | val2 |"
        assert TemplateFiller._is_table(text) is True

    def test_non_table(self):
        assert TemplateFiller._is_table("Just a plain sentence.") is False

    def test_single_line_with_pipe(self):
        assert TemplateFiller._is_table("one | two") is False


class TestEnsurePlaceholder:
    def test_already_has_placeholder(self):
        t = "Hello [DISPLAY_SPANS]"
        assert TemplateFiller.ensure_placeholder(t) == t

    def test_adds_placeholder(self):
        t = "Hello world"
        result = TemplateFiller.ensure_placeholder(t)
        assert "[DISPLAY_SPANS]" in result

    def test_respects_fact_placeholder(self):
        t = "Hello [FACT_1]"
        assert TemplateFiller.ensure_placeholder(t) == t


class TestStaticTemplate:
    def test_default_template(self):
        st = StaticTemplate()
        template = st.generate("Any question", ["span1"])
        assert "[DISPLAY_SPANS]" in template

    def test_custom_template(self):
        st = StaticTemplate(template="Custom: [DISPLAY_SPANS]")
        assert st.generate("Q", []) == "Custom: [DISPLAY_SPANS]"

    def test_fill_delegates_to_filler(self):
        st = StaticTemplate()
        result = st.fill(
            "Findings: [DISPLAY_SPANS]",
            [{"text": "A fact."}],
            [],
        )
        assert "A fact." in result

    def test_save_load_state(self):
        st = StaticTemplate(template="Custom [DISPLAY_SPANS]")
        state = st.save_state()
        assert state["type"] == "static"

        st2 = StaticTemplate()
        st2.load_state(state)
        assert st2.template == "Custom [DISPLAY_SPANS]"

    def test_create_academic(self):
        st = StaticTemplate.create_academic()
        assert "Literature" in st.template

    def test_create_brief(self):
        st = StaticTemplate.create_brief()
        assert "[DISPLAY_SPANS]" in st.template


class TestTemplateManager:
    def test_default_mode_is_static(self):
        tm = TemplateManager(llm_client=None, default_mode="static")
        assert tm.get_current_mode() == "static"

    def test_available_modes_without_llm(self):
        tm = TemplateManager(llm_client=None)
        modes = tm.get_available_modes()
        assert "static" in modes
        assert "contextual" not in modes

    def test_set_mode(self):
        tm = TemplateManager(llm_client=None)
        assert tm.set_mode("random") is True
        assert tm.get_current_mode() == "random"

    def test_set_unknown_mode(self):
        tm = TemplateManager(llm_client=None)
        assert tm.set_mode("nonexistent") is False

    def test_process_static(self):
        tm = TemplateManager(llm_client=None, default_mode="static")
        result = tm.process(
            question="What?",
            display_spans=[{"text": "Answer here."}],
            citation_spans=[],
        )
        assert "Answer here." in result

    def test_use_static_mode_custom_template(self):
        tm = TemplateManager(llm_client=None)
        tm.use_static_mode(template="Custom: [DISPLAY_SPANS]")
        result = tm.process("Q", [{"text": "Fact."}], [])
        assert "Custom:" in result
        assert "Fact." in result
