"""
Template filling utilities for the Verbatim RAG system.

Handles aggregate placeholders ([DISPLAY_SPANS], [RELEVANT_SENTENCES]) and
per-fact placeholders ([FACT_1], ...). Supports toggling inline citation
numbering on or off so callers can choose between numbered excerpts or
clean text with citations handled separately.
"""

import re
from typing import Any, Dict, List


class TemplateFiller:
    ALLOWED_MODES = {"inline", "hidden"}

    def __init__(self, citation_mode: str = "inline"):
        self.set_citation_mode(citation_mode)

    def set_citation_mode(self, citation_mode: str) -> None:
        if citation_mode not in self.ALLOWED_MODES:
            raise ValueError(
                f"Unsupported citation mode: {citation_mode}. "
                f"Allowed values: {sorted(self.ALLOWED_MODES)}"
            )
        self.citation_mode = citation_mode

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        if not template:
            return ""

        citation_number_by_id = self._build_citation_number_map(display_spans, citation_spans)
        has_linked_citations = self._has_linked_citations(display_spans)
        citation_refs = ""
        if citation_spans and self.citation_mode == "inline" and not has_linked_citations:
            start_num = len(display_spans) + 1
            end_num = len(display_spans) + len(citation_spans)
            citation_refs = " ".join(f"[{i}]" for i in range(start_num, end_num + 1))

        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        if fact_pattern.search(template):
            filled = self._fill_per_fact_placeholders(
                template,
                display_spans,
                citation_spans,
                citation_number_by_id,
            )
            if "[CITATION_REFS]" in filled:
                filled = filled.replace("[CITATION_REFS]", citation_refs)
        else:
            filled = self._fill_aggregate_placeholders(
                template,
                display_spans,
                citation_refs,
                citation_number_by_id,
            )

        return filled.strip()

    def _fill_per_fact_placeholders(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
        citation_number_by_id: Dict[str, int],
    ) -> str:
        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        total_spans = display_spans + citation_spans

        def replace_fact(match):
            idx = int(match.group(1))
            if 1 <= idx <= len(total_spans):
                if idx <= len(display_spans):
                    span_data = display_spans[idx - 1]
                    return self._format_span(span_data, idx, citation_number_by_id)
                else:
                    return f"[{idx}]" if self.citation_mode == "inline" else ""
            return ""

        return fact_pattern.sub(replace_fact, template)

    def _fill_aggregate_placeholders(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_refs: str,
        citation_number_by_id: Dict[str, int],
    ) -> str:
        if display_spans:
            formatted = []
            for i, span_data in enumerate(display_spans, 1):
                block = self._format_span(span_data, i, citation_number_by_id)
                if block:
                    formatted.append(block)

            display_content = (
                "\n\n".join(formatted)
                if formatted
                else "No relevant information found in the provided documents."
            )
        else:
            display_content = "No relevant information found in the provided documents."

        filled = template.replace("[DISPLAY_SPANS]", display_content)
        filled = filled.replace("[RELEVANT_SENTENCES]", display_content)

        if "[CITATION_REFS]" in filled:
            filled = filled.replace("[CITATION_REFS]", citation_refs)

        return filled

    def _format_span(
        self,
        span_data: Dict[str, Any],
        index: int,
        citation_number_by_id: Dict[str, int],
    ) -> str:
        text = span_data.get("text", "")
        cleaned = text.strip()
        if not cleaned:
            return ""

        citation_refs = self._format_linked_citation_refs(span_data, citation_number_by_id)
        is_table = self._is_table(cleaned)

        if self.citation_mode == "inline":
            if is_table:
                if citation_refs:
                    return f"[{index}] {citation_refs}\n\n{cleaned}"
                return f"[{index}]\n\n{cleaned}"
            if citation_refs:
                return f"[{index}] {cleaned} {citation_refs}"
            return f"[{index}] {cleaned}"

        return cleaned

    @staticmethod
    def _has_linked_citations(display_spans: List[Dict[str, Any]]) -> bool:
        return any(span.get("citation_ids") for span in display_spans)

    @staticmethod
    def _build_citation_number_map(
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        start_num = len(display_spans) + 1
        citation_number_by_id: Dict[str, int] = {}
        for offset, citation_span in enumerate(citation_spans):
            citation_id = citation_span.get("citation_id")
            if citation_id:
                citation_number_by_id[str(citation_id)] = start_num + offset
        return citation_number_by_id

    def _format_linked_citation_refs(
        self,
        span_data: Dict[str, Any],
        citation_number_by_id: Dict[str, int],
    ) -> str:
        if self.citation_mode != "inline":
            return ""

        citation_ids = span_data.get("citation_ids", [])
        citation_numbers = [
            citation_number_by_id[str(citation_id)]
            for citation_id in citation_ids
            if str(citation_id) in citation_number_by_id
        ]
        if not citation_numbers:
            return ""

        return " ".join(f"[{citation_number}]" for citation_number in citation_numbers)

    @staticmethod
    def _is_table(text: str) -> bool:
        lines = [line for line in text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if "|" in line)
        if pipe_lines < 2:
            return False

        return pipe_lines >= len(lines) / 2

    @staticmethod
    def _is_table_content(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if "|" in line)

        return pipe_lines >= len(lines) / 2

    @staticmethod
    def ensure_placeholder(template: str, placeholder: str = "[DISPLAY_SPANS]") -> str:
        acceptable = ["[RELEVANT_SENTENCES]", "[DISPLAY_SPANS]", "[FACT_1]"]

        if any(p in template for p in acceptable):
            return template

        if template.endswith(":"):
            return template + f"\n\n{placeholder}"
        return template + f"\n\n{placeholder}"
