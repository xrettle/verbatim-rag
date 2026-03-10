"""
Compatibility layer: expose extractors from verbatim_core.
"""

from __future__ import annotations

from verbatim_core.extractors import (
    LLMSpanExtractor,
    ModelSpanExtractor,
    SemanticHighlightExtractor,
    SpanExtractor,
)

__all__ = [
    "SpanExtractor",
    "ModelSpanExtractor",
    "LLMSpanExtractor",
    "SemanticHighlightExtractor",
]
