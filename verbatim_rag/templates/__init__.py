"""
Thin compatibility re-exports for templates from verbatim_core.
"""

from verbatim_core.templates import (
    ContextualTemplate,
    QuestionSpecificTemplate,
    RandomTemplate,
    StaticTemplate,
    TemplateFiller,
    TemplateManager,
    TemplateStrategy,
)

__all__ = [
    "TemplateStrategy",
    "StaticTemplate",
    "ContextualTemplate",
    "RandomTemplate",
    "QuestionSpecificTemplate",
    "TemplateManager",
    "TemplateFiller",
]
