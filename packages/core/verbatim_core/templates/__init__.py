"""
Template system (RAG-agnostic copy) mirroring verbatim_rag.templates.
"""

from .base import TemplateStrategy
from .contextual import ContextualTemplate
from .filler import TemplateFiller
from .manager import TemplateManager
from .question_specific import QuestionSpecificTemplate
from .random import RandomTemplate
from .static import StaticTemplate

__all__ = [
    "TemplateStrategy",
    "StaticTemplate",
    "ContextualTemplate",
    "RandomTemplate",
    "QuestionSpecificTemplate",
    "TemplateManager",
    "TemplateFiller",
]
