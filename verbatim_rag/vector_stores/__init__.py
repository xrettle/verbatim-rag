"""
Vector storage for the Verbatim RAG system.

This package provides vector store implementations for hybrid search
using dense and sparse embeddings.
"""

from .base import SearchResult, VectorStore
from .hybrid_search import (
    convert_hits_to_results as _convert_hits_to_results,
)
from .hybrid_search import (
    merge_hybrid_results as _merge_hybrid_results,
)
from .hybrid_search import (
    normalize_weights as _normalize_weights,
)

# Re-export internal functions with underscore prefix for backwards compatibility
from .hybrid_search import (
    sanitize_hybrid_weights as _sanitize_hybrid_weights,
)
from .milvus_cloud import CloudMilvusStore
from .milvus_local import LocalMilvusStore

__all__ = [
    "VectorStore",
    "SearchResult",
    "LocalMilvusStore",
    "CloudMilvusStore",
    "_convert_hits_to_results",
    "_merge_hybrid_results",
    "_normalize_weights",
    "_sanitize_hybrid_weights",
]
