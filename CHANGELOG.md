# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-03-22

### Added
- Linked citations: display spans can now reference specific citation sources via `citation_ids`, enabling per-claim traceability
- When linked citations are present, flat `[CITATION_REFS]` is suppressed to avoid double-referencing
- 3 new tests for linked citation behavior

### Changed
- `TemplateFiller._format_span` now takes full span data dict instead of raw text, enabling richer formatting
- Extracted `_get_template_inputs` and `_build_citation_number_map` as clean static methods

## [0.2.1] - 2026-03-16

### Added
- Fuzzy span matching (`span_match_mode="fuzzy"`) using rapidfuzz for documents with OCR artifacts or encoding issues
- Custom extraction prompts via `extraction_prompt` and `system_prompt` parameters (Jinja2 syntax)
- Prompt bank: built-in prompt templates in `verbatim_core.prompts` with `load_prompt()` and `list_prompts()`
- All internal prompts now load from the prompt bank -- no hardcoded prompts in Python code

### Changed
- Added `rapidfuzz>=3.0.0` and `jinja2>=3.0.0` to verbatim-core base dependencies
- Replaced print statements with proper `logging` throughout extractors

## [0.2.0] - 2026-03-10

### Added
- **verbatim-core** package: lean, separately installable extraction core (openai + pydantic only)
- Optional `[model]` extra for verbatim-core: `pip install verbatim-core[model]` for ModernBERT/Zilliz extractors
- CI pipeline: linting (ruff), tests (Python 3.10-3.12), pip-audit
- MkDocs Material documentation site with API reference
- 88 unit tests covering models, response builder, LLM client, extractors, templates, and transform
- CONTRIBUTING.md

### Changed
- Moved `verbatim_core/` into `packages/core/` (standard Python monorepo layout)
- `verbatim-rag` now depends on `verbatim-core` (transitive dependency)
- Lazy imports for numpy, torch, and transformers in verbatim_core to keep lean install clean
- Import sorting and linting fixes across verbatim_core, verbatim_rag, api, and tests

### Removed
- Old test files that required heavy dependencies (torch, milvus)

## [0.1.9] - 2026-02-28

### Added
- `max_tokens` parameter to `SemanticHighlightExtractor`
- Flag to return search results along with RAG response
- `verify_spans` parameter to `LLMSpanExtractor`

## [0.1.8] - 2026-01-15

### Added
- `SemanticHighlightExtractor` using Zilliz semantic-highlight model
- Reranker support (Cohere, Jina, SentenceTransformers)
- Intent detection for query routing
- `VerbatimDOC` for document-level processing
- `UniversalDocument` for unified document representation
- Hybrid search (dense + sparse) support
- SPLADE embedding provider for CPU-only operation
