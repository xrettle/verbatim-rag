# Verbatim Core (Extractor)

The `verbatim_core` package provides RAG-agnostic verbatim span extraction. It works with any retrieval system -- you provide the question and context, it returns grounded answers with citations.

## VerbatimTransform

The main entry point. Takes a question and retrieved context, extracts verbatim spans, and produces a cited response.

```python
from verbatim_core import VerbatimTransform

vt = VerbatimTransform(
    template_mode="contextual",  # or "static"
    extraction_mode="auto",      # "auto", "batch", or "individual"
    max_display_spans=5,
)

response = vt.transform(
    question="What methodology was used?",
    context=[
        {"content": "We used a mixed-methods approach combining surveys and interviews."},
        {"content": "The sample size was 500 participants across 3 hospitals."},
    ],
)
```

### Context Format

Context items can be:

- **Dicts** with `"content"` (or `"text"`), plus optional `"title"`, `"source"`, `"metadata"`
- **Objects** with a `.text` attribute (e.g., search results from any vector store)

## LLMSpanExtractor

Extracts verbatim spans from documents using an LLM. All extracted spans are verified to exist in the original text.

```python
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient

client = LLMClient(model="gpt-4o-mini")
extractor = LLMSpanExtractor(
    llm_client=client,
    extraction_mode="auto",  # batch for small inputs, individual for large
    batch_size=5,
)

spans = extractor.extract_spans("What is X?", search_results)
# Returns: {"doc text": ["verbatim span 1", "verbatim span 2"], ...}
```

## Template System

Controls how extracted spans are formatted into responses.

### Static Templates

Fast, deterministic -- no LLM calls:

```python
from verbatim_core.templates import TemplateManager

tm = TemplateManager(default_mode="static")
tm.use_static_mode(template="## Findings\n\n[DISPLAY_SPANS]")
```

### Contextual Templates

LLM-generated templates tailored to each question:

```python
from verbatim_core.llm_client import LLMClient

client = LLMClient()
tm = TemplateManager(llm_client=client, default_mode="contextual")
```

### Template Placeholders

- `[DISPLAY_SPANS]` -- all display spans aggregated
- `[FACT_1]`, `[FACT_2]`, ... -- individual span placeholders
- `[CITATION_REFS]` -- citation-only reference numbers

## @verbatim_enhance Decorator

Wrap any existing RAG function to add verbatim grounding:

```python
from verbatim_core import verbatim_enhance

@verbatim_enhance(max_display_spans=5)
def my_rag_function(question: str) -> list[dict]:
    # Your existing retrieval logic
    return [{"content": "Retrieved text...", "title": "Source"}]

response = my_rag_function("What is X?")
```

## CLI

Process JSON/JSONL files from the command line:

```bash
verbatim-enhance --input queries.jsonl --output results.jsonl
```

## Custom LLM Endpoints

Use any OpenAI-compatible API:

```python
from verbatim_core.llm_client import LLMClient

client = LLMClient(
    model="my-model",
    api_base="http://localhost:8080/v1",
)
```
