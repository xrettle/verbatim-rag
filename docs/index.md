# Verbatim RAG

<p align="center">
  <img src="https://github.com/KRLabsOrg/verbatim-rag/blob/main/assets/chiliground.png?raw=true" alt="ChiliGround Logo" width="400"/>
  <br><em>Chill, I Ground!</em>
</p>

A minimalistic approach to Retrieval-Augmented Generation (RAG) that prevents hallucination by ensuring all generated content is explicitly derived from source documents.

[![PyPI](https://img.shields.io/pypi/v/verbatim-rag)](https://pypi.org/project/verbatim-rag/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ACL 2025](https://img.shields.io/badge/ACL%20Anthology-2025.bionlp--share.8-blue)](https://aclanthology.org/2025.bionlp-share.8/)

## Why Verbatim RAG?

Traditional RAG systems retrieve relevant documents and then allow an LLM to freely generate responses based on that context. This can lead to hallucinations where the model invents facts not present in the source material.

Verbatim RAG solves this by **extracting verbatim text spans** from documents and composing responses entirely from these exact passages, with direct citations linking back to sources.

## Two Packages

| Package | Install | Dependencies | Use case |
|---------|---------|-------------|----------|
| **verbatim-rag** | `pip install verbatim-rag` | Full (torch, milvus, etc.) | Complete RAG pipeline with indexing, search, and extraction |
| **verbatim-core** | `pip install verbatim-core` | Lean (openai, pydantic) | Reusable grounding core for integration into existing systems |

## Key Features

- **Hallucination Prevention** -- All responses are grounded in exact source text
- **Verbatim Extraction** -- Text spans are extracted exactly as they appear
- **Citation Tracking** -- Every response includes precise document references
- **Multiple Extractors** -- LLM-based, fine-tuned ModernBERT, or Zilliz semantic highlighting
- **CPU-Only Operation** -- Full pipeline can run without GPU using SPLADE embeddings
- **Template System** -- Flexible response formatting with multiple strategies

## How It Works

1. **Document Processing** -- Documents are processed using docling and chunked with chonkie
2. **Document Indexing** -- Indexed using vector embeddings (dense and/or sparse)
3. **Query Processing** -- Relevant documents are retrieved
4. **Span Extraction** -- Key passages are extracted verbatim
5. **Response Generation** -- Templates structure the answer with citations

## Citation

If you use Verbatim RAG in your research, please cite our paper:

```bibtex
@inproceedings{kovacs-etal-2025-kr,
    title = "{KR} Labs at {A}rch{EHR}-{QA} 2025: A Verbatim Approach for Evidence-Based Question Answering",
    author = "Kovacs, Adam and Schmitt, Paul and Recski, Gabor",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bionlp-share.8/",
}
```
