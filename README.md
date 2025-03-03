# Verbatim RAG

A minimalistic approach to Retrieval-Augmented Generation (RAG) that prevents hallucination by ensuring all generated content is explicitly derived from source documents.

## Concept

Traditional RAG systems retrieve relevant documents and then allow an LLM to freely generate responses based on that context. This can lead to hallucinations where the model invents facts not present in the source material.

## Installation

```bash
# Clone the repository
git clone https://github.com/krlabsorg/verbatim-rag.git
cd verbatim-rag

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from verbatim_rag import Document, VerbatimIndex, VerbatimRAG, QueryRequest

# Create and index documents
index = VerbatimIndex()
index.add_documents([
    Document(content="The Golden Gate Bridge was opened in 1937.")
])

rag = VerbatimRAG(index)

response = rag.query("When was the Golden Gate Bridge opened?")
print(response.answer)
```

## Command Line Interface

Verbatim RAG provides a simple CLI with three main commands:

### 1. Index Documents

Create a searchable index from your documents:

```bash
verbatim-rag index --input docs/ --output index/
```

### 2. Manage Templates

Create response templates for common question types:

```bash
verbatim-rag template \
    --questions "When was X built?" "Who designed X?" \
    --output templates.json \
    --model gpt-4
```

### 3. Query the System

Ask questions and get answers from your indexed documents:

```bash
verbatim-rag query \
    --index index/ \
    --question "When was the Golden Gate Bridge built?" \
    --num-docs 5 \
    --templates templates.json \
    --output response.json
```

### Environment Setup

Set your OpenAI API key before using the system:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## How It Works

1. **Document Indexing**: Documents are indexed using FAISS for efficient retrieval
2. **Template Management**: Response templates are created and stored for common question types
3. **Query Processing**: 
   - Relevant documents are retrieved
   - Key passages are extracted verbatim
   - Responses are structured using templates
   - Citations link back to source documents

This ensures all responses are grounded in the source material, preventing hallucination while maintaining natural language fluency.
