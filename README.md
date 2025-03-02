# Verbatim RAG

A minimalistic approach to Retrieval-Augmented Generation (RAG) that prevents hallucination by ensuring all generated content is explicitly derived from source documents.

## Concept

Traditional RAG systems retrieve relevant documents and then allow an LLM to freely generate responses based on that context. This can lead to hallucinations where the model invents facts not present in the source material.

Verbatim RAG addresses this by:

1. **Dynamic Template Generation**: Using an LLM to create a response template based on the question
2. **XML-Based Context Marking**: Identifying and extracting relevant passages from source documents using XML tags
3. **Verbatim Extraction**: Ensuring extracted content matches exactly with the original document text
4. **Template Filling**: Inserting only the extracted passages into the template

This approach ensures that all factual information in the response comes directly from source documents, while still leveraging the LLM's abilities to structure coherent responses.

## Installation

```bash
# Clone the repository
git clone https://github.com/krlabsorg/verbatim-rag.git
cd verbatim-rag

# Install the package in development mode
pip install -e .
```

## Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here
```

```python
import os
from verbatim_rag import Document, VerbatimIndex, VerbatimRAG

# Create documents
documents = [
    Document(
        content="The Golden Gate Bridge was opened in 1937.",
        metadata={"source": "example"}
    )
]
# Create and populate the index
index = VerbatimIndex()
index.add_documents(documents)

verbatim_rag = VerbatimRAG(index)

# Query the system
response, details = verbatim_rag.query("When was the Golden Gate Bridge opened?")
print(response)
```

For a more complete example, see the comprehensive example in `examples/example.ipynb`.

## How It Works

1. **Dynamic Template Generation**: The system generates a template with an introduction, a [CONTENT] placeholder, and a conclusion
2. **Document Retrieval**: Relevant documents are retrieved using FAISS vector search
3. **Unified Document Processing**: All retrieved documents are processed in a single prompt, allowing the model to identify the most relevant information across all documents
4. **XML Marking**: The LLM marks relevant parts of documents with `<relevant>...</relevant>` tags
5. **Verbatim Extraction**: The system extracts the marked content and verifies it against the original text
6. **Template Filling**: The extracted content is formatted as a numbered list and inserted into the template

This approach ensures that all factual information comes directly from the source documents without any modification.

## Command-Line Interface

Verbatim RAG includes a command-line interface for indexing documents, managing templates, and querying the system:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Index documents
verbatim-rag index --input documents/ --output index/

# Create templates for common questions
verbatim-rag template create --questions "When was X built?" "Who designed X?" --output templates.json

# Match a question to existing templates
verbatim-rag template match --question "When was X constructed?" --templates templates.json

# Query with automatic template matching
verbatim-rag query --index index/ --question "When was X built?" --templates-file templates.json

# Start an interactive session with automatic template matching
verbatim-rag interactive --index index/ --templates-file templates.json
```

When using the `--templates-file` option, the system will:
1. Automatically match the question to existing templates
2. Use the best matching template if the similarity score is high enough
3. Generate a new template if no good match is found
4. Save any new templates back to the templates file
