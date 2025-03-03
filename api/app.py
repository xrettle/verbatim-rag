"""
Verbatim RAG API - FastAPI Implementation

This API provides endpoints for:
1. Loading resources (index and templates)
2. Querying the Verbatim RAG system
3. Checking system status
"""

import os
import sys
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key using:")
    print("export OPENAI_API_KEY=your_api_key_here")

try:
    from verbatim_rag import VerbatimIndex, VerbatimRAG, TemplateManager
except ImportError as e:
    print(f"Error importing verbatim_rag: {e}")
    sys.exit(1)

DEFAULT_INDEX_PATH = "index/"
DEFAULT_TEMPLATES_PATH = "templates.json"

index = None
rag = None
template_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, rag, template_manager

    try:
        if Path(DEFAULT_INDEX_PATH).exists():
            print(f"Loading index from {DEFAULT_INDEX_PATH}...")
            index = VerbatimIndex.load(DEFAULT_INDEX_PATH)

            if Path(DEFAULT_TEMPLATES_PATH).exists():
                print(f"Loading templates from {DEFAULT_TEMPLATES_PATH}...")
                template_manager = TemplateManager()
                template_manager.load_templates(DEFAULT_TEMPLATES_PATH)

            rag = VerbatimRAG(index=index, template_manager=template_manager)
            print("Resources loaded successfully! Ready to answer questions.")
        else:
            print(f"Warning: Index not found at {DEFAULT_INDEX_PATH}")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")

    yield  # This is where the app runs

    print("Shutting down...")


app = FastAPI(
    title="Verbatim RAG API",
    description="API for the Verbatim RAG system that prevents hallucination by ensuring all generated content is explicitly derived from source documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadResourcesRequest(BaseModel):
    api_key: Optional[str] = None


class LoadResourcesResponse(BaseModel):
    message: str
    success: bool


class Highlight(BaseModel):
    text: str
    start: int
    end: int


class DocumentWithHighlights(BaseModel):
    content: str
    highlights: List[Highlight]


class Citation(BaseModel):
    text: str
    doc_index: int
    highlight_index: int


class StructuredAnswer(BaseModel):
    text: str
    citations: List[Citation]


class QueryRequest(BaseModel):
    question: str
    num_docs: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    structured_answer: Optional[StructuredAnswer] = None
    documents: List[DocumentWithHighlights]


class StatusResponse(BaseModel):
    resources_loaded: bool
    message: str


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Check if resources are loaded."""
    global rag

    if rag:
        return StatusResponse(
            resources_loaded=True, message="Resources are loaded and ready to use."
        )
    else:
        return StatusResponse(
            resources_loaded=False,
            message="Resources are not loaded yet. Please try again later.",
        )


@app.post("/api/load-resources", response_model=LoadResourcesResponse)
async def load_resources(request: LoadResourcesRequest = None):
    """Reload the index and templates using default paths. This is a backup endpoint for manual reloading."""
    global index, rag, template_manager

    try:
        if request and request.api_key:
            os.environ["OPENAI_API_KEY"] = request.api_key

        if "OPENAI_API_KEY" not in os.environ:
            return LoadResourcesResponse(
                message="OpenAI API key not set. Please set it in the environment.",
                success=False,
            )

        if not Path(DEFAULT_INDEX_PATH).exists():
            return LoadResourcesResponse(
                message=f"Index not found at {DEFAULT_INDEX_PATH}", success=False
            )

        index = VerbatimIndex.load(DEFAULT_INDEX_PATH)

        template_manager = None
        if Path(DEFAULT_TEMPLATES_PATH).exists():
            template_manager = TemplateManager()
            template_manager.load_templates(DEFAULT_TEMPLATES_PATH)

        rag = VerbatimRAG(index=index, template_manager=template_manager)

        return LoadResourcesResponse(
            message="Resources reloaded successfully! Ready to answer questions.",
            success=True,
        )
    except Exception as e:
        return LoadResourcesResponse(
            message=f"Error reloading resources: {str(e)}", success=False
        )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return the answer with highlighted documents."""
    global rag

    if not rag:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please load resources first.",
        )

    try:
        rag.k = request.num_docs

        answer, metadata = rag.query(request.question)

        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        answer = answer.replace("\\n", "\n")

        relevant_spans = metadata.get("relevant_spans", {})
        documents = metadata.get("retrieved_docs", [])

        documents_with_highlights = []

        all_citations = []

        for i, doc_content in enumerate(documents):
            highlights = []

            if doc_content in relevant_spans and relevant_spans[doc_content]:
                for span in relevant_spans[doc_content]:
                    start = 0
                    while True:
                        start = doc_content.find(span, start)
                        if start == -1:
                            break

                        highlight = Highlight(
                            text=span, start=start, end=start + len(span)
                        )

                        highlights.append(highlight)

                        all_citations.append(
                            Citation(
                                text=span,
                                doc_index=i,
                                highlight_index=len(highlights) - 1,
                            )
                        )

                        start += len(span)

            documents_with_highlights.append(
                DocumentWithHighlights(content=doc_content, highlights=highlights)
            )

        structured_answer = StructuredAnswer(text=answer, citations=all_citations)

        return QueryResponse(
            question=request.question,
            answer=answer,
            structured_answer=structured_answer,
            documents=documents_with_highlights,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
