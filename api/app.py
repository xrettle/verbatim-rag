"""
FastAPI server for the Verbatim RAG system.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key using:")
    print("export OPENAI_API_KEY=your_api_key_here")

try:
    from verbatim_rag import (
        QueryRequest,
        QueryResponse,
        TemplateManager,
        VerbatimIndex,
        VerbatimRAG,
    )
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

            print("Initializing RAG system...")
            rag = VerbatimRAG(
                index=index,
                template_manager=template_manager,
            )
            print("RAG system initialized successfully.")
        else:
            print(
                f"Index not found at {DEFAULT_INDEX_PATH}. Please load resources first."
            )
    except Exception as e:
        print(f"Error during startup: {e}")

    yield

    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class LoadResourcesRequest(BaseModel):
    """Request model for loading resources."""

    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If not provided, will use the environment variable.",
        example="sk-...",
    )

    class Config:
        schema_extra = {"example": {"api_key": "sk-your-api-key-here"}}


class LoadResourcesResponse(BaseModel):
    """Response model for loading resources."""

    message: str = Field(
        description="Status message describing the result of the operation",
        example="Resources loaded successfully.",
    )
    success: bool = Field(
        description="Whether the operation was successful", example=True
    )

    class Config:
        schema_extra = {
            "example": {"message": "Resources loaded successfully.", "success": True}
        }


class StatusResponse(BaseModel):
    """Response model for the status endpoint."""

    resources_loaded: bool = Field(
        description="Whether the RAG system resources are loaded and ready",
        example=True,
    )
    message: str = Field(description="Status message", example="RAG system is ready.")

    class Config:
        schema_extra = {
            "example": {"resources_loaded": True, "message": "RAG system is ready."}
        }


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Check if resources are loaded and the system is ready."""
    global rag

    if rag:
        return StatusResponse(
            resources_loaded=True, message="RAG system is initialized and ready."
        )
    else:
        return StatusResponse(
            resources_loaded=False,
            message="RAG system not initialized. Please load resources first.",
        )


@app.post("/api/load-resources", response_model=LoadResourcesResponse)
async def load_resources(request: LoadResourcesRequest = None):
    """Load or reload resources for the RAG system."""
    global index, rag, template_manager

    if request and request.api_key:
        os.environ["OPENAI_API_KEY"] = request.api_key

    try:
        print("Loading index...")
        index = VerbatimIndex.load(DEFAULT_INDEX_PATH)

        print("Loading templates...")
        template_manager = TemplateManager()
        if Path(DEFAULT_TEMPLATES_PATH).exists():
            template_manager.load_templates(DEFAULT_TEMPLATES_PATH)

        print("Initializing RAG system...")
        rag = VerbatimRAG(
            index=index,
            template_manager=template_manager,
        )

        return LoadResourcesResponse(
            message="Resources loaded successfully.", success=True
        )
    except Exception as e:
        print(f"Error loading resources: {e}")
        return LoadResourcesResponse(
            message=f"Error loading resources: {str(e)}", success=False
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
        # Update the number of documents to retrieve
        rag.k = request.num_docs

        # Process the query using the core library
        result = rag.query(request.question)

        # Return the result directly as a QueryResponse
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            structured_answer=result.structured_answer,
            documents=result.documents,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
