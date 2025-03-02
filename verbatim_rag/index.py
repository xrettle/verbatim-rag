"""
Index class for the Verbatim RAG system.
"""

import os
import pickle
import numpy as np
import faiss
import openai

from verbatim_rag.document import Document


class VerbatimIndex:
    """
    A vector index for document retrieval using FAISS.
    """

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the VerbatimIndex.

        :param embedding_model: The OpenAI embedding model to use
        :return: None
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.index = None
        self.document_ids = []

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI's API.

        :param texts: List of text strings to embed
        :return: Numpy array of embeddings
        """
        response = openai.embeddings.create(model=self.embedding_model, input=texts)

        # Extract embeddings from the response
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the index.

        :param documents: List of Document objects to add
        """
        if not documents:
            return

        # Get embeddings for the documents
        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings(texts)

        # Store the documents
        start_idx = len(self.documents)
        self.documents.extend(documents)

        # Create document IDs for the new documents
        new_ids = list(range(start_idx, start_idx + len(documents)))
        self.document_ids.extend(new_ids)

        # Create or update the FAISS index
        if self.index is None:
            # Create a new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        else:
            # Add to existing index
            self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search for documents similar to the query.

        :param query: The search query
        :param k: Number of documents to retrieve

        :return: List of retrieved Document objects
        """
        if not self.index or not self.documents:
            return []

        # Get embedding for the query
        query_embedding = self._get_embeddings([query])[0].reshape(1, -1)

        # Search the index
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)

        # Return the retrieved documents
        retrieved_docs = [
            self.documents[self.document_ids.index(int(idx))] for idx in indices[0]
        ]
        return retrieved_docs

    def save(self, directory: str) -> None:
        """
        Save the index and documents to disk.

        :param directory: Directory to save the index in
        """
        os.makedirs(directory, exist_ok=True)

        # Save the documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        # Save the document IDs
        with open(os.path.join(directory, "document_ids.pkl"), "wb") as f:
            pickle.dump(self.document_ids, f)

        # Save the FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        # Save the embedding model name
        with open(os.path.join(directory, "embedding_model.txt"), "w") as f:
            f.write(self.embedding_model)

    @classmethod
    def load(cls, directory: str) -> "VerbatimIndex":
        """
        Load an index from disk.

        :param directory: Directory containing the saved index

        :return: Loaded VerbatimIndex
        """
        # Load the embedding model name
        with open(os.path.join(directory, "embedding_model.txt"), "r") as f:
            embedding_model = f.read().strip()

        # Create a new index instance
        index = cls(embedding_model=embedding_model)

        # Load the documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            index.documents = pickle.load(f)

        # Load the document IDs
        with open(os.path.join(directory, "document_ids.pkl"), "rb") as f:
            index.document_ids = pickle.load(f)

        # Load the FAISS index
        index.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        return index
