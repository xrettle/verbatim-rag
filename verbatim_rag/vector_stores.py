"""
Vector storage for the Verbatim RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


def _sanitize_hybrid_weights(hybrid_weights: Dict[str, float]) -> Dict[str, float]:
    """Validate and filter hybrid weight config."""
    if not hybrid_weights:
        raise ValueError("hybrid_weights must be a non-empty dict")

    allowed_methods = {"dense", "sparse", "full_text"}
    cleaned: Dict[str, float] = {}

    for method, weight in hybrid_weights.items():
        if method not in allowed_methods:
            logger.warning("Ignoring unsupported hybrid method '%s'", method)
            continue
        if not isinstance(weight, (int, float)) or weight <= 0:
            logger.warning(
                "Ignoring non-positive weight for method '%s': %s", method, weight
            )
            continue
        cleaned[method] = float(weight)

    if not cleaned:
        raise ValueError("No valid hybrid_weights after validation")

    return cleaned


def _normalize_weights(
    results_by_method: Dict[str, List], weights: Dict[str, float]
) -> Dict[str, float]:
    available_weights = {m: weights.get(m, 0.0) for m in results_by_method}
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        logger.warning(
            "No non-zero weights for available methods; using equal weights "
            f"for: {list(results_by_method.keys())}"
        )
        return {k: 1.0 / len(results_by_method) for k in results_by_method}
    return {k: v / total_weight for k, v in available_weights.items()}


def _merge_hybrid_results(
    results_by_method: Dict[str, List],
    top_k: int,
    weights: Dict[str, float],
    rrf_k: int = 60,
    log_label: str = "",
):
    """Merge search results from multiple methods using weighted RRF."""
    normalized_weights = _normalize_weights(results_by_method, weights)

    if log_label:
        logger.info(
            "Hybrid merge (%s): methods=%s normalized_weights=%s rrf_k=%s top_k=%s",
            log_label,
            list(results_by_method.keys()),
            normalized_weights,
            rrf_k,
            top_k,
        )

    scores_by_id = {}
    hit_map = {}

    for method_name, results in results_by_method.items():
        weight = normalized_weights.get(method_name, 0.0)
        for rank, hit in enumerate(results):
            hit_id = hit.get("id")
            if not hit_id:
                continue
            rrf_score = 1.0 / (rrf_k + rank + 1)
            weighted_score = weight * rrf_score

            if hit_id not in scores_by_id:
                scores_by_id[hit_id] = 0.0
                hit_map[hit_id] = hit
            scores_by_id[hit_id] += weighted_score

    sorted_ids = sorted(
        scores_by_id.keys(), key=lambda id: scores_by_id[id], reverse=True
    )
    merged_results = []
    for hit_id in sorted_ids[:top_k]:
        hit = hit_map[hit_id].copy()
        hit["distance"] = 1.0 - scores_by_id[hit_id]
        merged_results.append(hit)

    return merged_results


def _convert_hits_to_results(
    hits: List,
    dynamic_fields: Optional[List[str]] = None,
) -> List["SearchResult"]:
    """Convert raw hits to SearchResult objects."""
    if dynamic_fields is None:
        dynamic_fields = []

    search_results: List[SearchResult] = []
    for hit in hits:
        entity = hit.get("entity", {})
        metadata = entity.get("metadata", {}) or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw": metadata}

        for f in dynamic_fields:
            val = entity.get(f)
            if val is not None:
                metadata[f] = val

        search_results.append(
            SearchResult(
                id=hit.get("id"),
                score=hit.get("distance", 0.0),
                text=entity.get("text", ""),
                enhanced_text=entity.get("enhanced_text", ""),
                metadata=metadata,
            )
        )
    return search_results


def _sanitize_hybrid_weights(hybrid_weights: Dict[str, float]) -> Dict[str, float]:
    """Validate and filter hybrid weight config."""
    if not hybrid_weights:
        raise ValueError("hybrid_weights must be a non-empty dict")

    allowed_methods = {"dense", "sparse", "full_text"}
    cleaned: Dict[str, float] = {}

    for method, weight in hybrid_weights.items():
        if method not in allowed_methods:
            logger.warning("Ignoring unsupported hybrid method '%s'", method)
            continue
        if not isinstance(weight, (int, float)) or weight <= 0:
            logger.warning(
                "Ignoring non-positive weight for method '%s': %s", method, weight
            )
            continue
        cleaned[method] = float(weight)

    if not cleaned:
        raise ValueError("No valid hybrid_weights after validation")

    return cleaned


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str  # Original clean text for display
    enhanced_text: str = ""  # Enhanced text used for vectorization


class VectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata, original text, and enhanced text."""
        pass

    @abstractmethod
    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query for similar vectors using hybrid search."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass


class LocalMilvusStore(VectorStore):
    """Local Milvus Lite storage with hybrid search support."""

    def __init__(
        self,
        db_path: str = "./milvus_verbatim.db",
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_full_text: bool = False,  # Milvus Lite doesn't support BM25
        index_type: str = "IVF_FLAT",
        nlist: int = 8192,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.documents_collection_name = f"{collection_name}_documents"
        self.dense_dim = dense_dim
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse
        # Milvus Lite does not support BM25 full text search
        # This feature is only available in Milvus Standalone/Distributed/Cloud
        if enable_full_text:
            logger.warning(
                "Full text search (BM25) is not supported in Milvus Lite. "
                "This feature requires Milvus Standalone, Distributed, or Zilliz Cloud. "
                "Full text search will be disabled."
            )
            self.enable_full_text = False
        else:
            self.enable_full_text = False
        self.index_type = index_type
        self.nlist = nlist

        # Validate at least one embedding type is enabled
        if not enable_dense and not enable_sparse:
            raise ValueError(
                "At least one of enable_dense or enable_sparse must be True"
            )

        self._setup_client()

    def _setup_client(self):
        try:
            from pymilvus import MilvusClient, DataType

            # Create Milvus Lite client
            self.client = MilvusClient(self.db_path)

            # Create collection with sparse vector support
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )

                # Add fields
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                # Add only the vector fields that are enabled
                if self.enable_dense:
                    schema.add_field(
                        field_name="dense_vector",
                        datatype=DataType.FLOAT_VECTOR,
                        dim=self.dense_dim,
                    )

                if self.enable_sparse:
                    schema.add_field(
                        field_name="sparse_vector",
                        datatype=DataType.SPARSE_FLOAT_VECTOR,
                    )
                schema.add_field(
                    field_name="text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(
                    field_name="enhanced_text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                # Single JSON field for all metadata
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

                # Note: BM25 full text search is NOT supported in Milvus Lite
                # This feature is only available in Milvus Standalone/Distributed/Cloud
                # If enable_full_text is True, it will be disabled with a warning in __init__
                if self.enable_full_text:
                    logger.warning(
                        "Attempting to enable full text search in Milvus Lite, "
                        "but this feature is not supported. Skipping BM25 setup."
                    )
                    self.enable_full_text = False

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                # Create indexes for enabled vector fields
                index_params = self.client.prepare_index_params()

                if self.enable_dense:
                    # Dense vector index
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type=self.index_type,
                        metric_type="COSINE",
                        params={"nlist": self.nlist},
                    )

                if self.enable_sparse:
                    # Sparse vector index
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )

                # Note: BM25 index not created for Milvus Lite (not supported)

                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )

                logger.info(f"Created indexes for collection: {self.collection_name}")

            # Create documents collection (no vectors, just metadata)
            if not self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                doc_schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )

                # Add fields for documents (no chunks, just document metadata)
                doc_schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                doc_schema.add_field(
                    field_name="title", datatype=DataType.VARCHAR, max_length=512
                )
                doc_schema.add_field(
                    field_name="source", datatype=DataType.VARCHAR, max_length=512
                )
                doc_schema.add_field(
                    field_name="content_type", datatype=DataType.VARCHAR, max_length=50
                )
                doc_schema.add_field(
                    field_name="raw_content",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                )
                doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)

                # Create documents collection
                self.client.create_collection(
                    collection_name=self.documents_collection_name, schema=doc_schema
                )

                logger.info(
                    f"Created documents collection: {self.documents_collection_name}"
                )

            logger.info(f"Connected to Milvus Lite: {self.db_path}")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text."""

        # Validate inputs based on enabled embedding types
        if self.enable_dense and (dense_vectors is None or len(dense_vectors) == 0):
            raise ValueError("Dense vectors required but not provided")
        if self.enable_sparse and (sparse_vectors is None or len(sparse_vectors) == 0):
            raise ValueError("Sparse vectors required but not provided")

        # Prepare data for Milvus with conditional vectors
        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj):
            """Safely serialize objects to JSON, handling datetime and enum objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value  # Convert enum to its string value
            elif isinstance(obj, dict):
                return {k: json_serialize_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            else:
                return obj

        data = []
        for i in range(len(ids)):
            # Serialize all metadata - ensure all values are JSON-serializable
            safe_metadata = json_serialize_safe(metadatas[i])

            item = {
                "id": ids[i],
                "text": texts[i],  # Original clean text
                "enhanced_text": enhanced_texts[i],  # Enhanced text for vectorization
                "metadata": safe_metadata,
            }

            # Add vectors only for fields that exist in the schema
            if self.enable_dense and dense_vectors:
                item["dense_vector"] = dense_vectors[i]
            if self.enable_sparse and sparse_vectors:
                item["sparse_vector"] = sparse_vectors[i]

            data.append(item)

        # Insert into Milvus
        self.client.insert(collection_name=self.collection_name, data=data)

        logger.info(f"Added {len(data)} vectors to Milvus")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document metadata to the documents collection."""

        # Prepare document data (no chunks, just metadata)
        import json
        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj):
            """Safely serialize objects to JSON, handling datetime and enum objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value  # Convert enum to its string value
            elif isinstance(obj, dict):
                return {k: json_serialize_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            else:
                return obj

        doc_data = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                # Handle datetime objects in metadata, but keep as dict (not JSON string)
                safe_metadata = json_serialize_safe(metadata)
            elif isinstance(metadata, str):
                # If it's already a string, try to parse it back to dict for consistency
                try:
                    safe_metadata = json.loads(metadata)
                except:
                    safe_metadata = {
                        "raw": metadata
                    }  # Fallback for invalid JSON strings
            else:
                # Convert other types to dict format
                safe_metadata = json_serialize_safe(metadata) if metadata else {}

            item = {
                "id": doc.get("id", ""),
                "title": doc.get("title") or "",  # Convert None to empty string
                "source": doc.get("source") or "",  # Convert None to empty string
                "content_type": doc.get("doc_type")
                or doc.get("content_type")
                or "",  # Handle None values
                "raw_content": "",
                "metadata": safe_metadata,  # Store as dict, not JSON string
            }
            doc_data.append(item)

        # Insert into documents collection
        self.client.insert(
            collection_name=self.documents_collection_name, data=doc_data
        )

        logger.info(f"Added {len(doc_data)} documents to Milvus")

    def add_document_schema(self, document_dict: Dict[str, Any], doc_id: str = None):
        """
        Add a single document using the new schema system.

        :param document_dict: Document dict from DocumentSchema.to_storage_dict()
        :param doc_id: Optional document ID override
        """
        if doc_id:
            document_dict["id"] = doc_id

        self.add_documents([document_dict])

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Retrieve document metadata by ID."""

        results = self.client.query(
            collection_name=self.documents_collection_name,
            filter=f"id == '{document_id}'",
            output_fields=[
                "id",
                "title",
                "source",
                "content_type",
                "raw_content",
                "metadata",
            ],
        )

        if results:
            return results[0]
        else:
            return None

    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "dense",
        filter: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Query using vector search, full text search, or filter-only browsing.

        Args:
            dense_query: Optional dense vector query
            sparse_query: Optional sparse vector query
            text_query: Optional text query for full text search (BM25)
            top_k: Number of results to return
            search_type: Type of search ("dense", "sparse", "hybrid", "full_text", "auto")
            filter: Optional filter expression
            search_params: Optional dict of search parameters (e.g., {"nprobe": 128} for IVF indexes)
            hybrid_weights: Optional dict of weights for hybrid search
                          e.g., {"dense": 0.5, "sparse": 0.3, "full_text": 0.2}
                          If provided, overrides search_type and enables N-way hybrid search
            rrf_k: RRF constant for hybrid search (default 60)
        """

        # If hybrid_weights provided, use N-way hybrid search
        if hybrid_weights is not None:
            return self._hybrid_search_with_weights(
                dense_query,
                sparse_query,
                text_query,
                top_k,
                filter,
                search_params,
                hybrid_weights,
                rrf_k,
            )

        # Full text search using BM25
        if search_type == "full_text" and text_query and self.enable_full_text:
            return self._full_text_search(text_query, top_k, filter)

        # If no vectors provided, do filter-only query
        if not dense_query and not sparse_query:
            return self._filter_only_query(filter, top_k)

        output_fields = [
            "text",
            "enhanced_text",
            "metadata",
        ]

        if search_type == "dense" and dense_query:
            # Dense vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # Hybrid search: combine dense and sparse results
            # Note: Milvus doesn't have native hybrid search, so we implement it
            # by combining results from both dense and sparse searches
            try:
                # Perform dense search
                dense_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k * 2,  # Get more results for merging
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

                # Perform sparse search
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[sparse_query],
                    anns_field="sparse_vector",
                    limit=top_k * 2,  # Get more results for merging
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

                # Combine results using reciprocal rank fusion (RRF)
                results_by_method = {
                    "dense": dense_results[0],
                    "sparse": sparse_results[0],
                }
                merged = _merge_hybrid_results(
                    results_by_method,
                    top_k,
                    {"dense": 0.5, "sparse": 0.5},
                    rrf_k=rrf_k,
                    log_label="LocalMilvus",
                )
                results = [merged]  # Wrap in list to match expected format

            except Exception as e:
                logger.warning(
                    f"Hybrid search failed: {e}, falling back to dense search"
                )
                # Fallback to dense search
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

        else:
            raise ValueError(
                f"Invalid search configuration: type={search_type}, dense={dense_query is not None}, sparse={sparse_query is not None}"
            )

        return _convert_hits_to_results(results[0])

    def _filter_only_query(
        self, filter: Optional[str], limit: int
    ) -> List[SearchResult]:
        """Query without vector search - just filtering/browsing."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter or "",  # Empty filter gets all
                output_fields=["id", "text", "enhanced_text", "metadata"],
                limit=limit,
            )

            # Convert to SearchResult format
            search_results = []
            for result in results:
                metadata = result.get("metadata", {})
                # Handle both dict and JSON string cases
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"raw": metadata}

                search_results.append(
                    SearchResult(
                        id=result.get("id", ""),
                        score=1.0,  # No relevance score for filter-only queries
                        text=result.get("text", ""),
                        enhanced_text=result.get("enhanced_text", ""),
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Failed to query chunks: {e}")
            return []

    def _full_text_search(
        self, text_query: str, top_k: int, filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform full text search using BM25.

        Args:
            text_query: Natural language text query
            top_k: Number of results to return
            filter: Optional filter expression

        Returns:
            List of SearchResult objects
        """
        if not self.enable_full_text:
            raise ValueError("Full text search is not enabled for this collection")

        try:
            # Use Milvus search with text query - it will automatically convert to BM25
            results = self.client.search(
                collection_name=self.collection_name,
                data=[text_query],  # Pass text directly - Milvus converts to BM25
                anns_field="bm25_vector",
                limit=top_k,
                output_fields=["text", "enhanced_text", "metadata"],
                filter=filter,
            )

            # Convert results
            search_results = []
            for hit in results[0]:
                entity = hit.get("entity", {})
                metadata = entity.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"raw": metadata}

                search_results.append(
                    SearchResult(
                        id=hit.get("id"),
                        score=hit.get("distance", 0.0),
                        text=entity.get("text", ""),
                        enhanced_text=entity.get("enhanced_text", ""),
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Full text search failed: {e}")
            return []

    def _hybrid_search_with_weights(
        self,
        dense_query: Optional[List[float]],
        sparse_query: Optional[Dict[int, float]],
        text_query: Optional[str],
        top_k: int,
        filter: Optional[str],
        search_params: Optional[Dict[str, Any]],
        hybrid_weights: Dict[str, float],
        rrf_k: int,
    ) -> List[SearchResult]:
        """Execute N-way hybrid search with custom weights.

        Args:
            dense_query: Dense vector query
            sparse_query: Sparse vector query
            text_query: Text query for full-text search
            top_k: Number of results
            filter: Optional filter
            search_params: Optional search parameters
            hybrid_weights: Dict of weights for each method
            rrf_k: RRF constant

        Returns:
            List of SearchResult objects
        """
        hybrid_weights = _sanitize_hybrid_weights(hybrid_weights)

        # Warn if full_text requested but not available
        if "full_text" in hybrid_weights and not self.enable_full_text:
            logger.warning(
                "full_text not available on LocalMilvus (only Standalone/Cloud), "
                "removing from hybrid_weights"
            )
            hybrid_weights = {
                k: v for k, v in hybrid_weights.items() if k != "full_text"
            }

        if not hybrid_weights:
            raise ValueError("No valid search methods in hybrid_weights")

        output_fields = ["text", "enhanced_text", "metadata"]
        results_by_method = {}

        # Execute each search method
        if "dense" in hybrid_weights and dense_query is not None:
            dense_results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k * 2,  # Fetch more for better merging
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["dense"] = dense_results[0]

        if "sparse" in hybrid_weights and sparse_query is not None:
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k * 2,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["sparse"] = sparse_results[0]

        if (
            "full_text" in hybrid_weights
            and text_query is not None
            and self.enable_full_text
        ):
            # Note: LocalMilvus doesn't support full_text, but keeping for consistency
            # This code path won't be reached due to the warning above
            pass

        # Handle single method or empty results
        if len(results_by_method) == 0:
            logger.warning("Hybrid search: no valid methods executed after validation")
            return []
        elif len(results_by_method) == 1:
            # Single method, just convert and return top_k
            method_results = list(results_by_method.values())[0]
            logger.info(
                "Hybrid search (LocalMilvus): methods=%s weights=%s rrf_k=%s top_k=%s",
                list(results_by_method.keys()),
                hybrid_weights,
                rrf_k,
                top_k,
            )
            return _convert_hits_to_results(method_results[:top_k])

        # Multiple methods: merge with RRF
        logger.info(
            "Hybrid search (LocalMilvus): methods=%s weights=%s rrf_k=%s top_k=%s",
            list(results_by_method.keys()),
            hybrid_weights,
            rrf_k,
            top_k,
        )
        merged_hits = _merge_hybrid_results(
            results_by_method, top_k, hybrid_weights, rrf_k, log_label="LocalMilvus"
        )
        return _convert_hits_to_results(merged_hits)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents stored in the documents collection"""
        try:
            if self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                # Query all documents from the documents collection with limit
                results = self.client.query(
                    collection_name=self.documents_collection_name,
                    filter="",  # No filter to get all
                    output_fields=[
                        "id",
                        "title",
                        "source",
                        "content_type",
                        "raw_content",
                        "metadata",
                    ],
                    limit=1000,  # Set a reasonable limit
                )
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    def delete(self, ids: List[str]):
        # Delete by filter
        filter_expr = f"id in {ids}"
        self.client.delete(collection_name=self.collection_name, filter=filter_expr)


class CloudMilvusStore(VectorStore):
    """Cloud Milvus storage with hybrid search support."""

    def __init__(
        self,
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        sparse_dim: int = 30000,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_full_text: bool = True,
        index_type: str = "IVF_FLAT",
        nlist: int = 8192,
    ):
        self.collection_name = collection_name
        self.documents_collection_name = f"{collection_name}_documents"
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self.uri = uri
        self.token = token
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse
        # Full text search IS supported in Cloud Milvus (Standalone/Distributed/Cloud)
        self.enable_full_text = enable_full_text
        self.index_type = index_type
        self.nlist = nlist

        # Validate at least one embedding type is enabled
        if not enable_dense and not enable_sparse and not enable_full_text:
            raise ValueError(
                "At least one of enable_dense, enable_sparse, or enable_full_text must be True"
            )

        self._setup_milvus()

    def _setup_milvus(self):
        try:
            from pymilvus import MilvusClient, DataType

            if not self.uri:
                raise ValueError(
                    "CloudMilvusStore requires 'uri' for cloud connections"
                )

            # Connect via MilvusClient (Cloud)
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(uri=self.uri)

            # Create main chunks collection if missing
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False, enable_dynamic_field=True
                )
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                if self.enable_dense:
                    schema.add_field(
                        field_name="dense_vector",
                        datatype=DataType.FLOAT_VECTOR,
                        dim=self.dense_dim,
                    )
                if self.enable_sparse:
                    schema.add_field(
                        field_name="sparse_vector",
                        datatype=DataType.SPARSE_FLOAT_VECTOR,
                    )
                schema.add_field(
                    field_name="text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(
                    field_name="enhanced_text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

                # Add BM25 function for full text search if enabled
                if self.enable_full_text:
                    try:
                        from pymilvus import Function, FunctionType

                        # Add BM25 sparse vector field
                        schema.add_field(
                            field_name="bm25_vector",
                            datatype=DataType.SPARSE_FLOAT_VECTOR,
                        )

                        # Add BM25 function to automatically convert text to sparse vectors
                        bm25_function = Function(
                            name="text_bm25_emb",
                            input_field_names=["text"],
                            output_field_names=["bm25_vector"],
                            function_type=FunctionType.BM25,
                        )
                        schema.add_function(bm25_function)
                    except ImportError:
                        logger.warning(
                            "BM25 function not available. Full text search will be disabled. "
                            "Ensure you have a compatible version of pymilvus."
                        )
                        self.enable_full_text = False

                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                index_params = self.client.prepare_index_params()
                if self.enable_dense:
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type=self.index_type,
                        metric_type="COSINE",
                        params={"nlist": self.nlist},
                    )
                if self.enable_sparse:
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )
                if self.enable_full_text:
                    # BM25 sparse vector index for full text search
                    try:
                        index_params.add_index(
                            field_name="bm25_vector",
                            index_type="SPARSE_INVERTED_INDEX",
                            metric_type="BM25",
                            params={
                                "inverted_index_algo": "DAAT_MAXSCORE",
                                "bm25_k1": 1.2,
                                "bm25_b": 0.75,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create BM25 index: {e}. Full text search may not work."
                        )
                        self.enable_full_text = False
                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )
                # Ensure collection is loaded once after index creation
                try:
                    self.client.load_collection(collection_name=self.collection_name)
                    logger.info(f"Loaded collection: {self.collection_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load collection {self.collection_name}: {e}"
                    )
                logger.info(f"Created cloud Milvus collection: {self.collection_name}")

            # Create documents collection (include dummy vector to satisfy cloud constraint)
            if not self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                doc_schema = self.client.create_schema(
                    auto_id=False, enable_dynamic_field=True
                )
                doc_schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                doc_schema.add_field(
                    field_name="title", datatype=DataType.VARCHAR, max_length=4096
                )
                doc_schema.add_field(
                    field_name="source", datatype=DataType.VARCHAR, max_length=4096
                )
                doc_schema.add_field(
                    field_name="content_type", datatype=DataType.VARCHAR, max_length=50
                )
                doc_schema.add_field(
                    field_name="raw_content",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                )
                doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)
                # Add required dummy vector (dim=2)
                doc_schema.add_field(
                    field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
                )

                self.client.create_collection(
                    collection_name=self.documents_collection_name, schema=doc_schema
                )
                # Create flat index on dummy_vector so collection can be loaded
                doc_index_params = self.client.prepare_index_params()
                doc_index_params.add_index(
                    field_name="dummy_vector",
                    index_type="FLAT",
                    metric_type="L2",
                    params={},
                )
                self.client.create_index(
                    collection_name=self.documents_collection_name,
                    index_params=doc_index_params,
                )
                # Load the documents collection into memory
                try:
                    self.client.load_collection(
                        collection_name=self.documents_collection_name
                    )
                    logger.info(
                        f"Loaded documents collection: {self.documents_collection_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load documents collection {self.documents_collection_name}: {e}"
                    )
                logger.info(
                    f"Created cloud Milvus documents collection: {self.documents_collection_name}"
                )

            logger.info("Connected to Milvus Cloud via MilvusClient")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def _collection_exists(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def _create_documents_collection(self):
        from pymilvus import DataType

        doc_schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        doc_schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        doc_schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="content_type", datatype=DataType.VARCHAR, max_length=50
        )
        doc_schema.add_field(
            field_name="raw_content", datatype=DataType.VARCHAR, max_length=65535
        )
        doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)
        # Add required dummy vector (dim=2)
        doc_schema.add_field(
            field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
        )

        self.client.create_collection(
            collection_name=self.documents_collection_name, schema=doc_schema
        )
        logger.info(
            f"Created cloud Milvus documents collection: {self.documents_collection_name}"
        )

    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text using MilvusClient."""

        def promote_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            # Copy to avoid mutating original
            md = dict(metadata or {})

            # Keys to promote to top-level dynamic fields if present
            # Simplicity: only promote keys we need to filter on
            promotable_keys = {"user_id", "document_id", "dataset_id"}

            promoted: Dict[str, Any] = {}
            for key in list(md.keys()):
                if key in promotable_keys:
                    promoted[key] = md.pop(key)

            return promoted, md

        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return getattr(obj, "value", str(obj))
            if isinstance(obj, dict):
                return {str(k): json_serialize_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            return obj

        data: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            promoted, cleaned_metadata = promote_metadata(metadatas[i])
            safe_metadata = json_serialize_safe(cleaned_metadata)

            item: Dict[str, Any] = {
                "id": ids[i],
                "text": texts[i],
                "enhanced_text": enhanced_texts[i],
                "metadata": safe_metadata,
                **promoted,
            }

            if self.enable_dense and dense_vectors:
                item["dense_vector"] = dense_vectors[i]
            if self.enable_sparse and sparse_vectors:
                item["sparse_vector"] = sparse_vectors[i]

            data.append(item)

        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Added {len(data)} vectors to Milvus (cloud)")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document metadata to documents collection."""
        if not documents:
            return

        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return getattr(obj, "value", str(obj))
            if isinstance(obj, dict):
                return {str(k): json_serialize_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            return obj

        rows: List[Dict[str, Any]] = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            safe_metadata = (
                json_serialize_safe(metadata)
                if isinstance(metadata, dict)
                else metadata
            )
            # Promote common filter fields to dynamic fields
            promoted: Dict[str, Any] = {}
            if isinstance(metadata, dict):
                for key in ["user_id", "dataset_id", "document_id"]:
                    if key in metadata:
                        promoted[key] = metadata.get(key)
            row = {
                "id": doc.get("id", ""),
                "title": doc.get("title") or "",
                "source": doc.get("source") or "",
                "content_type": doc.get("doc_type") or doc.get("content_type") or "",
                "raw_content": doc.get("raw_content", ""),
                "metadata": safe_metadata,
                "dummy_vector": [0.0, 0.0],
                **promoted,
            }
            rows.append(row)

        self.client.insert(collection_name=self.documents_collection_name, data=rows)
        logger.info(f"Added {len(rows)} documents to Milvus (cloud)")

    def add_document_schema(self, document_dict: Dict[str, Any], doc_id: str = None):
        if doc_id:
            document_dict["id"] = doc_id
        self.add_documents([document_dict])

    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Query using hybrid dense + sparse vectors, full text search, or filter-only browsing.

        Args:
            dense_query: Optional dense vector query
            sparse_query: Optional sparse vector query
            text_query: Optional text query for full text search (BM25)
            top_k: Number of results to return
            search_type: Type of search ("dense", "sparse", "hybrid", "full_text", "auto")
            filter: Optional filter expression
            search_params: Optional dict of search parameters (e.g., {"nprobe": 128} for IVF indexes)
            hybrid_weights: Optional dict mapping method names to weights (e.g., {"dense": 0.5, "sparse": 0.3, "full_text": 0.2})
            rrf_k: RRF constant for hybrid search (default: 60)
        """

        # NEW: If hybrid_weights provided, use N-way hybrid search
        if hybrid_weights is not None:
            return self._hybrid_search_with_weights(
                dense_query,
                sparse_query,
                text_query,
                top_k,
                filter,
                search_params,
                hybrid_weights,
                rrf_k,
            )

        # LEGACY: Full text search using BM25
        if search_type == "full_text" and text_query and self.enable_full_text:
            return self._full_text_search(text_query, top_k, filter)

        # Request fields; include promoted dynamic fields
        dynamic_fields = ["user_id", "document_id", "dataset_id"]
        output_fields = ["text", "enhanced_text", "metadata", *dynamic_fields]

        if not dense_query and not sparse_query:
            try:
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=filter or "",
                    output_fields=["id", *output_fields],
                    limit=top_k,
                )
                search_results: List[SearchResult] = []
                for r in results:
                    md = r.get("metadata", {}) or {}
                    if isinstance(md, str):
                        try:
                            md = json.loads(md)
                        except Exception:
                            md = {"raw": md}
                    for f in dynamic_fields:
                        if f in r and r[f] is not None:
                            md[f] = r[f]
                    search_results.append(
                        SearchResult(
                            id=r.get("id", ""),
                            score=1.0,
                            text=r.get("text", ""),
                            enhanced_text=r.get("enhanced_text", ""),
                            metadata=md,
                        )
                    )
                return search_results
            except Exception as e:
                logger.error(f"Filter-only query failed: {e}")
                return []

        if search_type == "dense" and dense_query:
            # Dense vector search only
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search only
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # Hybrid search - combine dense and sparse
            # Fallback hybrid: combine results from two searches
            try:
                dense_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[sparse_query],
                    anns_field="sparse_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )
                # Use new N-way merge signature with 50/50 weights (equivalent to alpha=0.5)
                results_by_method = {
                    "dense": dense_results[0],
                    "sparse": sparse_results[0],
                }
                merged = _merge_hybrid_results(
                    results_by_method,
                    top_k,
                    {"dense": 0.5, "sparse": 0.5},
                    rrf_k,
                    log_label="CloudMilvus",
                )
                results = [merged]
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}, falling back to dense-only")
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

        else:
            raise ValueError(f"Invalid search configuration: type={search_type}")

        return _convert_hits_to_results(results[0], dynamic_fields)

    def _hybrid_search_with_weights(
        self,
        dense_query: Optional[List[float]],
        sparse_query: Optional[Dict[int, float]],
        text_query: Optional[str],
        top_k: int,
        filter: Optional[str],
        search_params: Optional[Dict[str, Any]],
        hybrid_weights: Dict[str, float],
        rrf_k: int,
    ) -> List[SearchResult]:
        """
        Execute N-way hybrid search using weighted RRF combination.

        Args:
            dense_query: Dense vector query
            sparse_query: Sparse vector query
            text_query: Text query for BM25 full text search
            top_k: Number of results to return
            filter: Optional filter expression
            search_params: Optional search parameters
            hybrid_weights: Dict mapping method names to weights (e.g., {"dense": 0.5, "sparse": 0.3, "full_text": 0.2})
            rrf_k: RRF constant

        Returns:
            List of SearchResult objects
        """
        hybrid_weights = _sanitize_hybrid_weights(hybrid_weights)

        # Request fields; include promoted dynamic fields
        dynamic_fields = ["user_id", "document_id", "dataset_id"]
        output_fields = ["text", "enhanced_text", "metadata", *dynamic_fields]

        # Execute searches for each method in hybrid_weights
        results_by_method = {}

        if "dense" in hybrid_weights and dense_query is not None:
            logger.info("Executing dense search for hybrid")
            dense_results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k * 2,  # Fetch more for better merging
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["dense"] = dense_results[0]

        if "sparse" in hybrid_weights and sparse_query is not None:
            logger.info("Executing sparse search for hybrid")
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k * 2,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["sparse"] = sparse_results[0]

        if "full_text" in hybrid_weights and text_query is not None:
            if self.enable_full_text:
                logger.info("Executing full text (BM25) search for hybrid")
                try:
                    full_text_results = self.client.search(
                        collection_name=self.collection_name,
                        data=[text_query],
                        anns_field="bm25_vector",
                        limit=top_k * 2,
                        output_fields=output_fields,
                        filter=filter,
                    )
                    results_by_method["full_text"] = full_text_results[0]
                except Exception as e:
                    logger.warning(
                        f"Full text search failed: {e}, excluding from hybrid"
                    )
            else:
                logger.warning(
                    "full_text requested but not enabled for this collection, excluding from hybrid"
                )
                # Remove full_text from weights
                hybrid_weights = {
                    k: v for k, v in hybrid_weights.items() if k != "full_text"
                }

        # Handle results
        if len(results_by_method) == 0:
            logger.warning("No valid search methods available for hybrid search")
            return []
        elif len(results_by_method) == 1:
            # Only one method available, just take top_k
            method_results = list(results_by_method.values())[0]
            logger.info(
                "Hybrid search (CloudMilvus): methods=%s weights=%s rrf_k=%s top_k=%s",
                list(results_by_method.keys()),
                hybrid_weights,
                rrf_k,
                top_k,
            )
            merged_hits = method_results[:top_k]
        else:
            # Multiple methods - merge using N-way RRF
            logger.info(f"Merging {len(results_by_method)} search methods using RRF")
            merged_hits = _merge_hybrid_results(
                results_by_method, top_k, hybrid_weights, rrf_k, log_label="CloudMilvus"
            )

        # Convert to SearchResult objects
        return _convert_hits_to_results(merged_hits, dynamic_fields)

    def _full_text_search(
        self, text_query: str, top_k: int, filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform full text search using BM25.

        Args:
            text_query: Natural language text query
            top_k: Number of results to return
            filter: Optional filter expression

        Returns:
            List of SearchResult objects
        """
        if not self.enable_full_text:
            raise ValueError("Full text search is not enabled for this collection")

        try:
            dynamic_fields = ["user_id", "document_id", "dataset_id"]
            output_fields = ["text", "enhanced_text", "metadata", *dynamic_fields]

            results = self.client.search(
                collection_name=self.collection_name,
                data=[text_query],
                anns_field="bm25_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

            return _convert_hits_to_results(results[0], dynamic_fields)
        except Exception as e:
            logger.error(f"Full text search failed: {e}")
            return []

    def delete(self, ids: List[str]):
        if not ids:
            return
        quoted = ",".join([f'"{_id}"' for _id in ids])
        self.client.delete(
            collection_name=self.collection_name, filter=f"id in [{quoted}]"
        )

    def delete_document(self, document_id: str):
        """Delete document row and all chunks for a document by document_id."""
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{document_id}"',
        )
        try:
            self.client.delete(
                collection_name=self.documents_collection_name,
                filter=f'id == "{document_id}"',
            )
        except Exception as e:
            logger.warning(f"Failed to delete document row for {document_id}: {e}")
