from typing import List, Tuple, Optional, Union, Dict, Any 
import numpy as np
import logging
from contextlib import contextmanager
from dataclasses import asdict , is_dataclass
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.models.Collection import Collection
except ImportError as e:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    Collection = None  # type: ignore

from .base import VectorDatabaseBase, sanitize_metadata
from ..chunks import DocumentChunk

logger = logging.getLogger(__name__)


def normalize_metadata(metadata: Any) -> Dict:
      """
      Convert metadata into plain dict.
      Supports:
      - dict
      - dataclass
      - None
      """

      if metadata is None:
          return {}

      if isinstance(metadata, dict):
          return metadata

      if is_dataclass(metadata):
          return asdict(metadata)

      raise TypeError(
          f"Unsupported metadata type: {type(metadata)}"
      )


class ChromaVectorDatabase(VectorDatabaseBase):
    """
    Production-Ready Chroma Vector Database Manager
    
    A comprehensive wrapper for ChromaDB that provides:
    - Automatic batching for large-scale operations
    - Multi-tenancy support with tenant isolation
    - Flexible embedding generation
    - Robust error handling and validation
    - Efficient similarity search with multiple distance metrics
    
    Attributes:
        embedder: Optional embedding model for text-to-vector conversion
        collection_name: Name of the ChromaDB collection
        distance_metric: Similarity metric ('cosine', 'l2', or 'ip')
        tenant_id: Optional tenant identifier for multi-tenancy
        batch_size: Batch size for bulk operations
        strict_mode: Enable strict validation checks
    
    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> embedder = SentenceTransformer('all-MiniLM-L6-v2')
        >>> db = ChromaVectorDatabase(
        ...     embedder=embedder,
        ...     collection_name="my_documents",
        ...     persist_directory="./chroma_db"
        ... )
        >>> db.add(
        ...     ids=["doc1", "doc2"],
        ...     documents=["Hello world", "Machine learning"],
        ...     metadatas=[{"source": "web"}, {"source": "paper"}]
        ... )
        >>> results = db.search("AI technology", top_k=5)
    """

    # Supported distance metrics and their ChromaDB equivalents
    SUPPORTED_METRICS = {
        "cosine": "cosine",
        "l2": "l2",
        "ip": "ip",  # Inner product
    }

    def __init__(
        self,
        embedder: Optional[Any] = None,
        collection_name: str = "default_collection",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
        tenant_id: Optional[str] = None,
        batch_size: int = 500,
        strict_mode: bool = True,
    ) -> None:
        """
        Initialize ChromaDB Vector Database
        
        Args:
            embedder: Embedding model with .encode() method (e.g., SentenceTransformer)
            collection_name: Name for the ChromaDB collection
            persist_directory: Path for persistent storage (None for in-memory)
            distance_metric: Similarity metric - 'cosine', 'l2', or 'ip'
            tenant_id: Optional tenant identifier for data isolation
            batch_size: Number of vectors to process per batch
            strict_mode: Enable strict validation of inputs
            
        Raises:
            ValueError: If distance_metric is not supported
            RuntimeError: If ChromaDB client initialization fails
        """
        super().__init__(embedder)
        if chromadb is None:
            raise ImportError(
                "chromadb library is required. Install it using: pip install chromadb"
            )
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.tenant_id = tenant_id
        self.batch_size = max(1, batch_size)  # Ensure positive batch size
        self.strict_mode = strict_mode
        self.embedding_dim: Optional[int] = None
        self._client: Optional[Any] = None
        self._collection: Optional[Collection] = None

        # Validate distance metric
        if distance_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported distance metric: '{distance_metric}'. "
                f"Supported metrics: {list(self.SUPPORTED_METRICS.keys())}"
            )

        self.metric_space = self.SUPPORTED_METRICS[distance_metric]

        # Initialize ChromaDB client and collection
        self._initialize_client(persist_directory)
        self._initialize_collection()

        logger.info(
            f"Vector database initialized: collection='{collection_name}', "
            f"metric='{distance_metric}', tenant='{tenant_id or 'none'}'"
        )

    def _initialize_client(self, persist_directory: Optional[str]) -> None:
        """
        Initialize ChromaDB client (persistent or in-memory)
        
        Args:
            persist_directory: Path for persistent storage, None for in-memory
            
        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
                logger.debug(f"Persistent client created at: {persist_directory}")
            else:
                self._client = chromadb.Client()
                logger.debug("In-memory client created")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}") from e

    def _initialize_collection(self) -> None:
        """
        Create or retrieve ChromaDB collection
        
        Raises:
            RuntimeError: If collection initialization fails
        """
        try:
            # Try to get existing collection
            self._collection = self._client.get_collection(name=self.collection_name)
            logger.debug(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            try:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.metric_space},
                )
                logger.debug(f"Created new collection: {self.collection_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create collection '{self.collection_name}': {e}"
                ) from e
            

    
    @property
    def client(self) -> Any:
        """Get ChromaDB client instance"""
        if self._client is None:
            raise RuntimeError("ChromaDB client not initialized")
        return self._client

    @property
    def collection(self) -> Collection: # type: ignore
        """Get ChromaDB collection instance"""
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        return self._collection

    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate embedding dimensions and consistency
        
        Args:
            embeddings: Array of embedding vectors
            
        Raises:
            ValueError: If embedding dimensions are inconsistent
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape: {embeddings.shape}"
            )

        # Set embedding dimension on first batch
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
            logger.debug(f"Embedding dimension set to: {self.embedding_dim}")

        # Validate dimension consistency
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

    def _validate_inputs(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
        embeddings: Optional[np.ndarray],
    ) -> None:
        """
        Validate input data consistency
        
        Args:
            ids: Document identifiers
            documents: Document texts
            metadatas: Document metadata
            embeddings: Optional pre-computed embeddings
            
        Raises:
            ValueError: If input lengths are inconsistent or invalid
        """
        if not ids or not documents or not metadatas:
            raise ValueError("ids, documents, and metadatas cannot be empty")

        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise ValueError(
                f"Length mismatch: ids({len(ids)}), "
                f"documents({len(documents)}), metadatas({len(metadatas)})"
            )

        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match ids length ({len(ids)})"
            )

        # Validate IDs are non-empty strings
        if self.strict_mode:
            for idx, doc_id in enumerate(ids):
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(f"Invalid ID at index {idx}: '{doc_id}'")

    def _enforce_tenant_isolation(self, metadata: Dict) -> Dict:
        """
        Add tenant identifier to metadata for multi-tenancy
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Metadata with tenant_id added if applicable
        """
        if self.tenant_id:
            metadata["tenant_id"] = self.tenant_id
        return metadata

    # ============================================================================
    # ADD / UPSERT OPERATIONS
    # ============================================================================

    def add(
        self,
        chunks: List[Any],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add or update document chunks in the vector database.
        Accepts a list of DocumentChunk objects (unified interface shared
        with FAISS and Pinecone backends) OR the legacy explicit-arguments
        style for backward compatibility::

            # Unified style (RAGSystem uses this)
            db.add(chunks)

            # Legacy style (still works)
            db.add(ids=[...], documents=[...], metadatas=[...])

        Args:
            chunks:     List of DocumentChunk objects  –OR–  List[str] IDs
                        when called in legacy style via keyword arguments.
            embeddings: Optional pre-computed embeddings (shape N × dim, float32).

        Raises:
            ValueError: If inputs are invalid or inconsistent.
            RuntimeError: If embedding generation or upsert fails.
        """
        # ── Detect unified style: List[DocumentChunk] ─────────────────────
        if (
            isinstance(chunks, list)
            and chunks
            and hasattr(chunks[0], "chunk_id")
        ):
            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [
                        sanitize_metadata({
                            **normalize_metadata(c.metadata),
                            "doc_id": c.doc_id
                        })
                        for c in chunks
                          ]

            if embeddings is None:
                # Use pre-computed embeddings from chunks if all present
                pre = [c.embedding for c in chunks]
                if all(e is not None for e in pre):
                    embeddings = np.asarray(pre, dtype=np.float32)
        else:
            # ── Legacy style: ids / documents / metadatas passed directly ──
            ids = chunks  # first arg was ids
            documents = documents  # already bound via caller's kwarg
            metadatas = [sanitize_metadata(m) for m in (metadatas or [])]

        # Validate inputs
        self._validate_inputs(ids, documents, metadatas, embeddings)

        # Generate embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError(
                    "Embedder is required when embeddings are not provided. "
                    "Provide either embedder in constructor or pre-computed embeddings."
                )
            try:
                embeddings = self.embedder.encode(documents)
                logger.debug(f"Generated embeddings for {len(documents)} documents")
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {e}") from e

        # Convert to numpy array and validate
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._validate_embeddings(embeddings)

        # Batch processing for efficiency
        total_upserted = 0
        for batch_start in range(0, len(ids), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(ids))

            # Extract batch data
            batch_ids = ids[batch_start:batch_end]
            batch_docs = documents[batch_start:batch_end]
            batch_meta = metadatas[batch_start:batch_end]
            batch_emb = embeddings[batch_start:batch_end]

            # Apply tenant isolation
            batch_meta = [
                self._enforce_tenant_isolation(meta.copy()) for meta in batch_meta
            ]

            try:
                # Upsert batch to ChromaDB
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                    embeddings=batch_emb.tolist(),
                )
                total_upserted += len(batch_ids)
                logger.debug(
                    f"Upserted batch {batch_start}-{batch_end} "
                    f"({len(batch_ids)} vectors)"
                )
            except Exception as e:
                logger.error(f"Failed to upsert batch {batch_start}-{batch_end}: {e}")
                raise RuntimeError(
                    f"Failed to upsert batch starting at index {batch_start}"
                ) from e

        logger.info(f"Successfully upserted {total_upserted} vectors")

    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents using semantic similarity\n        
        Args:
            query: Search query (text string or embedding vector)
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"category": "news"})
            
        Returns:
            List of tuples: (document_text, similarity_score, metadata)
            Results are sorted by similarity score (highest first)
            
        Raises:
            ValueError: If query is invalid or embedder is missing
            
        Example:
            >>> results = db.search("machine learning algorithms", top_k=3)
            >>> for doc, score, meta in results:
            ...     print(f"Score: {score:.3f} - {doc[:50]}...")
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")

        # Generate query embedding if text provided
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError(
                    "Embedder is required for text queries. "
                    "Provide embedder in constructor or pass embedding vector directly."
                )
            try:
                query_embedding = self.embedder.encode([query])[0]
            except Exception as e:
                raise RuntimeError(f"Failed to encode query: {e}") from e
        else:
            query_embedding = np.asarray(query, dtype=np.float32)

        # Validate embedding dimension
        if self.embedding_dim and len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) "
                f"does not match database dimension ({self.embedding_dim})"
            )

        # Apply tenant isolation to filters
        if self.tenant_id:
            tenant_filter = {"tenant_id": self.tenant_id}
            if filters:
                filters = {"$and": [filters, tenant_filter]}
            else:
                filters = tenant_filter

        # Execute search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters,
            )
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e

        # Parse and format results
        output = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id_str in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i] or {}
                document = results["documents"][0][i]

                # Convert distance to similarity score based on metric
                score = self._distance_to_score(distance)

                # Apply score threshold (unified with FAISS / Pinecone)
                if score_threshold is not None and score < score_threshold:
                    continue

                # Return (DocumentChunk, float) – unified interface
                chunk = DocumentChunk(
                    chunk_id=doc_id_str,
                    doc_id=metadata.get("doc_id", doc_id_str),
                    text=document,
                    metadata={k: v for k, v in metadata.items() if k != "doc_id"},
                )
                output.append((chunk, float(score)))

        logger.debug(f"Search returned {len(output)} results")
        return output

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert distance metric to similarity score
        
        Args:
            distance: Distance value from ChromaDB
            
        Returns:
            Similarity score (higher is better)
        """
        if self.distance_metric == "cosine":
            # Cosine distance: convert to similarity [0, 1]
            return 1.0 - distance
        elif self.distance_metric == "l2":
            # L2 distance: negate (smaller distance = higher score)
            return -distance
        else:  # ip (inner product)
            # Inner product: higher is better
            return distance

    # ============================================================================
    # DELETE OPERATIONS
    # ============================================================================

    def delete_by_ids(self, ids: List[str]) -> int:
        """
        Delete documents by their IDs\n
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
            
        Example:
            >>> deleted_count = db.delete_by_ids(["doc1", "doc2"])
        """
        if not ids:
            return 0

        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents by ID")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete by IDs: {e}")
            raise RuntimeError(f"Delete operation failed: {e}") from e

    def delete_by_filter(self, filters: Dict) -> int:
        """
        Delete documents matching metadata filters\n        
        Args:
            filters: Metadata filters (e.g., {"category": "spam"})
            
        Returns:
            Number of documents deleted
            
        Example:
            >>> deleted_count = db.delete_by_filter({"status": "archived"})
        """
        if not filters:
            logger.warning("delete_by_filter called with empty filters")
            return 0

        # Apply tenant isolation
        if self.tenant_id:
            filters = {"$and": [filters, {"tenant_id": self.tenant_id}]}

        try:
            # Get matching documents
            results = self.collection.get(where=filters)

            if not results["ids"]:
                logger.debug("No documents matched the filter")
                return 0

            # Delete matched documents
            self.collection.delete(ids=results["ids"])
            deleted_count = len(results["ids"])
            logger.info(f"Deleted {deleted_count} documents by filter")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}")
            raise RuntimeError(f"Delete operation failed: {e}") from e

    def clear_collection(self) -> int:
        """
        Delete all documents from the collection\n        
        Returns:
            Number of documents deleted
            
        Warning:
            This operation cannot be undone!
        """
        try:
            count = self.collection.count()
            if count > 0:
                # Get all IDs and delete
                all_data = self.collection.get()
                if all_data["ids"]:
                    self.collection.delete(ids=all_data["ids"])
                logger.warning(f"Cleared {count} documents from collection")
            return count
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RuntimeError(f"Clear operation failed: {e}") from e

    # ============================================================================
    # STATISTICS AND UTILITIES
    # ============================================================================

    def stats(self) -> Dict[str, Any]:
        """
        Get database statistics\n        
        Returns:
            Dictionary containing database statistics
            
        Example:
            >>> stats = db.stats()
            >>> print(f"Total vectors: {stats['total_vectors']}")
        """
        return {
            "collection_name": self.collection_name,
            "total_vectors": self.collection.count(),
            "distance_metric": self.distance_metric,
            "embedding_dimension": self.embedding_dim,
            "tenant_id": self.tenant_id,
            "batch_size": self.batch_size,
            "strict_mode": self.strict_mode,
        }

    def get_by_ids(self, ids: List[str]) -> List[Tuple[str, Dict]]:
        """
        Retrieve documents by their IDs\n        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of tuples: (document_text, metadata)
        """
        if not ids:
            return []

        try:
            results = self.collection.get(ids=ids)
            output = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    output.append((doc, metadata))
            return output
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise RuntimeError(f"Get operation failed: {e}") from e

    def __len__(self) -> int:
        """Return total number of vectors in the database"""
        return self.collection.count()

    def __repr__(self) -> str:
        """String representation of the database"""
        return (
            f"ChromaVectorDatabase(collection='{self.collection_name}', "
            f"metric='{self.distance_metric}', vectors={len(self)}, "
            f"tenant='{self.tenant_id or 'none'}')"
        )


    # ── Unified interface aliases ─────────────────────────────────────────────
    # These methods make ChromaVectorDatabase fully compatible with RAGSystem
    # and with the VectorDatabaseBase contract without breaking legacy callers.

    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks that belong to *doc_id*.\n
        Delegates to :meth:`delete_by_filter` with ``{"doc_id": doc_id}``.
        """
        return self.delete_by_filter({"doc_id": doc_id})

    def get_stats(self) -> Dict[str, Any]:
        """
        Return database statistics (unified name, delegates to :meth:`stats`).\n
        """
        base = self.stats()
        # Ensure mandatory keys are present
        base.setdefault("total_vectors", base.get("total_vectors", len(self)))
        base.setdefault("has_embedder", self.embedder is not None)
        return base

    # ============================================================================
    # CONTEXT MANAGER SUPPORT
    # ============================================================================

    @contextmanager
    def batch_operation(self):
        """
        Context manager for batch operations with automatic error handling
        
        Example:
            >>> with db.batch_operation():
            ...     db.add(ids1, docs1, metas1)
            ...     db.add(ids2, docs2, metas2)
        """
        logger.debug("Starting batch operation")
        try:
            yield self
            logger.debug("Batch operation completed successfully")
        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            raise
        finally:
            logger.debug("Batch operation finalized")

    # ── Async API ────────────────────────────────────────────────────
    async def asearch(self, query, top_k=10, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.search, query, top_k, **kwargs)

    async def aadd(self, chunks, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.add, chunks, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False