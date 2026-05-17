from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import logging
from pathlib import Path
import os
import sys
import time
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec
except ImportError as e:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    PodSpec = None  # type: ignore

from ..chunks import DocumentChunk

from .base import VectorDatabaseBase, sanitize_metadata

logger = logging.getLogger(__name__)


class PineconeVectorDatabase(VectorDatabaseBase):
    """
    Production-Ready Pinecone Vector Database Manager
    
    A comprehensive wrapper for Pinecone that provides:
    - Fully managed, serverless vector search infrastructure
    - Horizontal scaling with automatic sharding
    - Multiple cloud provider and region support
    - Rich metadata filtering with expression language
    - Namespace-based multi-tenancy
    - High availability and durability guarantees
    - Real-time updates with CRUD operations
    
    Attributes:
        embedder: Optional embedding model for text-to-vector conversion
        index_name: Name of the Pinecone index
        embedding_dim: Dimensionality of embedding vectors
        distance_metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
        chunks: Local cache of DocumentChunk objects
        
    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> embedder = SentenceTransformer('all-MiniLM-L6-v2')
        >>> 
        >>> db = PineconeVectorDatabase(
        ...     embedder=embedder,
        ...     index_name="my-index",
        ...     api_key="your-api-key",
        ...     environment="us-east-1",
        ...     metric="cosine"
        ... )
        >>> 
        >>> # Add documents
        >>> chunk = DocumentChunk(
        ...     chunk_id="c1",
        ...     doc_id="d1",
        ...     text="Machine learning is fascinating",
        ...     metadata={"category": "tech"}
        ... )
        >>> db.add_chunk(chunk)
        >>> 
        >>> # Search with filtering
        >>> results = db.search(
        ...     query="AI technology",
        ...     top_k=5,
        ...     filter_dict={"category": "tech"}
        ... )
        >>> for chunk, score in results:
        ...     print(f"Score: {score:.3f} - {chunk.text[:50]}")
    """

    # Supported configurations
    SUPPORTED_METRICS = {'cosine', 'euclidean', 'l2', 'dotproduct', 'dot'}
    METRIC_MAPPING = {
        'cosine': 'cosine',
        'euclidean': 'euclidean',
        'l2': 'euclidean',
        'dotproduct': 'dotproduct',
        'dot': 'dotproduct',
        'ip': 'dotproduct',
    }
    
    # Batch size limits
    UPSERT_BATCH_SIZE = 100  # Pinecone max upsert batch size
    DELETE_BATCH_SIZE = 1000  # Pinecone max delete batch size
    FETCH_BATCH_SIZE = 1000   # Pinecone max fetch batch size
    
    # Default parameters
    DEFAULT_ENVIRONMENT = "us-east-1"
    DEFAULT_CLOUD = "aws"
    DEFAULT_TOP_K = 5
    DEFAULT_INDEX_READY_TIMEOUT = 300  # 5 minutes

    def __init__(
        self,
        embedder: Optional[Any] = None,
        index_name: str = "default-index",
        embedding_dim: Optional[int] = None,
        api_key: Optional[str] = None,
        environment: str = DEFAULT_ENVIRONMENT,
        distance_metric: str = "cosine",
        cloud: str = DEFAULT_CLOUD,
        pod_type: Optional[str] = None,
        namespace: Optional[str] = None,
        cache_chunks: bool = True,
    ) -> None:
        """
        Initialize Pinecone Vector Database
        
        Args:
            embedder: Embedding model with .encode() method
            index_name: Name of the Pinecone index
            embedding_dim: Dimension of embeddings (required if embedder not provided)
            api_key: Pinecone API key (if None, reads from PINECONE_API_KEY env var)
            environment: Pinecone environment/region (e.g., 'us-east-1')
            distance_metric: Distance metric - 'cosine', 'euclidean', 'dotproduct'
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            pod_type: Pod type for pod-based indexes (None for serverless)
            namespace: Optional namespace for multi-tenancy
            cache_chunks: Whether to cache chunks locally
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If cannot connect to Pinecone
        """
        super().__init__(embedder)
        if Pinecone is None:
            raise ImportError(
                "pinecone-client library is required. Install it using: pip install pinecone-client"
            )
        self.index_name = index_name
        self.distance_metric = self._validate_metric(distance_metric)
        self.environment = environment
        self.cloud = cloud
        self.pod_type = pod_type
        self.namespace = namespace or ""  # Default namespace
        self.cache_chunks = cache_chunks
        
        # Local storage
        self.chunks: List[DocumentChunk] = []
        self._chunk_id_to_idx: Dict[str, int] = {}
        self._doc_id_to_indices: Dict[str, List[int]] = {}
        
        # Determine embedding dimension
        self.embedding_dim = self._determine_embedding_dim(embedder, embedding_dim)
        
        # Get Pinecone metric
        self.pinecone_metric = self.METRIC_MAPPING[self.distance_metric]
        
        # Get API key
        self.api_key = self._get_api_key(api_key)
        
        # Initialize Pinecone client and index
        self._pc: Optional[Pinecone] = None
        self._index = None
        self._initialize_pinecone()
        
        logger.info(
            f"✅ Pinecone database initialized: index='{index_name}', "
            f"dim={self.embedding_dim}, metric={self.pinecone_metric}, "
            f"namespace='{self.namespace or 'default'}'"
        )

    # ============================================================================
    # INITIALIZATION AND VALIDATION
    # ============================================================================

    def _validate_metric(self, metric: str) -> str:
        """Validate distance metric"""
        metric = metric.lower()
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported distance metric: '{metric}'. "
                f"Supported metrics: {self.SUPPORTED_METRICS}"
            )
        return metric

    def _determine_embedding_dim(
        self,
        embedder: Optional[Any],
        embedding_dim: Optional[int]
    ) -> int:
        """
        Determine embedding dimension from embedder or parameter
        
        Args:
            embedder: Optional embedder object
            embedding_dim: Optional explicit dimension
            
        Returns:
            Embedding dimension
            
        Raises:
            ValueError: If dimension cannot be determined
        """
        if embedding_dim is not None:
            if embedding_dim <= 0:
                raise ValueError(f"embedding_dim must be positive, got: {embedding_dim}")
            return embedding_dim
        
        if embedder is not None:
            try:
                sample_embedding = embedder.encode(["test"])
                dim = sample_embedding.shape[1] if len(sample_embedding.shape) > 1 else len(sample_embedding)
                logger.debug(f"Extracted embedding dimension from embedder: {dim}")
                return dim
            except Exception as e:
                raise ValueError(
                    f"Failed to extract embedding dimension from embedder: {e}"
                ) from e
        
        raise ValueError(
            "Must specify either 'embedder' or 'embedding_dim'. "
            "Cannot determine embedding dimension from provided arguments."
        )

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """
        Get Pinecone API key from parameter or environment
        
        Args:
            api_key: Optional API key parameter
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key not found
        """
        if api_key is not None:
            return api_key
        
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError(
                "Pinecone API key not found. "
                "Provide 'api_key' parameter or set PINECONE_API_KEY environment variable."
            )
        
        return api_key

    def _initialize_pinecone(self) -> None:
        """
        Initialize Pinecone client and index
        
        Raises:
            ConnectionError: If connection fails
            RuntimeError: If index initialization fails
        """
        try:
            # Create Pinecone client
            self._pc = Pinecone(api_key=self.api_key)
            logger.debug("Created Pinecone client")
            
            # Check if index exists
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                self._create_index()
            else:
                logger.info(f"Found existing index: {self.index_name}")
                # Verify index configuration
                self._verify_index_config()
            
            # Get index reference
            self._index = self._pc.Index(self.index_name)
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize Pinecone: {e}"
            ) from e

    def _create_index(self) -> None:
        """
        Create new Pinecone index
        
        Raises:
            RuntimeError: If index creation fails
        """
        try:
            # Determine spec (serverless vs pod-based)
            if self.pod_type:
                # Pod-based index
                spec = PodSpec(
                    environment=self.environment,
                    pod_type=self.pod_type
                )
                logger.debug(f"Creating pod-based index: {self.pod_type}")
            else:
                # Serverless index
                spec = ServerlessSpec(
                    cloud=self.cloud,
                    region=self.environment
                )
                logger.debug(f"Creating serverless index: {self.cloud}/{self.environment}")
            
            # Create index
            self._pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric=self.pinecone_metric,
                spec=spec
            )
            
            logger.info(f"✅ Created new index: {self.index_name}")
            
            # Wait until index is ready
            self._wait_for_index_ready()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create index: {e}") from e

    def _wait_for_index_ready(self, timeout: int = DEFAULT_INDEX_READY_TIMEOUT) -> None:
        """
        Wait for index to be ready
        
        Args:
            timeout: Maximum wait time in seconds
            
        Raises:
            TimeoutError: If index doesn't become ready within timeout
        """
        start_time = time.time()
        logger.debug(f"Waiting for index to be ready (timeout: {timeout}s)")
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Index '{self.index_name}' not ready after {timeout} seconds"
                )
            
            try:
                status = self._pc.describe_index(self.index_name).status
                if status.get('ready', False):
                    logger.debug("Index is ready")
                    break
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
            
            time.sleep(1)

    def _verify_index_config(self) -> None:
        """
        Verify existing index configuration matches requirements
        
        Raises:
            ValueError: If configuration doesn't match
        """
        try:
            index_info = self._pc.describe_index(self.index_name)
            
            # Check dimension
            if index_info.dimension != self.embedding_dim:
                raise ValueError(
                    f"Index dimension mismatch. "
                    f"Expected {self.embedding_dim}, got {index_info.dimension}"
                )
            
            # Check metric
            if index_info.metric != self.pinecone_metric:
                logger.warning(
                    f"Index metric mismatch. "
                    f"Expected {self.pinecone_metric}, got {index_info.metric}"
                )
            
        except Exception as e:
            logger.warning(f"Could not verify index configuration: {e}")

    @property
    def index(self):
        """Get Pinecone index instance"""
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")
        return self._index

    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================

    def _validate_embedding_dimension(self, embeddings: np.ndarray) -> None:
        """
        Validate embedding dimensions
        
        Args:
            embeddings: Embedding vectors
            
        Raises:
            ValueError: If dimensions don't match
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape: {embeddings.shape}"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

    def _validate_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Validate chunk objects
        
        Args:
            chunks: List of chunks to validate
            
        Raises:
            ValueError: If chunks are invalid
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")
        
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise ValueError(
                    f"Invalid chunk at index {idx}: expected DocumentChunk, "
                    f"got {type(chunk)}"
                )
            
            if not chunk.chunk_id or not isinstance(chunk.chunk_id, str):
                raise ValueError(f"Invalid chunk_id at index {idx}: '{chunk.chunk_id}'")
            
            if not chunk.doc_id or not isinstance(chunk.doc_id, str):
                raise ValueError(f"Invalid doc_id at index {idx}: '{chunk.doc_id}'")

    # ============================================================================
    # ADD OPERATIONS
    # ============================================================================

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """
        Add a single document chunk to the database
        
        Args:
            chunk: DocumentChunk object to add
            
        Raises:
            ValueError: If chunk is invalid
            RuntimeError: If addition fails
            
        Example:
            >>> chunk = DocumentChunk(
            ...     chunk_id="c1",
            ...     doc_id="d1",
            ...     text="Example text",
            ...     metadata={"category": "tech"}
            ... )
            >>> db.add_chunk(chunk)
        """
        if not isinstance(chunk, DocumentChunk):
            raise ValueError(f"Expected DocumentChunk, got {type(chunk)}")
        
        # Compute embedding if not present
        if chunk.embedding is None:
            if self.embedder is None:
                raise ValueError(
                    "No embedder available. Provide embedder in constructor or "
                    "set chunk.embedding before adding."
                )
            chunk.embedding = self.embedder.encode([chunk.text])[0]
        
        # Convert to array and add
        embedding = np.array([chunk.embedding])
        self.add(chunks=[chunk], embeddings=embedding)

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[np.ndarray] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Add multiple document chunks with their embeddings\n        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Optional pre-computed embeddings array
            namespace: Optional namespace (uses instance namespace if None)
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If addition fails
            
        Example:
            >>> chunks = [
            ...     DocumentChunk(chunk_id="c1", doc_id="d1", text="Text 1"),
            ...     DocumentChunk(chunk_id="c2", doc_id="d1", text="Text 2"),
            ... ]
            >>> db.add(chunks)
        """
        if not chunks:
            logger.warning("⚠️ No chunks to add")
            return
        
        # Validate chunks
        self._validate_chunks(chunks)
        
        # Compute embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError(
                    "No embedder available. Provide embedder in constructor or "
                    "pass pre-computed embeddings."
                )
            texts = [chunk.text for chunk in chunks]
            try:
                embeddings = self.embedder.encode(texts)
                logger.debug(f"Generated embeddings for {len(texts)} texts")
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {e}") from e
        
        # Convert to numpy array and validate
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._validate_embedding_dimension(embeddings)
        
        # Validate lengths match
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must equal "
                f"number of embeddings ({len(embeddings)})"
            )
        
        # Use provided namespace or instance namespace
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                # Create metadata
                metadata = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'text': chunk.text,
                }
                
                # Add additional metadata (sanitized via base helper)
                if chunk.metadata:
                    metadata.update(sanitize_metadata(chunk.metadata))
                
                # Pinecone vector format
                vectors.append({
                    'id': chunk.chunk_id,
                    'values': embedding.tolist(),
                    'metadata': metadata
                })
                
                # Cache chunk locally if enabled
                if self.cache_chunks:
                    chunk.embedding = embedding
                    
                    # Update local cache
                    if chunk.chunk_id in self._chunk_id_to_idx:
                        # Update existing chunk
                        idx = self._chunk_id_to_idx[chunk.chunk_id]
                        self.chunks[idx] = chunk
                    else:
                        # Add new chunk
                        idx = len(self.chunks)
                        self.chunks.append(chunk)
                        self._chunk_id_to_idx[chunk.chunk_id] = idx
                        
                        if chunk.doc_id not in self._doc_id_to_indices:
                            self._doc_id_to_indices[chunk.doc_id] = []
                        self._doc_id_to_indices[chunk.doc_id].append(idx)
            
            # Upload vectors to Pinecone in batches
            total_upserted = 0
            for batch_start in range(0, len(vectors), self.UPSERT_BATCH_SIZE):
                batch = vectors[batch_start:batch_start + self.UPSERT_BATCH_SIZE]
                self.index.upsert(
                    vectors=batch,
                    namespace=target_namespace
                )
                total_upserted += len(batch)
                logger.debug(
                    f"Upserted batch {batch_start}-{batch_start + len(batch)} "
                    f"({len(batch)} vectors)"
                )
            
            logger.info(
                f"✅ Added {total_upserted} chunks to Pinecone "
                f"(namespace: '{target_namespace or 'default'}')"
            )
            
        except Exception as e:
            logger.error(f"❌ Error adding chunks: {e}")
            raise RuntimeError(f"Failed to add chunks: {e}") from e

    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for most similar chunks to query\n        
        Args:
            query: Search query (text string or embedding vector)
            top_k: Number of results to return
            score_threshold: Optional minimum score threshold
            filter_dict: Optional metadata filter (e.g., {"doc_id": "123"})
            namespace: Optional namespace (uses instance namespace if None)
            include_metadata: Include metadata in results
            include_values: Include embedding vectors in results
            
        Returns:
            List of (chunk, score) tuples sorted by similarity (highest first)
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If search fails
            
        Example:
            >>> results = db.search(
            ...     query="machine learning",
            ...     top_k=5,
            ...     filter_dict={"category": "tech"},
            ...     score_threshold=0.7
            ... )
            >>> for chunk, score in results:
            ...     print(f"Score: {score:.3f} - {chunk.text[:50]}")
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")
        
        # Convert text to embedding if needed
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
        
        # Validate query dimension
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query dimension ({len(query_embedding)}) doesn't match "
                f"database dimension ({self.embedding_dim})"
            )
        
        # Use provided namespace or instance namespace
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            # Execute search
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filter_dict,
                namespace=target_namespace,
                include_metadata=include_metadata,
                include_values=include_values
            )
            
            # Convert results to chunks
            output = []
            for match in results.get('matches', []):
                score = float(match['score'])
                
                # Apply score threshold
                if score_threshold is not None and score < score_threshold:
                    continue
                
                metadata = match.get('metadata', {})
                
                # Reconstruct chunk
                chunk = DocumentChunk(
                    chunk_id=metadata.get('chunk_id', match['id']),
                    doc_id=metadata.get('doc_id', ''),
                    text=metadata.get('text', ''),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ['chunk_id', 'doc_id', 'text']
                    }
                )
                
                # Add embedding if included
                if include_values and 'values' in match:
                    chunk.embedding = np.array(match['values'], dtype=np.float32)
                
                output.append((chunk, score))
            
            logger.debug(f"Search returned {len(output)} results")
            return output
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e

    def search_by_id(
        self,
        chunk_id: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_self: bool = True,
        namespace: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find similar chunks to a specific chunk by ID\n        
        Args:
            chunk_id: Chunk identifier to use as query
            top_k: Number of results to return
            exclude_self: If True, exclude the query chunk from results
            namespace: Optional namespace
            
        Returns:
            List of (chunk, score) tuples
        """
        # Fetch the chunk
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            results = self.index.fetch(
                ids=[chunk_id],
                namespace=target_namespace
            )
            
            if chunk_id not in results.get('vectors', {}):
                logger.warning(f"Chunk ID '{chunk_id}' not found")
                return []
            
            # Get embedding
            vector_data = results['vectors'][chunk_id]
            embedding = np.array(vector_data['values'], dtype=np.float32)
            
            # Search with extra results if excluding self
            search_k = top_k + 1 if exclude_self else top_k
            search_results = self.search(
                query=embedding,
                top_k=search_k,
                namespace=namespace
            )
            
            # Filter out self if needed
            if exclude_self:
                search_results = [
                    (chunk, score) for chunk, score in search_results
                    if chunk.chunk_id != chunk_id
                ][:top_k]
            
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Error in search_by_id: {e}")
            raise RuntimeError(f"Failed to search by ID: {e}") from e

    # ============================================================================
    # DELETE OPERATIONS
    # ============================================================================

    def delete_by_ids(
        self,
        chunk_ids: List[str],
        namespace: Optional[str] = None
    ) -> int:
        """
        Delete chunks by their IDs\n        
        Args:
            chunk_ids: List of chunk identifiers to delete
            namespace: Optional namespace
            
        Returns:
            Number of chunks deleted
            
        Example:
            >>> deleted_count = db.delete_by_ids(["c1", "c2", "c3"])
        """
        if not chunk_ids:
            logger.warning("delete_by_ids called with empty list")
            return 0
        
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            # Delete from Pinecone in batches
            for batch_start in range(0, len(chunk_ids), self.DELETE_BATCH_SIZE):
                batch = chunk_ids[batch_start:batch_start + self.DELETE_BATCH_SIZE]
                self.index.delete(
                    ids=batch,
                    namespace=target_namespace
                )
                logger.debug(f"Deleted batch of {len(batch)} IDs")
            
            # Update local cache if enabled
            if self.cache_chunks:
                for chunk_id in chunk_ids:
                    if chunk_id in self._chunk_id_to_idx:
                        idx = self._chunk_id_to_idx[chunk_id]
                        chunk = self.chunks[idx]
                        
                        # Remove from doc index
                        if chunk.doc_id in self._doc_id_to_indices:
                            self._doc_id_to_indices[chunk.doc_id].remove(idx)
                            if not self._doc_id_to_indices[chunk.doc_id]:
                                del self._doc_id_to_indices[chunk.doc_id]
                        
                        # Mark as deleted (don't actually remove to preserve indices)
                        self.chunks[idx] = None
                        del self._chunk_id_to_idx[chunk_id]
            
            logger.info(f"✅ Deleted {len(chunk_ids)} chunks")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"❌ Error deleting by IDs: {e}")
            raise RuntimeError(f"Failed to delete by IDs: {e}") from e

    def delete_by_doc_id(
        self,
        doc_id: str,
        namespace: Optional[str] = None
    ) -> int:
        """
        Delete all chunks belonging to a specific document\n        
        Args:
            doc_id: Document identifier
            namespace: Optional namespace
            
        Returns:
            Number of chunks deleted
            
        Example:
            >>> deleted_count = db.delete_by_doc_id("document_123")
        """
        if not doc_id:
            logger.warning("delete_by_doc_id called with empty doc_id")
            return 0
        
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            # Find chunk IDs from local cache if available
            if self.cache_chunks and doc_id in self._doc_id_to_indices:
                chunk_ids = [
                    self.chunks[idx].chunk_id
                    for idx in self._doc_id_to_indices[doc_id]
                    if self.chunks[idx] is not None
                ]
                
                if chunk_ids:
                    return self.delete_by_ids(chunk_ids, namespace=namespace)
            
            # Otherwise use metadata filter
            self.index.delete(
                filter={'doc_id': doc_id},
                namespace=target_namespace
            )
            
            # Clean up local cache
            if self.cache_chunks and doc_id in self._doc_id_to_indices:
                for idx in self._doc_id_to_indices[doc_id]:
                    if self.chunks[idx] is not None:
                        chunk_id = self.chunks[idx].chunk_id
                        self.chunks[idx] = None
                        if chunk_id in self._chunk_id_to_idx:
                            del self._chunk_id_to_idx[chunk_id]
                del self._doc_id_to_indices[doc_id]
            
            logger.info(f"✅ Deleted all chunks from document: {doc_id}")
            return 0  # Pinecone doesn't return count for filter-based deletes
            
        except Exception as e:
            logger.error(f"❌ Error deleting by doc_id: {e}")
            raise RuntimeError(f"Failed to delete by doc_id: {e}") from e

    def delete_by_filter(
        self,
        filter_dict: Dict[str, Any],
        namespace: Optional[str] = None
    ) -> None:
        """
        Delete chunks matching metadata filter\n        
        Args:
            filter_dict: Metadata filter (e.g., {"category": "draft"})
            namespace: Optional namespace
            
        Example:
            >>> db.delete_by_filter({"status": "archived"})
        """
        if not filter_dict:
            logger.warning("delete_by_filter called with empty filter")
            return
        
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            self.index.delete(
                filter=filter_dict,
                namespace=target_namespace
            )
            
            logger.info(f"✅ Deleted chunks matching filter: {filter_dict}")
            
        except Exception as e:
            logger.error(f"❌ Error deleting by filter: {e}")
            raise RuntimeError(f"Failed to delete by filter: {e}") from e

    def delete_all(self, namespace: Optional[str] = None) -> None:
        """
        Delete all vectors from namespace\n
        
        Args:
            namespace: Optional namespace (uses instance namespace if None)
            
        Warning:
            This operation cannot be undone!
        """
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            self.index.delete(
                delete_all=True,
                namespace=target_namespace
            )
            
            # Clear local cache
            if self.cache_chunks:
                self.chunks.clear()
                self._chunk_id_to_idx.clear()
                self._doc_id_to_indices.clear()
            
            logger.warning(
                f"✅ Deleted all vectors from namespace: '{target_namespace or 'default'}'"
            )
            
        except Exception as e:
            logger.error(f"❌ Error deleting all: {e}")
            raise RuntimeError(f"Failed to delete all vectors: {e}") from e

    # ============================================================================
    # FETCH OPERATIONS
    # ============================================================================

    def fetch_by_ids(
        self,
        chunk_ids: List[str],
        namespace: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Fetch chunks by their IDs
        
        Args:
            chunk_ids: List of chunk identifiers
            namespace: Optional namespace
            
        Returns:
            List of DocumentChunk objects
        """
        if not chunk_ids:
            return []
        
        target_namespace = namespace if namespace is not None else self.namespace
        
        try:
            results = self.index.fetch(
                ids=chunk_ids,
                namespace=target_namespace
            )
            
            chunks = []
            for chunk_id, vector_data in results.get('vectors', {}).items():
                metadata = vector_data.get('metadata', {})
                
                chunk = DocumentChunk(
                    chunk_id=metadata.get('chunk_id', chunk_id),
                    doc_id=metadata.get('doc_id', ''),
                    text=metadata.get('text', ''),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ['chunk_id', 'doc_id', 'text']
                    }
                )
                
                # Add embedding
                if 'values' in vector_data:
                    chunk.embedding = np.array(vector_data['values'], dtype=np.float32)
                
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error fetching by IDs: {e}")
            raise RuntimeError(f"Failed to fetch by IDs: {e}") from e

    # ============================================================================
    # STATISTICS AND UTILITIES
    # ============================================================================

    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive database statistics\n        
        Args:
            namespace: Optional namespace for stats
            
        Returns:
            Dictionary containing database statistics
            
        Example:
            >>> stats = db.get_stats()
            >>> print(f"Total vectors: {stats['total_vector_count']}")
        """
        try:
            index_stats = self.index.describe_index_stats()
            
            # Get namespace-specific stats
            namespace_stats = {}
            if namespace:
                namespace_stats = index_stats.namespaces.get(namespace, {})
            elif self.namespace:
                namespace_stats = index_stats.namespaces.get(self.namespace, {})
            
            stats = {
                'index_name': self.index_name,
                'total_vector_count': index_stats.total_vector_count,
                'embedding_dim': self.embedding_dim,
                'distance_metric': self.distance_metric,
                'namespace': namespace or self.namespace or 'default',
                'namespace_vector_count': namespace_stats.get('vector_count', 0),
                'namespaces': list(index_stats.namespaces.keys()),
                'has_embedder': self.embedder is not None,
                'cache_enabled': self.cache_chunks,
                'cached_chunks': len([c for c in self.chunks if c is not None]) if self.cache_chunks else 0,
            }
            
            # Add local cache stats if enabled
            if self.cache_chunks:
                stats['unique_docs_cached'] = len(self._doc_id_to_indices)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {'error': str(e)}

    def list_namespaces(self) -> List[str]:
        """
        Get list of all namespaces in the index
        
        Returns:
            List of namespace names
        """
        try:
            stats = self.index.describe_index_stats()
            return list(stats.namespaces.keys())
        except Exception as e:
            logger.error(f"❌ Error listing namespaces: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a specific chunk by ID from local cache\n        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found in cache, None otherwise
        """
        if self.cache_chunks and chunk_id in self._chunk_id_to_idx:
            idx = self._chunk_id_to_idx[chunk_id]
            return self.chunks[idx]
        return None

    def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a document from local cache
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        if not self.cache_chunks or doc_id not in self._doc_id_to_indices:
            return []
        
        return [
            self.chunks[idx]
            for idx in self._doc_id_to_indices[doc_id]
            if self.chunks[idx] is not None
        ]

    def __len__(self) -> int:
        """Return total number of vectors in index"""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 0

    def __repr__(self) -> str:
        """String representation of database"""
        try:
            count = len(self)
        except:
            count = "unknown"
        
        return (
            f"PineconeVectorDatabase(index='{self.index_name}', "
            f"vectors={count}, dim={self.embedding_dim}, "
            f"metric={self.distance_metric}, namespace='{self.namespace or 'default'}')"
        )

    def __contains__(self, chunk_id: str) -> bool:
        """Check if chunk_id exists in local cache"""
        return self.cache_chunks and chunk_id in self._chunk_id_to_idx


    # ── Unified interface aliases ─────────────────────────────────────────────

    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks that belong to *doc_id*.\n
        Delegates to :meth:`delete_by_doc_id`.
        """
        return self.delete_by_doc_id(doc_id)

    def clear(self) -> int:
        """
        Delete all vectors from the index (all namespaces).\n
        """
        self.delete_all()
        return 0  # Pinecone doesn't return count for full deletes

    # ============================================================================
    # CONTEXT MANAGER SUPPORT
    # ============================================================================

    @contextmanager
    def batch_operation(self, namespace: Optional[str] = None):
        """
        Context manager for batch operations
        
        Args:
            namespace: Optional namespace for operations
            
        Example:
            >>> with db.batch_operation():
            ...     db.add(chunks1)
            ...     db.add(chunks2)
        """
        logger.debug("Starting batch operation")
        original_namespace = self.namespace
        if namespace is not None:
            self.namespace = namespace
        
        try:
            yield self
            logger.debug("Batch operation completed successfully")
        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            raise
        finally:
            self.namespace = original_namespace
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