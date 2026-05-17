from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None  # type: ignore
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass

# Add parent directory to path for imports
from .base import VectorDatabaseBase, sanitize_metadata
from ..chunks import DocumentChunk
logger = logging.getLogger(__name__)


class FAISSVectorDatabase(VectorDatabaseBase):
    """
    Production-Ready FAISS Vector Database Manager
    
    A comprehensive wrapper for FAISS that provides:
    - Multiple index types (Flat, IVF, HNSW) for different performance needs
    - Support for various distance metrics (cosine, L2, inner product)
    - Automatic embedding generation from text
    - Efficient similarity search with filtering
    - Persistent storage with save/load capabilities
    - Document-level operations (add, remove, update)
    
    Attributes:
        embedder: Optional embedding model for text-to-vector conversion
        embedding_dim: Dimensionality of embedding vectors
        index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        distance_metric: Similarity metric ('cosine', 'l2', 'ip')
        chunks: List of stored DocumentChunk objects
        
    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> embedder = SentenceTransformer('all-MiniLM-L6-v2')
        >>> db = FAISSVectorDatabase(
        ...     embedder=embedder,
        ...     index_type='flat',
        ...     distance_metric='cosine'
        ... )
        >>> # Add documents
        >>> chunk = DocumentChunk(
        ...     chunk_id="c1",
        ...     doc_id="d1",
        ...     text="Machine learning is fascinating"
        ... )
        >>> db.add_chunk(chunk)
        >>> 
        >>> # Search
        >>> results = db.search("AI technology", top_k=5)
        >>> for chunk, score in results:
        ...     print(f"Score: {score:.3f} - {chunk.text[:50]}")
    """

    # Supported configurations
    SUPPORTED_INDEX_TYPES = {'flat', 'ivf', 'hnsw'}
    SUPPORTED_METRICS = {'cosine', 'l2', 'ip'}
    
    # Default IVF parameters
    DEFAULT_IVF_CLUSTERS = 100
    DEFAULT_IVF_TRAIN_SIZE = 100
    
    # Default HNSW parameters
    DEFAULT_HNSW_M = 32  # Number of connections per layer

    def __init__(
        self,
        embedder: Optional[Any] = None,
        embedding_dim: Optional[int] = None,
        index_type: str = 'flat',
        distance_metric: str = 'cosine',
        ivf_clusters: Optional[int] = None,
        hnsw_m: Optional[int] = None,
    ) -> None:
        """
        Initialize FAISS Vector Database
        
        Args:
            embedder: Embedding model with .encode() method (e.g., SentenceTransformer)
            embedding_dim: Dimension of embeddings (required if embedder not provided)
            index_type: Type of FAISS index:
                - 'flat': Exact search (slower but accurate)
                - 'ivf': Inverted file index (faster for large datasets)
                - 'hnsw': Hierarchical NSW graph (fast approximate search)
            distance_metric: Similarity metric:
                - 'cosine': Cosine similarity (normalized vectors)
                - 'l2': Euclidean distance
                - 'ip': Inner product (dot product)
            ivf_clusters: Number of clusters for IVF index (default: 100)
            hnsw_m: Number of connections per layer for HNSW (default: 32)
            
        Raises:
            ValueError: If parameters are invalid or inconsistent
        """
        super().__init__(embedder)
        if faiss is None:
            raise ImportError(
                "faiss is required for FAISSVectorDatabase. "
                "Install it with: pip install faiss-cpu  (or faiss-gpu for GPU support)"
            )
        self.distance_metric = self._validate_metric(distance_metric)
        self.index_type = self._validate_index_type(index_type)
        
        # Determine embedding dimension
        self.embedding_dim = self._determine_embedding_dim(embedder, embedding_dim)
        
        # Index parameters
        self.ivf_clusters = ivf_clusters or self.DEFAULT_IVF_CLUSTERS
        self.hnsw_m = hnsw_m or self.DEFAULT_HNSW_M
        
        # Storage
        self.chunks: List[DocumentChunk] = []
        self._chunk_id_to_idx: Dict[str, int] = {}  # Fast lookup
        self._doc_id_to_indices: Dict[str, List[int]] = {}  # Fast document lookup
        self._total_vectors: int = 0
        
        # Create FAISS index
        self.index = self._create_index()
        
        logger.info(
            f"✅ FAISS database initialized: dim={self.embedding_dim}, "
            f"type={index_type}, metric={distance_metric}"
        )

    # ============================================================================
    # INITIALIZATION AND VALIDATION
    # ============================================================================

    def _validate_metric(self, metric: str) -> str:
        """Validate distance metric"""
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported distance metric: '{metric}'. "
                f"Supported metrics: {self.SUPPORTED_METRICS}"
            )
        return metric

    def _validate_index_type(self, index_type: str) -> str:
        """Validate index type"""
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: '{index_type}'. "
                f"Supported types: {self.SUPPORTED_INDEX_TYPES}"
            )
        return index_type

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
                # Test encode to extract dimension
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

    def _create_index(self) -> Any:
        """
        Create FAISS index based on configuration
        
        Returns:
            Initialized FAISS index
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Select base index function based on metric
        if self.distance_metric in ('cosine', 'ip'):
            base_index_func = faiss.IndexFlatIP
        elif self.distance_metric == 'l2':
            base_index_func = faiss.IndexFlatL2
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")
        
        # Create index based on type
        if self.index_type == 'flat':
            index = base_index_func(self.embedding_dim)
            logger.debug("Created Flat index for exact search")
            
        elif self.index_type == 'ivf':
            quantizer = base_index_func(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.ivf_clusters)
            logger.debug(f"Created IVF index with {self.ivf_clusters} clusters")
            
        elif self.index_type == 'hnsw':
            # HNSW in FAISS uses L2 internally; we handle cosine via normalization
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
            logger.debug(f"Created HNSW index with M={self.hnsw_m}")
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index

    # ============================================================================
    # EMBEDDING PROCESSING
    # ============================================================================

    def _normalize_if_needed(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vectors if using cosine similarity
        
        Args:
            embeddings: Input embedding vectors
            
        Returns:
            Normalized embeddings (if metric is cosine) or original embeddings
        """
        embeddings = embeddings.astype(np.float32)
        
        if self.distance_metric == 'cosine':
            # Normalize to unit length for cosine similarity
            normalized = embeddings.copy()
            faiss.normalize_L2(normalized)
            return normalized
        
        return embeddings

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for text using embedder
        
        Args:
            texts: List of text strings
            
        Returns:
            Embedding vectors as numpy array
            
        Raises:
            ValueError: If embedder is not available
            RuntimeError: If embedding computation fails
        """
        if self.embedder is None:
            raise ValueError(
                "No embedder available. Provide embedder in constructor or "
                "pass pre-computed embeddings to add() method."
            )
        
        try:
            embeddings = self.embedder.encode(texts)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings: {e}") from e

    def _validate_embedding_dimension(self, embeddings: np.ndarray) -> None:
        """
        Validate that embeddings match expected dimension
        
        Args:
            embeddings: Embedding vectors to validate
            
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

    # ============================================================================
    # ADD OPERATIONS
    # ============================================================================

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """
        Add a single document chunk to the database\n        
        Args:
            chunk: DocumentChunk object to add
            
        Raises:
            ValueError: If chunk is invalid or embedder is unavailable
            
        Example:
            >>> chunk = DocumentChunk(
            ...     chunk_id="c1",
            ...     doc_id="d1",
            ...     text="Example text"
            ... )
            >>> db.add_chunk(chunk)
        """
        if not isinstance(chunk, DocumentChunk):
            raise ValueError(f"Expected DocumentChunk, got {type(chunk)}")
        
        # Compute embedding if not present
        if chunk.embedding is None:
            chunk.embedding = self._compute_embeddings([chunk.text])[0]
        
        # Convert to array and add
        embedding = np.array([chunk.embedding])
        self.add(chunks=[chunk], embeddings=embedding)

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Add multiple document chunks with their embeddings\n
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Optional pre-computed embeddings array
                       If None, will be computed using embedder
            
        Raises:
            ValueError: If inputs are invalid or inconsistent
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
        
        # Validate inputs
        if not all(isinstance(c, DocumentChunk) for c in chunks):
            raise ValueError("All items in chunks must be DocumentChunk objects")
        
        # Compute embeddings if not provided
        if embeddings is None:
            texts = [chunk.text for chunk in chunks]
            embeddings = self._compute_embeddings(texts)
        
        # Validate dimensions
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must equal "
                f"number of embeddings ({len(embeddings)})"
            )
        
        self._validate_embedding_dimension(embeddings)
        
        try:
            # Normalize vectors if needed
            normalized_embeddings = self._normalize_if_needed(embeddings)
            
            # Train IVF index if needed and ready
            if self._should_train_ivf(len(normalized_embeddings)):
                self._train_ivf_index(normalized_embeddings)
            
            # Store chunks with embeddings
            start_idx = len(self.chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, normalized_embeddings)):
                chunk.embedding = embedding
                chunk.metadata = sanitize_metadata(chunk.metadata)
                self.chunks.append(chunk)
                
                # Update lookup tables
                current_idx = start_idx + i
                self._chunk_id_to_idx[chunk.chunk_id] = current_idx
                
                if chunk.doc_id not in self._doc_id_to_indices:
                    self._doc_id_to_indices[chunk.doc_id] = []
                self._doc_id_to_indices[chunk.doc_id].append(current_idx)
            
            # Add to FAISS index
            self.index.add(normalized_embeddings)
            self._total_vectors += len(normalized_embeddings)
            
            logger.info(f"✅ Added {len(chunks)} chunks. Total: {self._total_vectors}")
            
        except Exception as e:
            logger.error(f"❌ Error adding chunks: {e}")
            raise RuntimeError(f"Failed to add chunks: {e}") from e

    def _should_train_ivf(self, new_vectors: int) -> bool:
        """Check if IVF index should be trained"""
        return (
            self.index_type == 'ivf' and
            not self.index.is_trained and
            (self._total_vectors + new_vectors) >= self.DEFAULT_IVF_TRAIN_SIZE
        )

    def _train_ivf_index(self, embeddings: np.ndarray) -> None:
        """Train IVF index with embeddings"""
        try:
            self.index.train(embeddings)
            logger.info(f"✅ IVF index trained with {len(embeddings)} vectors")
        except Exception as e:
            logger.error(f"❌ Failed to train IVF index: {e}")
            raise RuntimeError(f"IVF training failed: {e}") from e

    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        doc_id_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for most similar chunks to query\n        
        Args:
            query: Search query (text string or embedding vector)
            top_k: Number of results to return
            score_threshold: Optional minimum score threshold
            doc_id_filter: Optional filter by document ID(s)
            
        Returns:
            List of (chunk, score) tuples sorted by similarity (highest first)
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If search fails
            
        Example:
            >>> results = db.search("machine learning", top_k=5, score_threshold=0.5)
            >>> for chunk, score in results:
            ...     print(f"Score: {score:.3f} - {chunk.text[:50]}")
        """
        if self._total_vectors == 0:
            logger.warning("⚠️ Database is empty, no results to return")
            return []
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")
        
        # Convert text to embedding if needed
        if isinstance(query, str):
            query_embedding = self._compute_embeddings([query])[0]
        else:
            query_embedding = np.asarray(query, dtype=np.float32)
        
        # Validate query dimension
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension ({query_embedding.shape[0]}) doesn't match "
                f"database dimension ({self.embedding_dim})"
            )
        
        try:
            # Normalize query vector
            query_vec = self._normalize_if_needed(query_embedding.reshape(1, -1))
            
            # Search FAISS index
            actual_k = min(top_k, self._total_vectors)
            
            # For IVF, we might need more results before filtering
            search_k = actual_k * 2 if doc_id_filter else actual_k
            
            distances, indices = self.index.search(query_vec, search_k)
            
            # Collect and filter results
            results = []
            for k, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                if idx >= len(self.chunks):  # Safety check
                    logger.warning(f"Invalid index returned by FAISS: {idx}")
                    continue
                
                chunk = self.chunks[idx]
                score = float(distances[0][k])
                
                # Apply score threshold
                if score_threshold is not None and score < score_threshold:
                    continue
                
                # Apply document filter
                if doc_id_filter is not None:
                    if isinstance(doc_id_filter, str):
                        if chunk.doc_id != doc_id_filter:
                            continue
                    elif chunk.doc_id not in doc_id_filter:
                        continue
                
                results.append((chunk, score))
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e

    def search_by_doc_id(
        self,
        doc_id: str,
        top_k: int = 5,
        exclude_same_doc: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find similar chunks to all chunks from a specific document\n        
        Args:
            doc_id: Document ID to use as query source
            top_k: Number of results per query chunk
            exclude_same_doc: If True, exclude results from same document
            
        Returns:
            List of (chunk, score) tuples
        """
        if doc_id not in self._doc_id_to_indices:
            logger.warning(f"Document ID '{doc_id}' not found in database")
            return []
        
        # Get all chunks from document
        doc_indices = self._doc_id_to_indices[doc_id]
        all_results = []
        
        for idx in doc_indices:
            chunk = self.chunks[idx]
            results = self.search(
                query=chunk.embedding,
                top_k=top_k + len(doc_indices) if exclude_same_doc else top_k
            )
            
            # Filter out same document if needed
            if exclude_same_doc:
                results = [(c, s) for c, s in results if c.doc_id != doc_id][:top_k]
            
            all_results.extend(results)
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for chunk, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_results.append((chunk, score))
        
        return unique_results[:top_k]

    # ============================================================================
    # DELETE OPERATIONS
    # ============================================================================

    def remove_by_chunk_id(self, chunk_id: str) -> bool:
        """
        Remove a specific chunk by its ID\n        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if chunk was found and removed, False otherwise
        """
        if chunk_id not in self._chunk_id_to_idx:
            logger.warning(f"Chunk ID '{chunk_id}' not found")
            return False
        
        idx = self._chunk_id_to_idx[chunk_id]
        chunk = self.chunks[idx]
        
        # Remove from lookup tables
        del self._chunk_id_to_idx[chunk_id]
        self._doc_id_to_indices[chunk.doc_id].remove(idx)
        
        if not self._doc_id_to_indices[chunk.doc_id]:
            del self._doc_id_to_indices[chunk.doc_id]
        
        # Mark for rebuild
        self._rebuild_index()
        
        logger.info(f"✅ Removed chunk: {chunk_id}")
        return True

    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a specific document\n        
        Args:
            doc_id: Document identifier
            
        Returns:
            Number of chunks removed
            
        Example:
            >>> removed_count = db.remove_by_doc_id("doc_123")
            >>> print(f"Removed {removed_count} chunks")
        """
        if doc_id not in self._doc_id_to_indices:
            logger.warning(f"Document ID '{doc_id}' not found")
            return 0
        
        # Get indices to remove
        indices_to_remove = sorted(self._doc_id_to_indices[doc_id], reverse=True)
        
        # Remove from chunks list and lookup tables
        for idx in indices_to_remove:
            chunk = self.chunks[idx]
            del self._chunk_id_to_idx[chunk.chunk_id]
            del self.chunks[idx]
        
        # Remove document from doc index
        del self._doc_id_to_indices[doc_id]
        
        # Rebuild index and lookup tables
        self._rebuild_index()
        
        removed_count = len(indices_to_remove)
        logger.info(f"✅ Removed {removed_count} chunks from document: {doc_id}")
        return removed_count

    def clear(self) -> int:
        """
        Remove all chunks from the database\n
        Returns:
            Number of chunks removed
        """
        count = self._total_vectors
        
        self.chunks.clear()
        self._chunk_id_to_idx.clear()
        self._doc_id_to_indices.clear()
        self._total_vectors = 0
        
        # Recreate empty index
        self.index = self._create_index()
        
        logger.warning(f"✅ Cleared database: {count} chunks removed")
        return count

    def _rebuild_index(self) -> None:
        """
        Rebuild FAISS index from existing chunks
        
        This is called after deletions to maintain index consistency
        """
        if not self.chunks:
            self.index = self._create_index()
            self._total_vectors = 0
            self._chunk_id_to_idx.clear()
            self._doc_id_to_indices.clear()
            logger.debug("Rebuilt empty index")
            return
        
        # Recreate index
        self.index = self._create_index()
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        
        # Train if needed
        if self.index_type == 'ivf' and len(embeddings) >= self.DEFAULT_IVF_TRAIN_SIZE:
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        self._total_vectors = len(self.chunks)
        
        # Rebuild lookup tables
        self._chunk_id_to_idx.clear()
        self._doc_id_to_indices.clear()
        
        for idx, chunk in enumerate(self.chunks):
            self._chunk_id_to_idx[chunk.chunk_id] = idx
            
            if chunk.doc_id not in self._doc_id_to_indices:
                self._doc_id_to_indices[chunk.doc_id] = []
            self._doc_id_to_indices[chunk.doc_id].append(idx)
        
        logger.info(f"✅ Index rebuilt with {self._total_vectors} vectors")

    # ============================================================================
    # PERSISTENCE
    # ============================================================================

    def save(self, path: Union[str, Path]) -> None:
        """
        Save database to disk\n        
        Saves three files:
        - index.faiss: FAISS index
        - chunks.pkl: Document chunks
        - metadata.pkl: Database configuration
        
        Args:
            path: Directory path to save database
            
        Raises:
            RuntimeError: If save operation fails
            
        Example:
            >>> db.save("./my_database")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(path / "index.faiss"))
            logger.debug("Saved FAISS index")
            
            # Prepare chunks data
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
                }
                chunks_data.append(chunk_dict)
            
            # Save chunks
            with open(path / "chunks.pkl", 'wb') as f:
                pickle.dump(chunks_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved {len(chunks_data)} chunks")
            
            # Save metadata
            metadata = {
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'distance_metric': self.distance_metric,
                'total_vectors': self._total_vectors,
                'ivf_clusters': self.ivf_clusters,
                'hnsw_m': self.hnsw_m,
            }
            with open(path / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Saved metadata")
            
            logger.info(f"✅ Database saved to: {path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving database: {e}")
            raise RuntimeError(f"Failed to save database: {e}") from e

    @classmethod
    def load(cls, path: Union[str, Path], embedder: Optional[Any] = None) -> 'FAISSVectorDatabase':
        """
        Load database from disk\n        
        Args:
            path: Directory path containing saved database
            embedder: Optional embedder for computing new embeddings
            
        Returns:
            Loaded FAISSVectorDatabase instance
            
        Raises:
            FileNotFoundError: If path doesn't exist
            RuntimeError: If load operation fails
            
        Example:
            >>> db = FAISSVectorDatabase.load("./my_database", embedder=embedder)
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Database path does not exist: {path}")
        
        required_files = ['index.faiss', 'chunks.pkl', 'metadata.pkl']
        missing_files = [f for f in required_files if not (path / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {missing_files}. "
                f"Database at {path} is incomplete."
            )
        
        try:
            # Load metadata
            with open(path / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            logger.debug("Loaded metadata")
            
            # Create database instance
            db = cls(
                embedder=embedder,
                embedding_dim=metadata['embedding_dim'],
                index_type=metadata['index_type'],
                distance_metric=metadata.get('distance_metric', 'cosine'),
                ivf_clusters=metadata.get('ivf_clusters'),
                hnsw_m=metadata.get('hnsw_m'),
            )
            
            # Load FAISS index
            db.index = faiss.read_index(str(path / "index.faiss"))
            db._total_vectors = metadata['total_vectors']
            logger.debug("Loaded FAISS index")
            
            # Load chunks
            with open(path / "chunks.pkl", 'rb') as f:
                chunks_data = pickle.load(f)
            
            for idx, chunk_dict in enumerate(chunks_data):
                chunk = DocumentChunk(
                    chunk_id=chunk_dict['chunk_id'],
                    doc_id=chunk_dict['doc_id'],
                    text=chunk_dict['text'],
                    metadata=chunk_dict.get('metadata', {})
                )
                
                if chunk_dict.get('embedding'):
                    chunk.embedding = np.array(chunk_dict['embedding'], dtype=np.float32)
                
                db.chunks.append(chunk)
                
                # Rebuild lookup tables
                db._chunk_id_to_idx[chunk.chunk_id] = idx
                if chunk.doc_id not in db._doc_id_to_indices:
                    db._doc_id_to_indices[chunk.doc_id] = []
                db._doc_id_to_indices[chunk.doc_id].append(idx)
            
            logger.info(f"✅ Database loaded from: {path} ({len(chunks_data)} chunks)")
            return db
            
        except Exception as e:
            logger.error(f"❌ Error loading database: {e}")
            raise RuntimeError(f"Failed to load database: {e}") from e

    # ============================================================================
    # STATISTICS AND UTILITIES
    # ============================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics\n        
        Returns:
            Dictionary containing database statistics
            
        Example:
            >>> stats = db.get_stats()
            >>> print(f"Total chunks: {stats['total_chunks']}")
            >>> print(f"Unique documents: {stats['unique_docs']}")
        """
        unique_docs = len(self._doc_id_to_indices)
        
        return {
            'total_chunks': self._total_vectors,
            'unique_docs': unique_docs,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'distance_metric': self.distance_metric,
            'ivf_clusters': self.ivf_clusters if self.index_type == 'ivf' else None,
            'hnsw_m': self.hnsw_m if self.index_type == 'hnsw' else None,
            'is_trained': self.index.is_trained if self.index_type == 'ivf' else True,
            'has_embedder': self.embedder is not None,
        }

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific chunk by ID\n        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        if chunk_id in self._chunk_id_to_idx:
            idx = self._chunk_id_to_idx[chunk_id]
            return self.chunks[idx]
        return None

    def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific document\n        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        if doc_id not in self._doc_id_to_indices:
            return []
        
        indices = self._doc_id_to_indices[doc_id]
        return [self.chunks[idx] for idx in indices]

    def list_document_ids(self) -> List[str]:
        """Get list of all unique document IDs in database"""
        return list(self._doc_id_to_indices.keys())

    def __len__(self) -> int:
        """Return total number of vectors in database"""
        return self._total_vectors

    def __repr__(self) -> str:
        """String representation of database"""
        return (
            f"FAISSVectorDatabase(vectors={self._total_vectors}, "
            f"docs={len(self._doc_id_to_indices)}, "
            f"dim={self.embedding_dim}, type={self.index_type}, "
            f"metric={self.distance_metric})"
        )

    def __contains__(self, chunk_id: str) -> bool:
        """Check if chunk_id exists in database"""
        return chunk_id in self._chunk_id_to_idx

    # ── Async API ────────────────────────────────────────────────────

    async def asearch(self, query: str, top_k: int = 10,
                      score_threshold: float = 0.0, **kwargs):
        """
        Async vector search — runs blocking search in a thread pool.
        """
        import asyncio
        return await asyncio.to_thread(
            self.search, query, top_k, score_threshold, **kwargs
        )

    async def aadd(self, chunks, **kwargs):
        """
        Async add chunks — runs blocking add in a thread pool.
        """
        import asyncio
        return await asyncio.to_thread(self.add, chunks, **kwargs)

    async def aremove_by_doc_id(self, doc_id: str) -> int:
        """Async remove document"""
        import asyncio
        return await asyncio.to_thread(self.remove_by_doc_id, doc_id)

    async def asave(self, path) -> None:
        """Async save """
        import asyncio
        return await asyncio.to_thread(self.save, path)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False