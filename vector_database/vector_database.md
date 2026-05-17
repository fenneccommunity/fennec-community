# Vector Database — Enterprise API Reference

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Module-Level Function: `sanitize_metadata`](#module-level-function-sanitize_metadata)
5. [Abstract Base Class: `VectorDatabaseBase`](#abstract-base-class-vectordatabasebase)
   - [`__init__`](#__init__-base)
   - [Abstract Interface](#abstract-interface)
     - [`add`](#add-abstract)
     - [`search`](#search-abstract)
     - [`remove_by_doc_id`](#remove_by_doc_id-abstract)
     - [`get_stats`](#get_stats-abstract)
     - [`__len__`](#__len__-abstract)
   - [Optional Interface (with default implementations)](#optional-interface-with-default-implementations)
     - [`clear`](#clear-base)
     - [`get_chunk_by_id`](#get_chunk_by_id-base)
     - [`get_chunks_by_doc_id`](#get_chunks_by_doc_id-base)
     - [`save`](#save-base)
     - [`load`](#load-base)
   - [Async Interface](#async-interface)
     - [`asearch`](#asearch-base)
     - [`aadd`](#aadd-base)
     - [`aremove_by_doc_id`](#aremove_by_doc_id-base)
6. [Class: `FAISSVectorDatabase`](#class-faissvectordatabase)
   - [`__init__`](#__init__-faiss)
   - [Ingestion](#ingestion-faiss)
     - [`add_chunk`](#add_chunk-faiss)
     - [`add`](#add-faiss)
   - [Search](#search-faiss)
     - [`search`](#search-faiss-1)
     - [`search_by_doc_id`](#search_by_doc_id)
   - [Deletion](#deletion-faiss)
     - [`remove_by_chunk_id`](#remove_by_chunk_id)
     - [`remove_by_doc_id`](#remove_by_doc_id-faiss)
     - [`clear`](#clear-faiss)
   - [Retrieval & Lookup](#retrieval--lookup-faiss)
     - [`get_chunk_by_id`](#get_chunk_by_id-faiss)
     - [`get_chunks_by_doc_id`](#get_chunks_by_doc_id-faiss)
     - [`list_document_ids`](#list_document_ids)
   - [Persistence](#persistence-faiss)
     - [`save`](#save-faiss)
     - [`load`](#load-faiss)
   - [Statistics](#statistics-faiss)
     - [`get_stats`](#get_stats-faiss)
   - [Async API](#async-api-faiss)
     - [`asearch`](#asearch-faiss)
     - [`aadd`](#aadd-faiss)
     - [`aremove_by_doc_id`](#aremove_by_doc_id-faiss)
     - [`asave`](#asave-faiss)
7. [Class: `ChromaVectorDatabase`](#class-chromavectordatabase)
   - [`__init__`](#__init__-chroma)
   - [Ingestion](#ingestion-chroma)
     - [`add`](#add-chroma)
   - [Search](#search-chroma)
     - [`search`](#search-chroma-1)
   - [Deletion](#deletion-chroma)
     - [`delete_by_ids`](#delete_by_ids-chroma)
     - [`delete_by_filter`](#delete_by_filter-chroma)
     - [`remove_by_doc_id`](#remove_by_doc_id-chroma)
     - [`clear_collection`](#clear_collection)
   - [Retrieval & Lookup](#retrieval--lookup-chroma)
     - [`get_by_ids`](#get_by_ids)
   - [Statistics](#statistics-chroma)
     - [`stats`](#stats-chroma)
     - [`get_stats`](#get_stats-chroma)
   - [Context Manager](#context-manager-chroma)
     - [`batch_operation`](#batch_operation-chroma)
   - [Async API](#async-api-chroma)
     - [`asearch`](#asearch-chroma)
     - [`aadd`](#aadd-chroma)
8. [Class: `PineconeVectorDatabase`](#class-pineconevectordatabase)
   - [`__init__`](#__init__-pinecone)
   - [Ingestion](#ingestion-pinecone)
     - [`add_chunk`](#add_chunk-pinecone)
     - [`add`](#add-pinecone)
   - [Search](#search-pinecone)
     - [`search`](#search-pinecone-1)
     - [`search_by_id`](#search_by_id-pinecone)
   - [Deletion](#deletion-pinecone)
     - [`delete_by_ids`](#delete_by_ids-pinecone)
     - [`delete_by_doc_id`](#delete_by_doc_id-pinecone)
     - [`delete_by_filter`](#delete_by_filter-pinecone)
     - [`delete_all`](#delete_all-pinecone)
     - [`remove_by_doc_id`](#remove_by_doc_id-pinecone)
     - [`clear`](#clear-pinecone)
   - [Fetch Operations](#fetch-operations-pinecone)
     - [`fetch_by_ids`](#fetch_by_ids-pinecone)
   - [Retrieval & Lookup](#retrieval--lookup-pinecone)
     - [`get_chunk_by_id`](#get_chunk_by_id-pinecone)
     - [`get_chunks_by_doc_id`](#get_chunks_by_doc_id-pinecone)
   - [Statistics & Info](#statistics--info-pinecone)
     - [`get_stats`](#get_stats-pinecone)
     - [`list_namespaces`](#list_namespaces-pinecone)
   - [Context Manager](#context-manager-pinecone)
     - [`batch_operation`](#batch_operation-pinecone)
   - [Async API](#async-api-pinecone)
     - [`asearch`](#asearch-pinecone)
     - [`aadd`](#aadd-pinecone)
9. [Backend Comparison Matrix](#backend-comparison-matrix)
10. [Unified Interface Contract](#unified-interface-contract)
11. [Distance Metric Reference](#distance-metric-reference)
12. [Metadata Sanitisation Rules](#metadata-sanitisation-rules)
13. [Installation Guide](#installation-guide)
14. [Complete Examples](#complete-examples)

---

## Overview

`vector_database` is a **unified, multi-backend vector store abstraction layer** that provides a single consistent API across three production-grade vector database engines: **FAISS** (local, CPU/GPU), **ChromaDB** (local/cloud), and **Pinecone** (fully managed cloud). All three backends implement the same `VectorDatabaseBase` contract, so application code (including the `GraphRAG`, `MultiHopRAG`, and `FederatedRAG` layers) can swap backends without any changes at the call site.

Key capabilities at a glance:

| Capability | FAISS | Chroma | Pinecone |
|---|---|---|---|
| Storage | Local in-memory + disk | Local/Persistent | Fully managed cloud |
| Scalability | Single machine | Single machine | Horizontally unlimited |
| GPU support | ✅ | ❌ | N/A (cloud) |
| Metadata filtering | doc_id filter only | Rich `where` filters | Full expression language |
| Multi-tenancy | ❌ | tenant_id isolation | Namespaces |
| Async API | ✅ | ✅ | ✅ |
| Local persistence | ✅ save/load | ✅ PersistentClient | ✅ Always persistent |
| Index types | Flat / IVF / HNSW | HNSW (auto) | Managed |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  vector_database                         │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │              VectorDatabaseBase (ABC)              │   │
│  │  add · search · remove_by_doc_id · get_stats      │   │
│  │  clear · get_chunk_by_id · get_chunks_by_doc_id   │   │
│  │  save · load · asearch · aadd · aremove_by_doc_id │   │
│  └───────────┬──────────────────┬──────────────────┘   │
│              │                  │                 │      │
│   ┌──────────▼──┐   ┌──────────▼──┐   ┌─────────▼──┐  │
│   │    FAISS    │   │   Chroma    │   │  Pinecone  │  │
│   │ Flat/IVF/   │   │  HNSW +    │   │  Serverless│  │
│   │ HNSW index  │   │ PersistDB  │   │  / Pod     │  │
│   └─────────────┘   └────────────┘   └────────────┘  │
│                                                          │
│  sanitize_metadata()  ←  called by all three backends   │
└─────────────────────────────────────────────────────────┘
```

**Unified data flow:**

```
User Code
    │
    ├─ add(chunks)               →  backend.add()  →  index upsert
    ├─ search(query, top_k)      →  backend.search() →  [(DocumentChunk, float), ...]
    ├─ remove_by_doc_id(doc_id)  →  backend.remove_by_doc_id() →  int (removed count)
    └─ get_stats()               →  backend.get_stats() →  Dict[str, Any]
```

---

## Quick Start

```python
# ── Option A: FAISS (local, no cloud account needed) ───────────────────────
from fennec_community.vector_database import FAISSVectorDatabase
from fennec_community.embeddings import GeminiEmbedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = FAISSVectorDatabase(embedder=embedder, index_type="flat", distance_metric="cosine")

# ── Option B: ChromaDB (local persistent) ──────────────────────────────────
from fennec_community.vector_database import ChromaVectorDatabase
db = ChromaVectorDatabase(
    embedder=embedder,
    collection_name="my_docs",
    persist_directory="./chroma_db",
)

# ── Option C: Pinecone (cloud) ─────────────────────────────────────────────
from fennec_community.vector_database import PineconeVectorDatabase
db = PineconeVectorDatabase(
    embedder=embedder,
    index_name="production-index",
    api_key="your-key",          # or set PINECONE_API_KEY env var
    distance_metric="cosine",
)

# ── Unified usage (identical for all three backends) ──────────────────────
from fennec_community.chunks import DocumentChunk

chunks = [
    DocumentChunk(chunk_id="c1", doc_id="doc_001", text="Machine learning is a subset of AI."),
    DocumentChunk(chunk_id="c2", doc_id="doc_001", text="Deep learning uses neural networks."),
]
db.add(chunks)

results = db.search("AI and neural networks", top_k=5)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text}")

db.remove_by_doc_id("doc_001")
print(db.get_stats())
```

---

## Module-Level Function: `sanitize_metadata`

```python
from fennec_community.vector_database import sanitize_metadata
```

### `sanitize_metadata`

```python
sanitize_metadata(metadata: Dict) -> Dict
```

**Purpose:** Converts any metadata dictionary into a format that is safe and compatible with **all three backends** simultaneously (FAISS, ChromaDB, and Pinecone). This is the single authoritative sanitisation implementation; every backend calls it internally so you never have to worry about cross-backend compatibility.

This function is **always called automatically** by `add()` in every backend. You only need to call it manually if you pre-process metadata before passing it to the database.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `metadata` | `Dict` | Raw metadata dictionary attached to a `DocumentChunk`. Can contain any Python values. |

**Returns:** `Dict` — a sanitised dictionary where all values are backend-compatible primitives.

**Transformation rules:**

| Input value type | Output | Why |
|---|---|---|
| `str` / `int` / `float` / `bool` | Unchanged | Supported natively by all backends |
| `None` | **Key removed entirely** | ChromaDB rejects `None` values; removal is the safest cross-backend behaviour |
| `list[str]` | Joined as `", "` comma-separated string | ChromaDB rejects list values entirely; the string can be re-parsed for Pinecone `$in` filters if needed |
| `list` (mixed types) | Joined as `", "` string (each item stringified) | Same reason as above |
| `dict` / `set` / other | JSON-serialised string via `json.dumps` | Ensures portability; falls back to `str()` if JSON serialisation fails |

**Example:**

```python
from fennec_community.vector_database import sanitize_metadata

raw = {
    "title":    "Introduction to ML",
    "tags":     ["ai", "machine-learning", "python"],
    "score":    0.95,
    "verified": True,
    "source":   None,                    # ← will be dropped
    "extras":   {"nested": "dict"},      # ← JSON-serialised
}

clean = sanitize_metadata(raw)
# {
#   "title":   "Introduction to ML",
#   "tags":    "ai, machine-learning, python",   ← list joined
#   "score":   0.95,
#   "verified": True,
#   "extras":  '{"nested": "dict"}'              ← JSON string
#   # "source" is absent — None was dropped
# }
```

---

## Abstract Base Class: `VectorDatabaseBase`

```python
from fennec_community.vector_database import VectorDatabaseBase
```

The abstract foundation of the module. Defines the **unified contract** that all three backends implement. Use this class as the type annotation throughout your application code to keep it backend-agnostic.

```python
db: VectorDatabaseBase = FAISSVectorDatabase(...)  # or Chroma or Pinecone
```

---

### `__init__` _(Base)_

```python
VectorDatabaseBase(embedder: Optional[Any] = None)
```

**Purpose:** Base constructor that stores the shared embedder reference. Called automatically by each backend's own `__init__` via `super().__init__(embedder)`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embedder` | `Optional[Any]` | `None` | Any object exposing `.encode(texts: List[str]) -> np.ndarray` — e.g., `SentenceTransformer`, `GeminiEmbedder`, or any custom encoder. Pass `None` when you always supply pre-computed embeddings. |

---

## Abstract Interface

These five methods **must** be implemented by every backend. They form the minimal API surface that all orchestration layers (RAGSystem, GraphRAG, etc.) rely upon.

---

### `add` _(abstract)_

```python
@abstractmethod
db.add(
    chunks:     List[DocumentChunk],
    embeddings: Optional[np.ndarray] = None,
) -> None
```

**Purpose:** Adds (or upserts) a list of document chunks to the backend. If `embeddings` are not provided, the backend calls the configured embedder to compute them.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunks` | `List[DocumentChunk]` | — | List of `DocumentChunk` objects to add. Each must have `chunk_id`, `doc_id`, and `text`. |
| `embeddings` | `Optional[np.ndarray]` | `None` | Pre-computed embeddings of shape `(N, dim)`, `dtype=float32`. Computed automatically when `None`. |

**Returns:** `None`

**Backend notes:**
- **FAISS**: Stores vectors in the FAISS index; updates `_chunk_id_to_idx` and `_doc_id_to_indices` lookup tables.
- **Chroma**: Upserts in configurable batches; automatically applies `sanitize_metadata`.
- **Pinecone**: Upserts in batches of 100 (hard limit); caches chunks locally when `cache_chunks=True`.

---

### `search` _(abstract)_

```python
@abstractmethod
db.search(
    query:           Union[str, np.ndarray],
    top_k:           int = 5,
    score_threshold: Optional[float] = None,
    **kwargs,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Returns the `top_k` most similar chunks to the query. All backends return `(DocumentChunk, float)` tuples sorted by score descending.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `Union[str, np.ndarray]` | — | Text query (encoded on-the-fly) or a pre-computed 1-D float32 embedding vector. |
| `top_k` | `int` | `5` | Maximum number of results to return. |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score filter. Results below this value are dropped. Pass `None` to disable. |
| `**kwargs` | | | Backend-specific parameters. FAISS: `doc_id_filter`. Chroma: `filters`. Pinecone: `filter_dict`, `namespace`, `include_values`. |

**Returns:** `List[Tuple[DocumentChunk, float]]` — list of `(chunk, score)` pairs, sorted highest score first. Empty list when the database contains no vectors.

---

### `remove_by_doc_id` _(abstract)_

```python
@abstractmethod
db.remove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Removes all chunks that belong to the specified document. This is the unified deletion method across all backends, regardless of how each backend calls it internally.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | `str` | Document identifier. All chunks whose `doc_id` field matches will be removed. |

**Returns:** `int` — number of chunks removed. Returns `0` if the document was not found.

**Backend naming aliases:**
- FAISS → native `remove_by_doc_id`
- Pinecone → delegates to `delete_by_doc_id`
- Chroma → delegates to `delete_by_filter({"doc_id": doc_id})`

---

### `get_stats` _(abstract)_

```python
@abstractmethod
db.get_stats() -> Dict[str, Any]
```

**Purpose:** Returns a dictionary of backend statistics. Every backend guarantees at minimum four standardised keys.

**Returns:** `Dict[str, Any]` — guaranteed keys:

| Key | Type | Description |
|---|---|---|
| `total_vectors` | `int` | Total number of stored vectors. |
| `embedding_dim` | `int` | Dimensionality of stored vectors. |
| `distance_metric` | `str` | Active distance metric (e.g., `"cosine"`). |
| `has_embedder` | `bool` | Whether an embedder is configured. |

Additional backend-specific keys are listed in each backend's `get_stats` section.

---

### `__len__` _(abstract)_

```python
@abstractmethod
len(db) -> int
```

**Purpose:** Returns the total number of vectors stored in the database. Enables standard Python `len()` usage.

**Returns:** `int`

**Example:**

```python
print(f"Database has {len(db)} vectors")
```

---

## Optional Interface (with default implementations)

These methods have default implementations in the base class. Backends may override them to provide richer behaviour.

---

### `clear` _(Base)_

```python
db.clear() -> int
```

**Purpose:** Removes **all** chunks from the database. The base class raises `NotImplementedError` by default; backends that support clearing override this.

**Returns:** `int` — number of chunks removed.

**Raises:** `NotImplementedError` if the backend does not support clearing (base default).

> ⚠️ This operation cannot be undone.

---

### `get_chunk_by_id` _(Base)_

```python
db.get_chunk_by_id(chunk_id: str) -> Optional[DocumentChunk]
```

**Purpose:** Retrieves a single chunk by its `chunk_id` from a local cache. The base class returns `None` (no local cache). FAISS and Pinecone (when `cache_chunks=True`) override this with O(1) lookup.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `chunk_id` | `str` | The chunk's unique identifier. |

**Returns:** `DocumentChunk` if found; `None` otherwise.

---

### `get_chunks_by_doc_id` _(Base)_

```python
db.get_chunks_by_doc_id(doc_id: str) -> List[DocumentChunk]
```

**Purpose:** Retrieves all chunks belonging to a specific document from local cache. Base class returns `[]`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | `str` | Document identifier. |

**Returns:** `List[DocumentChunk]` — all chunks for that document. Empty list if not found or no local cache.

---

### `save` _(Base)_

```python
db.save(path: Union[str, Path]) -> None
```

**Purpose:** Persists the database to disk. Base class logs a warning (cloud backends like Pinecone are always persistent and do not need a local save). FAISS overrides this with a full three-file serialisation.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[str, Path]` | Directory path to write database files into. |

**Returns:** `None`

---

### `load` _(Base)_

```python
@classmethod
VectorDatabaseBase.load(path: Union[str, Path], **kwargs) -> VectorDatabaseBase
```

**Purpose:** Loads a previously saved database from disk. Base class raises `NotImplementedError`. FAISS provides a full implementation.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[str, Path]` | Directory path written by `save()`. |
| `**kwargs` | | Backend-specific arguments (e.g., `embedder` for FAISS). |

**Returns:** A fully restored `VectorDatabaseBase` instance.

---

## Async Interface

All three async methods are defined at base level. They wrap the synchronous equivalents via `asyncio.to_thread`. Backends with native async support can override them.

---

### `asearch` _(Base)_

```python
async db.asearch(
    query:           Union[str, np.ndarray],
    top_k:           int = 10,
    score_threshold: Optional[float] = None,
    **kwargs,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Async version of `search`. Runs `search()` in a thread pool so the event loop is never blocked.

**Parameters / Returns:** Identical to `search`.

---

### `aadd` _(Base)_

```python
async db.aadd(
    chunks:     List[DocumentChunk],
    embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> None
```

**Purpose:** Async version of `add`. Runs `add()` in a thread pool.

**Parameters / Returns:** Identical to `add`.

---

### `aremove_by_doc_id` _(Base)_

```python
async db.aremove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Async version of `remove_by_doc_id`. Runs `remove_by_doc_id()` in a thread pool.

**Parameters / Returns:** Identical to `remove_by_doc_id`.

---

---

## Class: `FAISSVectorDatabase`

```python
from fennec_community.vector_database import FAISSVectorDatabase
```

A production-ready, local FAISS vector store with three index types (Flat, IVF, HNSW), cosine/L2/inner-product distance metrics, O(1) chunk/doc lookup tables, and full disk persistence via three-file serialisation.

**Install:** `pip install faiss-cpu` (or `faiss-gpu` for GPU support)

---

### `__init__` _(FAISS)_

```python
FAISSVectorDatabase(
    embedder:        Optional[Any]  = None,
    embedding_dim:   Optional[int]  = None,
    index_type:      str            = "flat",
    distance_metric: str            = "cosine",
    ivf_clusters:    Optional[int]  = None,
    hnsw_m:          Optional[int]  = None,
)
```

**Purpose:** Initialises the FAISS database, determines embedding dimensionality (by probing the embedder if not specified), and creates the FAISS index.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embedder` | `Optional[Any]` | `None` | Embedding model (e.g. `SentenceTransformer`). Must expose `.encode(List[str]) -> np.ndarray`. |
| `embedding_dim` | `Optional[int]` | `None` | Explicit embedding dimension. Required when `embedder` is `None`. Auto-detected from embedder when not provided. |
| `index_type` | `str` | `"flat"` | FAISS index type. One of `"flat"` (exact, slower), `"ivf"` (faster at scale, requires training), `"hnsw"` (approximate, graph-based). |
| `distance_metric` | `str` | `"cosine"` | Similarity metric. One of `"cosine"` (L2-normalised inner product), `"l2"` (Euclidean distance), `"ip"` (raw inner product). |
| `ivf_clusters` | `Optional[int]` | `100` | Number of IVF cluster cells (applies only to `index_type="ivf"`). |
| `hnsw_m` | `Optional[int]` | `32` | Number of HNSW connections per layer (applies only to `index_type="hnsw"`). Higher = better recall, more RAM. |

**Raises:**
- `ImportError` if `faiss` is not installed.
- `ValueError` for unsupported `index_type` or `distance_metric`, or if dimension cannot be determined.

**Index type selection guide:**

| Index | Best for | Recall | Speed |
|---|---|---|---|
| `"flat"` | < 100 K vectors, exact results required | 100% | Slow at scale |
| `"ivf"` | 100 K – 10 M vectors | Very high | Fast (requires training) |
| `"hnsw"` | Large datasets, fastest search | High (approximate) | Fastest |

---

## Ingestion _(FAISS)_

### `add_chunk` _(FAISS)_

```python
db.add_chunk(chunk: DocumentChunk) -> None
```

**Purpose:** Convenience method for adding a **single** `DocumentChunk`. Computes the embedding automatically if `chunk.embedding` is `None`, then delegates to `add()`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `chunk` | `DocumentChunk` | A single chunk with `chunk_id`, `doc_id`, and `text`. Embedding is computed on-the-fly if absent. |

**Returns:** `None`

**Raises:** `ValueError` if `chunk` is not a `DocumentChunk`, or if no embedder is available and no pre-computed embedding is set.

**Example:**

```python
chunk = DocumentChunk(chunk_id="c1", doc_id="doc_1", text="Hello world")
db.add_chunk(chunk)
```

---

### `add` _(FAISS)_

```python
db.add(
    chunks:     List[DocumentChunk],
    embeddings: Optional[np.ndarray] = None,
) -> None
```

**Purpose:** Adds a batch of `DocumentChunk` objects to the FAISS index. Validates dimension consistency, normalises vectors for cosine similarity, triggers IVF training when enough data is available, and updates both the FAISS index and all O(1) lookup dictionaries.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunks` | `List[DocumentChunk]` | — | List of chunks to add. Empty lists are silently ignored. |
| `embeddings` | `Optional[np.ndarray]` | `None` | Pre-computed embeddings of shape `(N, dim)`, `dtype=float32`. Auto-computed when `None`. |

**Returns:** `None`

**Raises:**
- `ValueError` if all items are not `DocumentChunk`, if lengths mismatch, or if embedding dimensions don't match.
- `RuntimeError` if the FAISS add operation fails.

**Example:**

```python
chunks = [
    DocumentChunk(chunk_id="c1", doc_id="d1", text="Text one"),
    DocumentChunk(chunk_id="c2", doc_id="d1", text="Text two"),
]
db.add(chunks)

# With pre-computed embeddings:
import numpy as np
embeddings = np.random.rand(2, 384).astype(np.float32)
db.add(chunks, embeddings=embeddings)
```

---

## Search _(FAISS)_

### `search` _(FAISS)_

```python
db.search(
    query:          Union[str, np.ndarray],
    top_k:          int                           = 5,
    score_threshold: Optional[float]              = None,
    doc_id_filter:  Optional[Union[str, List[str]]] = None,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Searches the FAISS index for the most similar chunks. Accepts either a text string or a pre-computed embedding. Applies optional score threshold and document-level filtering.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `Union[str, np.ndarray]` | — | Text query (encoded on-the-fly via embedder) or pre-computed 1-D `float32` embedding. |
| `top_k` | `int` | `5` | Maximum number of results. Must be > 0. |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score. Results below this are excluded. `None` disables filtering. |
| `doc_id_filter` | `Optional[Union[str, List[str]]]` | `None` | Restrict results to a single `doc_id` (string) or a list of allowed `doc_id`s. |

**Returns:** `List[Tuple[DocumentChunk, float]]` — `(chunk, score)` pairs sorted by score descending. Higher score = more similar. Empty list when the database is empty.

**Raises:**
- `ValueError` if `top_k ≤ 0` or query dimension mismatches the index.
- `RuntimeError` if the FAISS search operation fails.

**Score semantics by metric:**

| Metric | Score range | Interpretation |
|---|---|---|
| `cosine` | `[-1.0, 1.0]` | 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite |
| `l2` | `≥ 0` | 0 = identical; smaller = more similar |
| `ip` | any | Higher = more similar |

**Example:**

```python
results = db.search("machine learning algorithms", top_k=5, score_threshold=0.5)
for chunk, score in results:
    print(f"[{score:.3f}] [{chunk.doc_id}] {chunk.text[:80]}")

# Filter to a specific document
results = db.search("AI", top_k=5, doc_id_filter="doc_001")

# Filter to multiple documents
results = db.search("AI", top_k=5, doc_id_filter=["doc_001", "doc_002"])

# Use a pre-computed embedding
import numpy as np
query_vec = np.random.rand(384).astype(np.float32)
results = db.search(query_vec, top_k=3)
```

---

### `search_by_doc_id`

```python
db.search_by_doc_id(
    doc_id:           str,
    top_k:            int  = 5,
    exclude_same_doc: bool = True,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Finds chunks most similar to the content of a specific document. Internally uses each chunk of `doc_id` as a query and aggregates, deduplicates, and sorts the results. Useful for finding related documents given a document you already have in the database.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `doc_id` | `str` | — | Document identifier to use as the search source. All its chunks are used as query vectors. |
| `top_k` | `int` | `5` | Maximum results to return in total. |
| `exclude_same_doc` | `bool` | `True` | When `True`, results from the same document as the query are excluded, returning only results from other documents. |

**Returns:** `List[Tuple[DocumentChunk, float]]` — deduplicated `(chunk, score)` pairs sorted by score descending, capped at `top_k`. Returns `[]` if `doc_id` is not found.

**Example:**

```python
# Find documents related to "doc_001"
related = db.search_by_doc_id("doc_001", top_k=5, exclude_same_doc=True)
for chunk, score in related:
    print(f"[{score:.3f}] [{chunk.doc_id}] {chunk.text[:60]}")
```

---

## Deletion _(FAISS)_

### `remove_by_chunk_id`

```python
db.remove_by_chunk_id(chunk_id: str) -> bool
```

**Purpose:** Removes a single chunk by its `chunk_id`. Because FAISS does not natively support in-place deletion, this triggers a full index rebuild after removal.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `chunk_id` | `str` | The unique chunk identifier to remove. |

**Returns:** `bool` — `True` if found and removed; `False` if not found.

> ⚠️ **Performance note:** Index rebuild is O(N). For bulk deletions, prefer `remove_by_doc_id` (one rebuild for all chunks of a document).

**Example:**

```python
removed = db.remove_by_chunk_id("chunk_42")
print("Removed:", removed)
```

---

### `remove_by_doc_id` _(FAISS)_

```python
db.remove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Removes all chunks belonging to `doc_id`. Performs a single index rebuild after removing all matching chunks, making this more efficient than calling `remove_by_chunk_id` in a loop.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | `str` | Document identifier. All chunks with this `doc_id` are removed. |

**Returns:** `int` — number of chunks removed. `0` if document not found.

**Example:**

```python
count = db.remove_by_doc_id("doc_001")
print(f"Removed {count} chunks from doc_001")
```

---

### `clear` _(FAISS)_

```python
db.clear() -> int
```

**Purpose:** Removes all chunks from the database, resets all internal data structures, and recreates an empty FAISS index of the same type.

**Returns:** `int` — number of chunks that existed before clearing.

> ⚠️ This operation cannot be undone.

**Example:**

```python
count = db.clear()
print(f"Cleared {count} chunks. Database is now empty: {len(db) == 0}")
```

---

## Retrieval & Lookup _(FAISS)_

### `get_chunk_by_id` _(FAISS)_

```python
db.get_chunk_by_id(chunk_id: str) -> Optional[DocumentChunk]
```

**Purpose:** O(1) lookup of a `DocumentChunk` by its `chunk_id` from the in-memory chunk list.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Unique chunk identifier. |

**Returns:** `DocumentChunk` if found; `None` otherwise.

---

### `get_chunks_by_doc_id` _(FAISS)_

```python
db.get_chunks_by_doc_id(doc_id: str) -> List[DocumentChunk]
```

**Purpose:** Returns all chunks belonging to `doc_id` from the in-memory lookup table.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | `str` | Document identifier. |

**Returns:** `List[DocumentChunk]` — all chunks for that document. Empty list if not found.

---

### `list_document_ids`

```python
db.list_document_ids() -> List[str]
```

**Purpose:** Returns a list of all unique document IDs currently stored in the database. Useful for iterating over all documents for export, backup, or auditing.

**Parameters:** None.

**Returns:** `List[str]` — all distinct `doc_id` values.

**Example:**

```python
doc_ids = db.list_document_ids()
print(f"Database contains {len(doc_ids)} documents:")
for doc_id in doc_ids:
    chunks = db.get_chunks_by_doc_id(doc_id)
    print(f"  {doc_id}: {len(chunks)} chunks")
```

---

## Persistence _(FAISS)_

### `save` _(FAISS)_

```python
db.save(path: Union[str, Path]) -> None
```

**Purpose:** Persists the full database state to three files in `path`. Creates the directory automatically if it does not exist.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[str, Path]` | Directory path to write files into (created if missing). |

**Returns:** `None`

**Raises:** `RuntimeError` if any file write fails.

**Saved file layout:**

```
<path>/
├── index.faiss    ← FAISS binary index
├── chunks.pkl     ← List of serialised DocumentChunk dicts
└── metadata.pkl   ← Config: embedding_dim, index_type, distance_metric, etc.
```

**Example:**

```python
db.save("./saved_db/v1")
```

---

### `load` _(FAISS)_

```python
@classmethod
FAISSVectorDatabase.load(
    path:    Union[str, Path],
    embedder: Optional[Any] = None,
) -> FAISSVectorDatabase
```

**Purpose:** Restores a fully operational `FAISSVectorDatabase` from a directory previously written by `save()`. Validates that all three required files are present before loading.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `Union[str, Path]` | — | Directory path containing `index.faiss`, `chunks.pkl`, and `metadata.pkl`. |
| `embedder` | `Optional[Any]` | `None` | Optional embedder for computing new embeddings on queries after loading. |

**Returns:** `FAISSVectorDatabase` — fully loaded instance.

**Raises:**
- `FileNotFoundError` if the path or any required file is missing.
- `RuntimeError` if loading fails.

**Example:**

```python
db = FAISSVectorDatabase.load("./saved_db/v1", embedder=my_embedder)
results = db.search("What is machine learning?", top_k=5)
```

---

## Statistics _(FAISS)_

### `get_stats` _(FAISS)_

```python
db.get_stats() -> Dict[str, Any]
```

**Purpose:** Returns comprehensive statistics about the FAISS index and its contents.

**Parameters:** None.

**Returns:** `Dict[str, Any]`:

| Key | Type | Description |
|---|---|---|
| `total_chunks` | `int` | Total vectors stored. |
| `unique_docs` | `int` | Number of distinct `doc_id` values. |
| `embedding_dim` | `int` | Vector dimensionality. |
| `index_type` | `str` | `"flat"`, `"ivf"`, or `"hnsw"`. |
| `distance_metric` | `str` | Active metric. |
| `ivf_clusters` | `int \| None` | IVF cluster count (only for `"ivf"`; `None` otherwise). |
| `hnsw_m` | `int \| None` | HNSW connections (only for `"hnsw"`; `None` otherwise). |
| `is_trained` | `bool` | Whether the IVF index has been trained. Always `True` for Flat/HNSW. |
| `has_embedder` | `bool` | Whether an embedder is configured. |

---

## Async API _(FAISS)_

### `asearch` _(FAISS)_

```python
async db.asearch(
    query:           Union[str, np.ndarray],
    top_k:           int   = 10,
    score_threshold: float = 0.0,
    **kwargs,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Async wrapper for `search()`. Runs in a thread pool via `asyncio.to_thread`. All parameters and return values are identical to `search()`.

---

### `aadd` _(FAISS)_

```python
async db.aadd(chunks: List[DocumentChunk], **kwargs) -> None
```

**Purpose:** Async wrapper for `add()`. Runs in a thread pool. All parameters and return values are identical to `add()`.

---

### `aremove_by_doc_id` _(FAISS)_

```python
async db.aremove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Async wrapper for `remove_by_doc_id()`. Runs in a thread pool.

---

### `asave` _(FAISS)_

```python
async db.asave(path: Union[str, Path]) -> None
```

**Purpose:** Async wrapper for `save()`. Runs in a thread pool, keeping the event loop unblocked during file I/O.

**Parameters / Returns:** Identical to `save()`.

---

---

## Class: `ChromaVectorDatabase`

```python
from fennec_community.vector_database import ChromaVectorDatabase
```

A production-ready wrapper for ChromaDB offering automatic batching, multi-tenancy via `tenant_id`, persistent or in-memory storage, and the same unified API as FAISS.

**Install:** `pip install chromadb`

---

### `__init__` _(Chroma)_

```python
ChromaVectorDatabase(
    embedder:          Optional[Any] = None,
    collection_name:   str           = "default_collection",
    persist_directory: Optional[str] = None,
    distance_metric:   str           = "cosine",
    tenant_id:         Optional[str] = None,
    batch_size:        int           = 500,
    strict_mode:       bool          = True,
)
```

**Purpose:** Initialises a ChromaDB client (persistent or in-memory) and creates or retrieves a named collection.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embedder` | `Optional[Any]` | `None` | Embedding model with `.encode()` method. Required for text queries when no pre-computed embeddings are provided. |
| `collection_name` | `str` | `"default_collection"` | Name of the ChromaDB collection. Existing collections are retrieved; new collections are created. |
| `persist_directory` | `Optional[str]` | `None` | Path for on-disk persistence via `chromadb.PersistentClient`. Pass `None` for in-memory (data lost on restart). |
| `distance_metric` | `str` | `"cosine"` | Distance metric: `"cosine"`, `"l2"`, or `"ip"`. |
| `tenant_id` | `Optional[str]` | `None` | Optional tenant identifier. When set, all added documents get a `tenant_id` metadata key, and all searches are automatically scoped to this tenant. |
| `batch_size` | `int` | `500` | Number of vectors per upsert batch. Must be ≥ 1. |
| `strict_mode` | `bool` | `True` | When `True`, validates that all IDs are non-empty strings before any upsert. |

**Raises:**
- `ImportError` if `chromadb` is not installed.
- `ValueError` if `distance_metric` is unsupported.
- `RuntimeError` if the ChromaDB client or collection cannot be created.

---

## Ingestion _(Chroma)_

### `add` _(Chroma)_

```python
db.add(
    chunks:     List[DocumentChunk],
    embeddings: Optional[np.ndarray] = None,
) -> None
```

**Purpose:** Adds or updates chunks in ChromaDB using upsert semantics. Accepts `DocumentChunk` objects (unified interface) or raw `ids`/`documents`/`metadatas` lists (legacy interface). Processes data in batches of `batch_size`. Automatically applies `sanitize_metadata` and tenant isolation.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunks` | `List[DocumentChunk]` | — | List of `DocumentChunk` objects. Detected automatically by the presence of `chunk_id` attribute. |
| `embeddings` | `Optional[np.ndarray]` | `None` | Pre-computed embeddings of shape `(N, dim)`. Uses `chunk.embedding` if available. Generates via embedder when neither is present. |

**Returns:** `None`

**Raises:**
- `ValueError` if input lengths mismatch, IDs are invalid (in strict mode), or no embedder is available.
- `RuntimeError` if any batch upsert fails.

**Example:**

```python
chunks = [
    DocumentChunk(chunk_id="c1", doc_id="d1", text="Hello world", metadata={"lang": "en"}),
    DocumentChunk(chunk_id="c2", doc_id="d1", text="Bonjour monde", metadata={"lang": "fr"}),
]
db.add(chunks)
```

---

## Search _(Chroma)_

### `search` _(Chroma)_

```python
db.search(
    query:           Union[str, np.ndarray],
    top_k:           int                   = 5,
    score_threshold: Optional[float]       = None,
    filters:         Optional[Dict]        = None,
    **kwargs,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Semantic similarity search using the ChromaDB collection. Distances returned by ChromaDB are converted to similarity scores. Tenant isolation is enforced automatically when `tenant_id` is set.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `Union[str, np.ndarray]` | — | Text (encoded on-the-fly) or pre-computed embedding. |
| `top_k` | `int` | `5` | Maximum number of results. |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score. Results below this are excluded. |
| `filters` | `Optional[Dict]` | `None` | ChromaDB `where` metadata filter dict (e.g., `{"category": "news"}`). Combined with tenant filter automatically when `tenant_id` is set. |

**Returns:** `List[Tuple[DocumentChunk, float]]` — `(chunk, score)` pairs, score descending. Each chunk is reconstructed from ChromaDB metadata.

**Score conversion:**

| Metric | Conversion |
|---|---|
| `cosine` | `score = 1.0 - distance` (range: `[0, 1]`) |
| `l2` | `score = -distance` (more negative = less similar) |
| `ip` | `score = distance` (higher = better) |

**Example:**

```python
results = db.search(
    "machine learning",
    top_k=5,
    score_threshold=0.6,
    filters={"lang": "en"},
)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text[:60]}")
```

---

## Deletion _(Chroma)_

### `delete_by_ids` _(Chroma)_

```python
db.delete_by_ids(ids: List[str]) -> int
```

**Purpose:** Deletes documents from ChromaDB by their chunk IDs.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ids` | `List[str]` | List of `chunk_id` strings to delete. Empty lists are silently ignored. |

**Returns:** `int` — number of IDs deleted (= `len(ids)` on success).

**Raises:** `RuntimeError` if the delete operation fails.

---

### `delete_by_filter` _(Chroma)_

```python
db.delete_by_filter(filters: Dict) -> int
```

**Purpose:** Deletes all documents matching a metadata filter. First queries ChromaDB to find matching IDs, then deletes them. Tenant isolation is automatically applied when `tenant_id` is set.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `filters` | `Dict` | ChromaDB `where` clause (e.g., `{"status": "archived"}`). Empty dicts are silently ignored. |

**Returns:** `int` — number of documents deleted.

**Raises:** `RuntimeError` if the operation fails.

**Example:**

```python
count = db.delete_by_filter({"category": "draft"})
print(f"Deleted {count} draft documents")
```

---

### `remove_by_doc_id` _(Chroma)_

```python
db.remove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Removes all chunks belonging to `doc_id`. Delegates to `delete_by_filter({"doc_id": doc_id})`. This is the unified interface method used by `RAGSystem` and `GraphRAG`.

**Parameters / Returns:** Same as `delete_by_filter`.

---

### `clear_collection`

```python
db.clear_collection() -> int
```

**Purpose:** Deletes **all** documents from the ChromaDB collection. Fetches all IDs first, then deletes them in one batch.

**Returns:** `int` — number of documents deleted.

**Raises:** `RuntimeError` if the operation fails.

> ⚠️ Irreversible for in-memory clients. For persistent clients, the collection file on disk is also cleared.

---

## Retrieval & Lookup _(Chroma)_

### `get_by_ids`

```python
db.get_by_ids(ids: List[str]) -> List[Tuple[str, Dict]]
```

**Purpose:** Retrieves documents by their `chunk_id` strings directly from ChromaDB. Returns `(text, metadata)` tuples. Note: returns raw text and metadata, not `DocumentChunk` objects.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ids` | `List[str]` | List of chunk IDs to retrieve. Empty lists return `[]`. |

**Returns:** `List[Tuple[str, Dict]]` — list of `(document_text, metadata_dict)` tuples.

**Raises:** `RuntimeError` if the ChromaDB get operation fails.

**Example:**

```python
docs = db.get_by_ids(["c1", "c2"])
for text, meta in docs:
    print(f"Text: {text[:60]}  Meta: {meta}")
```

---

## Statistics _(Chroma)_

### `stats` _(Chroma)_

```python
db.stats() -> Dict[str, Any]
```

**Purpose:** Returns Chroma-specific statistics and configuration. Includes the collection name, tenant, batch size, and strict mode settings.

**Returns:** `Dict[str, Any]`:

| Key | Type | Description |
|---|---|---|
| `collection_name` | `str` | Name of the ChromaDB collection. |
| `total_vectors` | `int` | Current count from `collection.count()`. |
| `distance_metric` | `str` | Active metric. |
| `embedding_dimension` | `int \| None` | Auto-detected dimension (`None` until first `add`). |
| `tenant_id` | `str \| None` | Configured tenant, or `None`. |
| `batch_size` | `int` | Configured batch size. |
| `strict_mode` | `bool` | Whether strict validation is enabled. |

---

### `get_stats` _(Chroma)_

```python
db.get_stats() -> Dict[str, Any]
```

**Purpose:** Unified statistics method. Delegates to `stats()` and ensures the mandatory base keys (`total_vectors`, `has_embedder`) are present. Use this for cross-backend monitoring code.

**Returns:** `Dict[str, Any]` — all keys from `stats()` plus `has_embedder`.

---

## Context Manager _(Chroma)_

### `batch_operation` _(Chroma)_

```python
with db.batch_operation():
    db.add(chunks_1)
    db.add(chunks_2)
    db.delete_by_filter({"status": "stale"})
```

**Purpose:** Context manager for grouping multiple operations with automatic error logging. On success, logs completion. On exception, logs the error and re-raises. Does not provide transactional atomicity at the ChromaDB level.

**Returns:** `self` (the `ChromaVectorDatabase` instance).

---

## Async API _(Chroma)_

### `asearch` _(Chroma)_

```python
async db.asearch(query, top_k=10, **kwargs) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Async wrapper for `search()`. Runs in a thread pool.

### `aadd` _(Chroma)_

```python
async db.aadd(chunks, **kwargs) -> None
```

**Purpose:** Async wrapper for `add()`. Runs in a thread pool.

---

---

## Class: `PineconeVectorDatabase`

```python
from fennec_community.vector_database import PineconeVectorDatabase
```

A fully managed cloud vector store wrapper for Pinecone. Supports serverless and pod-based indexes, namespace-based multi-tenancy, rich metadata filtering, and an optional local chunk cache for zero-latency metadata reads.

**Install:** `pip install pinecone-client`  
**API key:** Pass as `api_key=` or set `PINECONE_API_KEY` environment variable.

---

### `__init__` _(Pinecone)_

```python
PineconeVectorDatabase(
    embedder:        Optional[Any] = None,
    index_name:      str           = "default-index",
    embedding_dim:   Optional[int] = None,
    api_key:         Optional[str] = None,
    environment:     str           = "us-east-1",
    distance_metric: str           = "cosine",
    cloud:           str           = "aws",
    pod_type:        Optional[str] = None,
    namespace:       Optional[str] = None,
    cache_chunks:    bool          = True,
)
```

**Purpose:** Connects to Pinecone, creates the index if it does not exist (waiting up to 5 minutes for readiness), verifies configuration consistency on existing indexes, and sets up an optional local chunk cache.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embedder` | `Optional[Any]` | `None` | Embedding model. Required for text queries. |
| `index_name` | `str` | `"default-index"` | Pinecone index name. Created automatically if not found. |
| `embedding_dim` | `Optional[int]` | `None` | Embedding dimension. Auto-detected from embedder when `None`. **Required** when `embedder` is `None`. |
| `api_key` | `Optional[str]` | `None` | Pinecone API key. Falls back to `PINECONE_API_KEY` env var. |
| `environment` | `str` | `"us-east-1"` | Pinecone region (e.g., `"us-east-1"`, `"eu-west-1"`). |
| `distance_metric` | `str` | `"cosine"` | Metric: `"cosine"`, `"euclidean"` / `"l2"`, `"dotproduct"` / `"dot"` / `"ip"`. |
| `cloud` | `str` | `"aws"` | Cloud provider for serverless indexes: `"aws"`, `"gcp"`, `"azure"`. |
| `pod_type` | `Optional[str]` | `None` | Pod type for pod-based indexes (e.g., `"p1.x1"`). Pass `None` for serverless. |
| `namespace` | `Optional[str]` | `None` | Default namespace for all operations. Empty string = default namespace. |
| `cache_chunks` | `bool` | `True` | When `True`, chunks are stored in local memory for O(1) metadata reads via `get_chunk_by_id`. |

**Raises:**
- `ImportError` if `pinecone-client` is not installed.
- `ValueError` if metric is unsupported, dimension invalid, or API key not found.
- `ConnectionError` if Pinecone cannot be initialised.

**Metric mapping:**

| Input value | Pinecone metric |
|---|---|
| `"cosine"` | `"cosine"` |
| `"euclidean"` / `"l2"` | `"euclidean"` |
| `"dotproduct"` / `"dot"` / `"ip"` | `"dotproduct"` |

---

## Ingestion _(Pinecone)_

### `add_chunk` _(Pinecone)_

```python
db.add_chunk(chunk: DocumentChunk) -> None
```

**Purpose:** Adds a single `DocumentChunk` to the Pinecone index. Computes the embedding on-the-fly if `chunk.embedding` is `None`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `chunk` | `DocumentChunk` | Single chunk with `chunk_id`, `doc_id`, and `text`. |

**Returns:** `None`

**Raises:** `ValueError` if `chunk` is not a `DocumentChunk` or if no embedder is available and no embedding is set.

---

### `add` _(Pinecone)_

```python
db.add(
    chunks:     List[DocumentChunk],
    embeddings: Optional[np.ndarray] = None,
    namespace:  Optional[str]        = None,
) -> None
```

**Purpose:** Upserts a batch of chunks to Pinecone in batches of 100 (Pinecone API hard limit). Stores `chunk_id`, `doc_id`, `text`, and sanitised metadata in Pinecone vector metadata. Optionally caches chunks locally.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunks` | `List[DocumentChunk]` | — | List of chunks to upsert. Empty lists are silently ignored. |
| `embeddings` | `Optional[np.ndarray]` | `None` | Pre-computed embeddings of shape `(N, dim)`, `float32`. |
| `namespace` | `Optional[str]` | `None` | Override the instance namespace for this batch only. Uses instance `namespace` when `None`. |

**Returns:** `None`

**Raises:**
- `ValueError` if chunks are invalid, lengths mismatch, or dimension mismatches.
- `RuntimeError` if any Pinecone upsert batch fails.

**Example:**

```python
chunks = [
    DocumentChunk("c1", "d1", "Text one", metadata={"category": "tech"}),
    DocumentChunk("c2", "d1", "Text two", metadata={"category": "science"}),
]

# Add to default namespace
db.add(chunks)

# Add to a specific namespace
db.add(chunks, namespace="team_a")
```

---

## Search _(Pinecone)_

### `search` _(Pinecone)_

```python
db.search(
    query:            Union[str, np.ndarray],
    top_k:            int                    = 5,
    score_threshold:  Optional[float]        = None,
    filter_dict:      Optional[Dict[str, Any]] = None,
    namespace:        Optional[str]          = None,
    include_metadata: bool                   = True,
    include_values:   bool                   = False,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Searches the Pinecone index with full metadata filtering and optional namespace scoping. Reconstructs `DocumentChunk` objects from Pinecone metadata.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `Union[str, np.ndarray]` | — | Text (encoded on-the-fly) or pre-computed 1-D `float32` embedding. |
| `top_k` | `int` | `5` | Maximum number of results. |
| `score_threshold` | `Optional[float]` | `None` | Minimum score filter. |
| `filter_dict` | `Optional[Dict]` | `None` | Pinecone metadata filter expression (e.g., `{"category": {"$in": ["tech", "science"]}}`, `{"year": {"$gte": 2020}}`). |
| `namespace` | `Optional[str]` | `None` | Namespace to search in. Uses instance namespace when `None`. |
| `include_metadata` | `bool` | `True` | Whether to include metadata in results. Set `False` for faster queries when text is not needed. |
| `include_values` | `bool` | `False` | When `True`, includes the embedding vector in the returned `chunk.embedding`. |

**Returns:** `List[Tuple[DocumentChunk, float]]` — `(chunk, score)` pairs, score descending.

**Example:**

```python
# Simple search
results = db.search("machine learning", top_k=5)

# Filtered search with Pinecone expression language
results = db.search(
    "AI trends",
    top_k=10,
    score_threshold=0.7,
    filter_dict={"year": {"$gte": 2022}, "category": "tech"},
    namespace="production",
)

for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text[:80]}")
```

---

### `search_by_id` _(Pinecone)_

```python
db.search_by_id(
    chunk_id:     str,
    top_k:        int            = 5,
    exclude_self: bool           = True,
    namespace:    Optional[str]  = None,
) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Fetches a chunk's embedding by ID from Pinecone, then uses it as the query vector for a similarity search. Useful for finding chunks semantically related to a specific stored chunk.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_id` | `str` | — | ID of the chunk to use as the query. |
| `top_k` | `int` | `5` | Maximum results. |
| `exclude_self` | `bool` | `True` | When `True`, the query chunk itself is excluded from the results. |
| `namespace` | `Optional[str]` | `None` | Namespace to search in. |

**Returns:** `List[Tuple[DocumentChunk, float]]`. Returns `[]` if `chunk_id` is not found.

**Example:**

```python
similar = db.search_by_id("chunk_42", top_k=5, exclude_self=True)
for chunk, score in similar:
    print(f"[{score:.3f}] {chunk.chunk_id}: {chunk.text[:50]}")
```

---

## Deletion _(Pinecone)_

### `delete_by_ids` _(Pinecone)_

```python
db.delete_by_ids(
    chunk_ids: List[str],
    namespace: Optional[str] = None,
) -> int
```

**Purpose:** Deletes chunks by their IDs from Pinecone in batches of 1000. Also updates the local cache when `cache_chunks=True`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_ids` | `List[str]` | — | List of `chunk_id` strings to delete. |
| `namespace` | `Optional[str]` | `None` | Target namespace. |

**Returns:** `int` — `len(chunk_ids)` (Pinecone ID-based deletes are always considered successful).

---

### `delete_by_doc_id` _(Pinecone)_

```python
db.delete_by_doc_id(
    doc_id:    str,
    namespace: Optional[str] = None,
) -> int
```

**Purpose:** Deletes all chunks belonging to `doc_id`. Prefers the local cache for fast ID lookup; falls back to a Pinecone metadata filter delete when the cache is unavailable.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `doc_id` | `str` | — | Document identifier. |
| `namespace` | `Optional[str]` | `None` | Target namespace. |

**Returns:** `int` — number of chunks deleted (0 when using filter-based delete, since Pinecone does not return the count).

---

### `delete_by_filter` _(Pinecone)_

```python
db.delete_by_filter(
    filter_dict: Dict[str, Any],
    namespace:   Optional[str] = None,
) -> None
```

**Purpose:** Deletes all vectors matching a Pinecone metadata filter. Supports the full Pinecone expression language.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_dict` | `Dict[str, Any]` | — | Pinecone filter expression (e.g., `{"status": "archived"}`). Empty dicts are ignored. |
| `namespace` | `Optional[str]` | `None` | Target namespace. |

**Returns:** `None` (Pinecone does not return a deletion count for filter-based deletes).

---

### `delete_all` _(Pinecone)_

```python
db.delete_all(namespace: Optional[str] = None) -> None
```

**Purpose:** Deletes **all** vectors from the specified namespace (or the instance's default namespace). Also clears the local cache when `cache_chunks=True`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `namespace` | `Optional[str]` | `None` | Target namespace. Uses instance namespace when `None`. |

**Returns:** `None`

> ⚠️ This operation is irreversible. It removes all data from the namespace.

---

### `remove_by_doc_id` _(Pinecone)_

```python
db.remove_by_doc_id(doc_id: str) -> int
```

**Purpose:** Unified interface alias. Delegates to `delete_by_doc_id(doc_id)`.

---

### `clear` _(Pinecone)_

```python
db.clear() -> int
```

**Purpose:** Unified interface alias. Calls `delete_all()` for the instance's default namespace.

**Returns:** `int` — always `0` (Pinecone does not return count for full namespace clears).

---

## Fetch Operations _(Pinecone)_

### `fetch_by_ids` _(Pinecone)_

```python
db.fetch_by_ids(
    chunk_ids: List[str],
    namespace: Optional[str] = None,
) -> List[DocumentChunk]
```

**Purpose:** Fetches complete vector data (including embedding values) for specific chunk IDs directly from Pinecone. Unlike `search`, this is an exact lookup by ID — no similarity computation.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_ids` | `List[str]` | — | List of chunk IDs to fetch. Empty lists return `[]`. |
| `namespace` | `Optional[str]` | `None` | Target namespace. |

**Returns:** `List[DocumentChunk]` — each chunk has its `chunk_id`, `doc_id`, `text`, `metadata`, and `embedding` fully populated from the Pinecone response.

**Raises:** `RuntimeError` if the Pinecone fetch fails.

**Example:**

```python
chunks = db.fetch_by_ids(["c1", "c2", "c3"])
for chunk in chunks:
    print(f"{chunk.chunk_id}: embedding dim={len(chunk.embedding)}")
```

---

## Retrieval & Lookup _(Pinecone)_

### `get_chunk_by_id` _(Pinecone)_

```python
db.get_chunk_by_id(chunk_id: str) -> Optional[DocumentChunk]
```

**Purpose:** O(1) lookup of a `DocumentChunk` from the **local cache** (only when `cache_chunks=True`). Does not make a network call to Pinecone.

**Returns:** `DocumentChunk` if found in cache; `None` if not cached or `cache_chunks=False`.

---

### `get_chunks_by_doc_id` _(Pinecone)_

```python
db.get_chunks_by_doc_id(doc_id: str) -> List[DocumentChunk]
```

**Purpose:** Returns all locally cached chunks for `doc_id`. Only works when `cache_chunks=True`.

**Returns:** `List[DocumentChunk]` — empty list when cache is disabled or document not found.

---

## Statistics & Info _(Pinecone)_

### `get_stats` _(Pinecone)_

```python
db.get_stats(namespace: Optional[str] = None) -> Dict[str, Any]
```

**Purpose:** Fetches live statistics from Pinecone's `describe_index_stats()` API and combines them with local cache information.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `namespace` | `Optional[str]` | `None` | Namespace to get stats for. Uses instance namespace when `None`. |

**Returns:** `Dict[str, Any]`:

| Key | Type | Description |
|---|---|---|
| `index_name` | `str` | Name of the Pinecone index. |
| `total_vector_count` | `int` | Total vectors across all namespaces. |
| `embedding_dim` | `int` | Vector dimensionality. |
| `distance_metric` | `str` | Active metric. |
| `namespace` | `str` | Queried namespace. |
| `namespace_vector_count` | `int` | Vectors in the queried namespace. |
| `namespaces` | `List[str]` | All existing namespace names. |
| `has_embedder` | `bool` | Whether an embedder is configured. |
| `cache_enabled` | `bool` | Whether local cache is active. |
| `cached_chunks` | `int` | Number of non-null entries in the local cache. |
| `unique_docs_cached` | `int` | *(only when `cache_chunks=True`)* Distinct `doc_id` values in the local cache. |

Returns `{"error": "<message>"}` if the Pinecone API call fails.

---

### `list_namespaces` _(Pinecone)_

```python
db.list_namespaces() -> List[str]
```

**Purpose:** Returns all namespace names currently in the Pinecone index.

**Returns:** `List[str]` — namespace names. Returns `[]` on API error.

**Example:**

```python
namespaces = db.list_namespaces()
print("Active namespaces:", namespaces)
```

---

## Context Manager _(Pinecone)_

### `batch_operation` _(Pinecone)_

```python
with db.batch_operation(namespace="team_b"):
    db.add(chunks_1)
    db.add(chunks_2)
```

**Purpose:** Context manager that temporarily overrides the instance namespace for the duration of the block. On exit (success or error), the original namespace is restored.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `namespace` | `Optional[str]` | `None` | Namespace to use inside the block. Instance namespace unchanged when `None`. |

**Returns:** `self` (the `PineconeVectorDatabase` instance).

---

## Async API _(Pinecone)_

### `asearch` _(Pinecone)_

```python
async db.asearch(query, top_k=10, **kwargs) -> List[Tuple[DocumentChunk, float]]
```

**Purpose:** Async wrapper for `search()`. Runs in a thread pool.

### `aadd` _(Pinecone)_

```python
async db.aadd(chunks, **kwargs) -> None
```

**Purpose:** Async wrapper for `add()`. Runs in a thread pool.

---

---

## Backend Comparison Matrix

| Feature | FAISSVectorDatabase | ChromaVectorDatabase | PineconeVectorDatabase |
|---|---|---|---|
| **Storage** | Local in-memory | Local/Persistent | Cloud (always persistent) |
| **Index types** | Flat / IVF / HNSW | HNSW (auto) | Managed by Pinecone |
| **GPU support** | ✅ (`faiss-gpu`) | ❌ | N/A |
| **Distance metrics** | cosine, l2, ip | cosine, l2, ip | cosine, euclidean, dotproduct |
| **Score threshold** | ✅ | ✅ | ✅ |
| **Metadata filter** | doc_id only | Rich `where` clauses | Full expression language |
| **Multi-tenancy** | ❌ | tenant_id isolation | Namespaces |
| **Local persistence** | ✅ `save()`/`load()` | ✅ `PersistentClient` | N/A (always cloud) |
| **Chunk cache** | ✅ (always on) | ❌ | ✅ (optional) |
| **search_by_doc_id** | ✅ | ❌ | ❌ |
| **fetch_by_ids** | ❌ | via `get_by_ids` | ✅ |
| **Async** | ✅ | ✅ | ✅ |
| **Install** | `faiss-cpu` | `chromadb` | `pinecone-client` |
| **API key** | ❌ | ❌ | ✅ required |

---

## Unified Interface Contract

Any code that uses `VectorDatabaseBase` as the type annotation is guaranteed to work with all three backends without modification:

```python
def ingest_documents(db: VectorDatabaseBase, chunks: List[DocumentChunk]) -> None:
    db.add(chunks)
    print(f"Indexed {len(db)} vectors")

def retrieve(db: VectorDatabaseBase, query: str) -> str:
    results = db.search(query, top_k=5)
    return "\n".join(chunk.text for chunk, _ in results)

def delete_document(db: VectorDatabaseBase, doc_id: str) -> None:
    removed = db.remove_by_doc_id(doc_id)
    print(f"Removed {removed} chunks from {doc_id}")

# Works identically for FAISS, Chroma, and Pinecone:
for backend in [faiss_db, chroma_db, pinecone_db]:
    ingest_documents(backend, chunks)
    answer = retrieve(backend, "What is machine learning?")
    delete_document(backend, "doc_001")
```

---

## Distance Metric Reference

| Metric | FAISS | Chroma | Pinecone | Best for |
|---|---|---|---|---|
| `cosine` | ✅ | ✅ | ✅ | NLP, text embeddings (normalised vectors) |
| `l2` / `euclidean` | ✅ | ✅ | ✅ | Image embeddings, absolute distance matters |
| `ip` / `dotproduct` | ✅ | ✅ | ✅ (as `dotproduct`) | Recommendation systems, un-normalised vectors |

**Score polarity:**

| Backend | Metric | Higher score = more similar? |
|---|---|---|
| FAISS | cosine / ip | ✅ Yes |
| FAISS | l2 | ❌ No (smaller = more similar) |
| Chroma | cosine | ✅ Yes (converted: `1 - distance`) |
| Chroma | l2 | ❌ No (negated distance) |
| Chroma | ip | ✅ Yes |
| Pinecone | all | ✅ Yes (Pinecone always returns similarity scores) |

---

## Metadata Sanitisation Rules

Called by every backend's `add()` method. See [`sanitize_metadata`](#module-level-function-sanitize_metadata) for full details.

| Input type | Output | Example |
|---|---|---|
| `str` / `int` / `float` / `bool` | Unchanged | `"text"` → `"text"` |
| `None` | Key dropped | `None` → *(key absent)* |
| `list` (any) | `", "` joined string | `["a", "b"]` → `"a, b"` |
| `dict` | JSON string | `{"k": "v"}` → `'{"k": "v"}'` |
| other | `str()` fallback | `{1, 2, 3}` → `"{1, 2, 3}"` |

---

## Installation Guide

```bash
# Core (required — provides VectorDatabaseBase and sanitize_metadata)
pip install numpy

# FAISS (choose one)
pip install faiss-cpu    # CPU-only (most systems)
pip install faiss-gpu    # GPU-accelerated (requires CUDA)

# ChromaDB
pip install chromadb

# Pinecone
pip install pinecone-client

# Embedding model (recommended)
pip install sentence-transformers
```

---

## Complete Examples

### Example 1 — FAISS: full lifecycle

```python
from fennec_community.vector_database import FAISSVectorDatabase
from fennec_community.embeddings import ArabicEmbedder
from fennec_community.chunks import DocumentChunk

embedder = ArabicEmbedder()
db = FAISSVectorDatabase(embedder=embedder, index_type="flat", distance_metric="cosine")

# Add documents
chunks = [
    DocumentChunk("c1", "doc_1", "Python is a high-level programming language."),
    DocumentChunk("c2", "doc_1", "Python was created by Guido van Rossum."),
    DocumentChunk("c3", "doc_2", "Machine learning is a subset of artificial intelligence."),
    DocumentChunk("c4", "doc_2", "Deep learning uses multi-layer neural networks."),
]
db.add(chunks)

print(repr(db))
# FAISSVectorDatabase(vectors=4, docs=2, dim=384, type=flat, metric=cosine)

# Search
results = db.search("Who created Python?", top_k=2, score_threshold=0.4)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text}")

# Find related documents
related = db.search_by_doc_id("doc_1", top_k=2, exclude_same_doc=True)

# Stats
print(db.get_stats())

# Persist
db.save("./my_faiss_db")
db2 = FAISSVectorDatabase.load("./my_faiss_db", embedder=embedder)
print(f"Loaded: {len(db2)} vectors")

# Remove document
removed = db.remove_by_doc_id("doc_1")
print(f"Removed {removed} chunks. Total: {len(db)}")

# Full clear
db.clear()
```

---

### Example 2 — ChromaDB: multi-tenant setup

```python
from fennec_community.vector_database import ChromaVectorDatabase
from fennec_community.chunks import DocumentChunk

db = ChromaVectorDatabase(
    embedder=embedder,
    collection_name="company_docs",
    persist_directory="./chroma_store",
    distance_metric="cosine",
    tenant_id="tenant_42",       # All data isolated to this tenant
    batch_size=100,
)

# Add with metadata
chunks = [
    DocumentChunk("c1", "d1", "Q4 revenue report", metadata={"dept": "finance"}),
    DocumentChunk("c2", "d2", "AI project proposal", metadata={"dept": "engineering"}),
]
db.add(chunks)

# Filtered search (automatically scoped to tenant_42)
results = db.search("revenue", top_k=5, filters={"dept": "finance"})

# Batch operations
with db.batch_operation():
    db.add([DocumentChunk("c3", "d3", "HR policy update", metadata={"dept": "hr"})])
    db.delete_by_filter({"dept": "archived"})

# Stats
print(db.get_stats())

# Remove all data for a document
count = db.remove_by_doc_id("d1")
print(f"Removed {count} chunks from d1")
```

---

### Example 3 — Pinecone: cloud with namespaces

```python
import os
from fennec_community.vector_database import PineconeVectorDatabase
from fennec_community.chunks import DocumentChunk

db = PineconeVectorDatabase(
    embedder=embedder,
    index_name="production-rag",
    api_key=os.environ["PINECONE_API_KEY"],
    environment="us-east-1",
    distance_metric="cosine",
    namespace="v2",
    cache_chunks=True,
)

# Add to namespace "v2"
chunks = [
    DocumentChunk("c1", "d1", "Intro to ML", metadata={"year": 2024, "lang": "en"}),
    DocumentChunk("c2", "d2", "Deep Learning guide", metadata={"year": 2023, "lang": "en"}),
]
db.add(chunks, namespace="v2")

# Rich metadata search
results = db.search(
    "neural networks",
    top_k=5,
    score_threshold=0.6,
    filter_dict={"year": {"$gte": 2023}, "lang": "en"},
)

# Temporarily switch namespace for a batch
with db.batch_operation(namespace="v3"):
    db.add([DocumentChunk("c5", "d5", "v3 document")])

# Search by chunk ID
similar = db.search_by_id("c1", top_k=3, exclude_self=True)

# Fetch raw data
fetched = db.fetch_by_ids(["c1", "c2"])

# Stats
stats = db.get_stats()
print(f"Total: {stats['total_vector_count']} vectors across {stats['namespaces']}")

# Namespaces
print(db.list_namespaces())

# Delete operations
db.delete_by_ids(["c1", "c2"])
db.delete_by_doc_id("d2")
db.delete_by_filter({"year": 2021})
```

---

### Example 4 — Async usage in FastAPI

```python
from fastapi import FastAPI
from fennec_community.vector_database import FAISSVectorDatabase

app = FastAPI()
db = FAISSVectorDatabase(embedder=embedder, index_type="flat")

@app.post("/index")
async def index_chunks(chunks: list):
    doc_chunks = [DocumentChunk(**c) for c in chunks]
    await db.aadd(doc_chunks)
    return {"indexed": len(doc_chunks), "total": len(db)}

@app.get("/search")
async def search(q: str, top_k: int = 5, threshold: float = None):
    results = await db.asearch(q, top_k=top_k, score_threshold=threshold)
    return [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
         "text": c.text, "score": float(s)}
        for c, s in results
    ]

@app.delete("/document/{doc_id}")
async def delete_doc(doc_id: str):
    count = await db.aremove_by_doc_id(doc_id)
    return {"removed": count}

@app.get("/stats")
def stats():
    return db.get_stats()
```

---

### Example 5 — Backend-agnostic code using `VectorDatabaseBase`

```python
from fenenc_community.vector_database import VectorDatabaseBase, FAISSVectorDatabase, ChromaVectorDatabase
import os

def build_db(backend: str = "faiss") -> VectorDatabaseBase:
    if backend == "faiss":
        return FAISSVectorDatabase(embedder=embedder, index_type="ivf")
    elif backend == "chroma":
        return ChromaVectorDatabase(embedder=embedder, persist_directory="./chroma")
    elif backend == "pinecone":
        from vector_database import PineconeVectorDatabase
        return PineconeVectorDatabase(embedder=embedder, index_name="my-index")
    raise ValueError(f"Unknown backend: {backend}")


def rag_pipeline(db: VectorDatabaseBase, documents: list, query: str) -> str:
    chunks = [DocumentChunk(f"c{i}", f"d{i}", text) for i, text in enumerate(documents)]
    db.add(chunks)

    results = db.search(query, top_k=3, score_threshold=0.4)
    context = "\n".join(c.text for c, _ in results)
    return context


# Switch backend without changing application code:
backend = os.getenv("VECTOR_BACKEND", "faiss")
db = build_db(backend)
print(rag_pipeline(db, ["Doc one", "Doc two", "Doc three"], "What is in doc one?"))
print(db.get_stats())
```

---

### Example 6 — `sanitize_metadata` standalone usage

```python
from fennec_community.vector_database import sanitize_metadata

raw = {
    "title":     "Enterprise RAG Guide",
    "tags":      ["rag", "llm", "production"],    # list → joined
    "version":   None,                             # None → dropped
    "score":     0.98,
    "nested":    {"key": "value"},                 # dict → JSON string
    "count":     42,
    "published": True,
}

clean = sanitize_metadata(raw)
# {
#   "title":     "Enterprise RAG Guide",
#   "tags":      "rag, llm, production",
#   "score":     0.98,
#   "count":     42,
#   "published": True,
#   "nested":    '{"key": "value"}'
# }

# Safe to pass to any backend:
chunk = DocumentChunk("c1", "d1", "Some text", metadata=clean)
db.add([chunk])
```

---

