<div align="center">

# 🦊 Fennec Community

**A production-grade Python framework for building intelligent RAG pipelines**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

</div>

---

## Overview

**Fennec Community** is a modular, batteries-included Python library for building Retrieval-Augmented Generation (RAG) systems. It provides every layer of the RAG stack — from document loading and intelligent chunking to LLM abstraction, vector storage, prompt engineering, and query routing — in a single coherent package with first-class Arabic language support.

---

## ✨ Key Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini, Mistral, Groq, and Ollama via a unified interface
- **Intelligent chunking** — Semantic, adaptive, structure-aware, context-aware, and dedicated Arabic chunking strategies
- **Rich RAG variants** — Conversational, Agentic, Graph, Hybrid, Multi-hop, Domain-specific, Federated, Streaming, and Self-improving RAG
- **Vector database backends** — FAISS, Chroma, and Pinecone out of the box
- **Advanced prompt engineering** — Strategy-based prompt building with guardrails, optimization, and multi-format output (OpenAI / Anthropic)
- **Chain composition** — Build pipelines with `>>` operator using Sequential, Parallel, and Conditional chains
- **Semantic routing** — Hierarchical query routing with embedding-based scoring, caching, and feedback
- **Plugin system** — Extensible plugin framework with security sandboxing and LLM tool-descriptor generation
- **Output parsing** — Structured output parsing (JSON, YAML, CSV, Pydantic) with auto-fixing and retry handling
- **First-class Arabic support** — Dedicated Arabic chunker, embedder, normalizer, and RAG prompt routing

---

## 📦 Installation

```bash
pip install fennec-community
```

Or install from source:

```bash
git clone https://github.com/your-org/fennec-community.git
cd fennec-community
pip install -e .
```

---

## 🗂️ Architecture

```
fennec_community/
├── llm/                  # LLM provider interfaces
├── embeddings/           # Embedding providers (OpenAI, Gemini, HuggingFace, Arabic…)
├── document_loaders/     # File & web loaders (PDF, DOCX, CSV, HTML, JSON…)
├── chunks/               # Chunking strategies & orchestrator
├── vector_database/      # FAISS, Chroma, Pinecone backends
├── context/              # Context Intelligence Engine
├── rag/
│   ├── core/             # Base RAG system, reranker, query expansion, caching
│   ├── types/            # RAG variants (Agentic, Graph, Hybrid, Conversational…)
│   └── rag_ui/           # Ready-made RAG chat UI adapter
├── prompt/               # Prompt engine, builder, guardrails, optimizer
├── chain/                # Composable chain primitives
├── router/               # Hierarchical semantic router
├── output_parser/        # Structured output parsing & validation
└── plugins/              # Plugin framework & lifecycle manager
```

---

## 🚀 Quick Start

### Build a RAG pipeline in minutes

```python
from fennec_community.llm import OpenAIInterface
from fennec_community.embeddings import OpenAIEmbedder
from fennec_community.vector_database import FAISSVectorDatabase
from fennec_community.chunks import ChunkManager
from fennec_community.context import ContextEngine
from fennec_community.rag.core import RAGSystem

# 1. Initialize components
llm     = OpenAIInterface(api_key="YOUR_KEY", model="gpt-4o")
embedder = OpenAIEmbedder(api_key="YOUR_KEY")
vector_db = FAISSVectorDatabase(embedder=embedder)
chunker   = ChunkManager()
context   = ContextEngine()

# 2. Build the RAG system
rag = RAGSystem(
    vector_db=vector_db,
    llm=llm,
    chunker=chunker,
    context_manager=context,
)

# 3. Ingest documents
rag.add_documents(["path/to/document.pdf"])

# 4. Query
answer = rag.query("What is the main topic of the document?")
print(answer)
```

---

### Compose chains with `>>`

```python
from fennec_community.chain import BaseChain, SequentialChain

class FetchChain(BaseChain):
    async def _aexecute(self, inp):
        ...

class SummarizeChain(BaseChain):
    async def _aexecute(self, inp):
        ...

pipeline = FetchChain() >> SummarizeChain()
result   = pipeline.run("https://example.com/article")
```

---

### Arabic RAG

```python
from fennec_community.chunks import ArabicTextChunker
from fennec_community.embeddings import ArabicEmbedder

chunker  = ArabicTextChunker()
embedder = ArabicEmbedder()

chunks = chunker.chunk("النص العربي يذهب هنا...")
```

---

### Build a prompt for any LLM

```python
from fennec_community.prompt import PromptEngine

engine = PromptEngine()

prompt = engine.build(
    query="What caused the 2008 financial crisis?",
    documents=[{"content": "...", "source": "wiki", "score": 0.92}],
    strategy="multi_hop",
    output_format="json",
)

# Use with OpenAI
messages = prompt.to_messages()

# Use with Anthropic
payload = prompt.to_anthropic()
```

---

### Semantic query routing

```python
from fennec_community.router import HierarchicalRouter, Route

router = HierarchicalRouter()
router.add_route(Route(name="rag",   handler=rag_handler,  keywords=["document", "search"]))
router.add_route(Route(name="tools", handler=tool_handler, keywords=["calculate", "run"]))

result = await router.route("Search for recent papers on LLMs")
await result.execute()
```

---

## 🤖 Supported LLM Providers

| Provider   | Interface               |
|------------|-------------------------|
| OpenAI     | `OpenAIInterface`       |
| Anthropic  | `AnthropicInterface`    |
| Google Gemini | `GeminiInterface`    |
| Mistral    | `MistralInterface`      |
| Groq       | `GroqInterface`         |
| Ollama     | `OllamaInterface`       |

---

## 📄 Supported Document Types

PDF, DOCX, TXT, Markdown, CSV, Excel, JSON, JSONL, HTML, and web URLs — with `AutoLoader` for automatic type detection.

---

## 🧩 RAG Variants

| Variant | Class |
|---------|-------|
| Standard | `RAGSystem` |
| Conversational | `ConversationalRAG` |
| Agentic | `AgenticRAG` |
| Graph | `GraphRAG` |
| Hybrid Search | `HybridSearchRAG` |
| Multi-hop | `MultiHopRAG` |
| Multi-document | `MultiDocRAG` |
| Domain-specific | `DomainSpecificRAG` |
| Federated | `FederatedRAG` |
| Streaming | `StreamingRAG` |
| Self-improving | `SelfImprovingRAG` |

---

## 🔧 Configuration

All subsystems are configurable via dataclasses:

```python
from fennec_community.rag.core import RAGConfig

config = RAGConfig(
    chunk_size=512,
    overlap=128,
    top_k=5,
    enable_reranking=True,
    rerank_mode="hybrid",       # "heuristic" | "llm" | "hybrid"
    enable_retrieval_cache=True,
    cache_ttl=300.0,
)
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
