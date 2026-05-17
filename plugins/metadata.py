"""
Plugin Metadata System — AI-Ready, Schema-Driven
=================================================
Rich metadata that enables LLMs and agents to intelligently select,
describe, and invoke plugins via function calling / tool use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class PluginStatus(str, Enum):
    DISABLED     = "disabled"
    ENABLED      = "enabled"
    LOADING      = "loading"
    ERROR        = "error"
    UPDATING     = "updating"
    DEPRECATED   = "deprecated"
    INITIALIZING = "initializing"


class PluginPriority(int, Enum):
    CRITICAL   = 1000
    HIGH       = 100
    NORMAL     = 50
    LOW        = 10
    BACKGROUND = 1


class PermissionType(str, Enum):
    FILE_READ        = "file_read"
    FILE_WRITE       = "file_write"
    NETWORK_ACCESS   = "network_access"
    SYSTEM_CALL      = "system_call"
    DATABASE_ACCESS  = "database_access"
    USER_DATA_ACCESS = "user_data_access"
    LLM_CALL         = "llm_call"
    VECTOR_STORE     = "vector_store"


class CostTier(str, Enum):
    FREE      = "free"       # no external calls
    CHEAP     = "cheap"      # < $0.001 per call
    MODERATE  = "moderate"   # $0.001 – $0.01
    EXPENSIVE = "expensive"  # > $0.01


# ─────────────────────────────────────────────
# JSON-Schema-compatible property descriptor
# ─────────────────────────────────────────────

@dataclass
class SchemaProperty:
    """
    Describes one field in a plugin's input or output schema.
    Compatible with JSON Schema draft-07 and OpenAI function-calling format.
    """
    name:        str
    type:        Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    required:    bool                     = True
    enum:        Optional[List[Any]]      = None
    default:     Optional[Any]            = None
    items:       Optional[Dict[str, Any]] = None   # for type=array
    properties:  Optional[Dict[str, Any]] = None   # for type=object

    def to_json_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "type":        self.type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.items is not None:
            schema["items"] = self.items
        if self.properties is not None:
            schema["properties"] = self.properties
        return schema


# ─────────────────────────────────────────────
# Plugin Metadata
# ─────────────────────────────────────────────

@dataclass
class PluginMetadata:
    """
    Complete, AI-queryable metadata for a plugin.

    The ``capabilities``, ``tags``, ``use_cases``, and schema fields allow
    an LLM-based orchestrator to:
      1. Find the right plugin for a user query (semantic/keyword match).
      2. Build a function-calling descriptor automatically.
      3. Estimate cost before execution.
    """
    # ── Identity ───────────────────────────────
    name:        str
    version:     str
    author:      str
    description: str

    # ── AI discovery fields ────────────────────
    capabilities: List[str]        = field(default_factory=list)
    tags:         List[str]        = field(default_factory=list)
    use_cases:    List[str]        = field(default_factory=list)
    cost_tier:    CostTier         = CostTier.FREE
    cost_per_call_usd: float       = 0.0        # estimated per-call cost

    # ── Schema ────────────────────────────────
    input_schema:  List[SchemaProperty] = field(default_factory=list)
    output_schema: List[SchemaProperty] = field(default_factory=list)

    # ── Compatibility / dependencies ──────────
    dependencies:        List[str] = field(default_factory=list)
    required_hooks:      List[str] = field(default_factory=list)
    permissions:         List[str] = field(default_factory=list)
    min_system_version:  str       = "1.0.0"
    max_system_version:  str       = "999.0.0"

    # ── Misc ──────────────────────────────────
    homepage:   str               = ""
    license:    str               = "MIT"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # ─────────────────────────────────────────
    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Plugin name is required.")
        if not _valid_version(self.version):
            raise ValueError(f"Invalid semantic version: '{self.version}'")
        if not self.description or not self.description.strip():
            raise ValueError("Plugin description is required.")
        if not self.author or not self.author.strip():
            raise ValueError("Plugin author is required.")
        self.created_at = self.created_at or datetime.utcnow()
        self.updated_at = self.updated_at or datetime.utcnow()

    # ─────────────────────────────────────────
    # AI Tool / Function-Calling descriptor
    # ─────────────────────────────────────────

    def to_tool_descriptor(self) -> Dict[str, Any]:
        """
        Converts this metadata into an OpenAI / Anthropic function-calling
        compatible descriptor that an LLM can use directly.

        Example output (OpenAI format):
        {
          "type": "function",
          "function": {
            "name": "web_search",
            "description": "...",
            "parameters": { ... }
          }
        }
        """
        properties: Dict[str, Any] = {}
        required_fields: List[str] = []

        for prop in self.input_schema:
            properties[prop.name] = prop.to_json_schema()
            if prop.required:
                required_fields.append(prop.name)

        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self._build_llm_description(),
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required_fields,
                },
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Anthropic Claude tool-use format."""
        properties: Dict[str, Any] = {}
        required_fields: List[str] = []

        for prop in self.input_schema:
            properties[prop.name] = prop.to_json_schema()
            if prop.required:
                required_fields.append(prop.name)

        return {
            "name":        self.name,
            "description": self._build_llm_description(),
            "input_schema": {
                "type":       "object",
                "properties": properties,
                "required":   required_fields,
            },
        }

    def _build_llm_description(self) -> str:
        """Build a rich description for LLM consumption."""
        parts = [self.description]
        if self.use_cases:
            parts.append("Use cases: " + "; ".join(self.use_cases))
        if self.cost_tier != CostTier.FREE:
            parts.append(f"Cost: {self.cost_tier.value} (~${self.cost_per_call_usd:.4f}/call)")
        return " | ".join(parts)

    # ─────────────────────────────────────────
    # Compatibility helpers
    # ─────────────────────────────────────────

    def is_compatible_with(self, system_version: str) -> bool:
        return (
            _compare_versions(system_version, self.min_system_version) >= 0
            and _compare_versions(system_version, self.max_system_version) <= 0
        )

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":              self.name,
            "version":           self.version,
            "author":            self.author,
            "description":       self.description,
            "capabilities":      self.capabilities,
            "tags":              self.tags,
            "use_cases":         self.use_cases,
            "cost_tier":         self.cost_tier.value,
            "cost_per_call_usd": self.cost_per_call_usd,
            "dependencies":      self.dependencies,
            "required_hooks":    self.required_hooks,
            "permissions":       self.permissions,
            "min_system_version": self.min_system_version,
            "max_system_version": self.max_system_version,
            "homepage":          self.homepage,
            "license":           self.license,
            "created_at":        self.created_at.isoformat() if self.created_at else None,
            "updated_at":        self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        data = dict(data)
        for key in ("created_at", "updated_at"):
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        if "cost_tier" in data:
            data["cost_tier"] = CostTier(data["cost_tier"])
        return cls(**data)


# ─────────────────────────────────────────────
# Plugin execution context
# ─────────────────────────────────────────────

@dataclass
class ExecutionContext:
    """
    Runtime context passed to every plugin.execute() call.
    Carries RAG-relevant state: query, memory, cache, session, etc.
    """
    query:       str                     = ""
    session_id:  str                     = ""
    user_id:     str                     = ""
    metadata:    Dict[str, Any]          = field(default_factory=dict)
    memory:      Optional[Any]           = None   # memory store handle
    cache:       Optional[Any]           = None   # cache handle
    router:      Optional[Any]           = None   # router handle
    trace_id:    str                     = ""
    max_tokens:  int                     = 2048


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_SEM_VER = re.compile(r"^\d+\.\d+\.\d+$")


def _valid_version(v: str) -> bool:
    return bool(_SEM_VER.match(v or ""))


def _compare_versions(v1: str, v2: str) -> int:
    p1 = [int(x) for x in v1.split(".")]
    p2 = [int(x) for x in v2.split(".")]
    for a, b in zip(p1, p2):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0
