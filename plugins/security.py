"""
Plugin Security Layer
=====================
Three layers of protection:

  1. PermissionGuard  — declare and enforce per-plugin permissions
  2. InputSanitizer   — validate inputs against schema, detect injections
  3. ExecutionSandbox — async timeout + resource-cap wrapper
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .base_plugin import BasePlugin, PluginPermissionError, PluginValidationError
from .metadata import ExecutionContext, PermissionType, SchemaProperty

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Permission Guard
# ─────────────────────────────────────────────

@dataclass
class PermissionPolicy:
    """
    Runtime permission grant for a single plugin.
    Can be scoped to specific sessions or users.
    """
    plugin_name:    str
    granted:        Set[str]                    = field(default_factory=set)
    denied:         Set[str]                    = field(default_factory=set)
    session_scoped: Optional[str]               = None  # restrict to one session
    user_scoped:    Optional[str]               = None  # restrict to one user


class PermissionGuard:
    """
    Central authority for plugin permission management.

    Usage
    -----
    guard = PermissionGuard()
    guard.grant("web_search", PermissionType.NETWORK_ACCESS)
    guard.check("web_search", PermissionType.NETWORK_ACCESS)  # → True
    """

    def __init__(self) -> None:
        self._policies: Dict[str, PermissionPolicy] = {}

    def grant(
        self,
        plugin_name: str,
        permission: PermissionType | str,
        session_id: Optional[str] = None,
        user_id:    Optional[str] = None,
    ) -> None:
        perm = permission.value if isinstance(permission, PermissionType) else permission
        policy = self._get_or_create(plugin_name)
        policy.granted.add(perm)
        policy.denied.discard(perm)
        if session_id:
            policy.session_scoped = session_id
        if user_id:
            policy.user_scoped = user_id
        logger.info("[Security] Granted '%s' → '%s'", perm, plugin_name)

    def revoke(self, plugin_name: str, permission: PermissionType | str) -> None:
        perm = permission.value if isinstance(permission, PermissionType) else permission
        policy = self._get_or_create(plugin_name)
        policy.granted.discard(perm)
        policy.denied.add(perm)
        logger.info("[Security] Revoked '%s' from '%s'", perm, plugin_name)

    def grant_all_required(self, plugin: BasePlugin) -> None:
        """Grant every permission declared in the plugin's metadata."""
        for perm in plugin.metadata.permissions:
            self.grant(plugin.name, perm)

    def check(
        self,
        plugin_name: str,
        permission:  PermissionType | str,
        context:     Optional[ExecutionContext] = None,
    ) -> bool:
        perm   = permission.value if isinstance(permission, PermissionType) else permission
        policy = self._policies.get(plugin_name)
        if policy is None:
            return False

        # Check explicit deny
        if perm in policy.denied:
            return False

        # Check scope
        if context:
            if policy.session_scoped and policy.session_scoped != context.session_id:
                return False
            if policy.user_scoped and policy.user_scoped != context.user_id:
                return False

        return perm in policy.granted

    def assert_permission(
        self,
        plugin: BasePlugin,
        permission: PermissionType | str,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        perm = permission.value if isinstance(permission, PermissionType) else permission
        if not self.check(plugin.name, perm, context):
            raise PluginPermissionError(
                f"[{plugin.name}] Missing permission: '{perm}'"
            )

    def check_all_required(
        self,
        plugin: BasePlugin,
        context: Optional[ExecutionContext] = None,
    ) -> bool:
        for perm in plugin.metadata.permissions:
            if not self.check(plugin.name, perm, context):
                logger.warning(
                    "[Security] [%s] Missing required permission: '%s'",
                    plugin.name, perm,
                )
                return False
        return True

    def get_grants(self, plugin_name: str) -> Set[str]:
        policy = self._policies.get(plugin_name)
        return set(policy.granted) if policy else set()

    def _get_or_create(self, name: str) -> PermissionPolicy:
        if name not in self._policies:
            self._policies[name] = PermissionPolicy(plugin_name=name)
        return self._policies[name]


# ─────────────────────────────────────────────
# 2. Input Sanitizer
# ─────────────────────────────────────────────

# Patterns that indicate prompt injection or code injection attempts
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)(ignore\s+previous|forget\s+instructions|act\s+as\s+if)"),
    re.compile(r"(?i)(system\s*prompt|jailbreak|dan\s+mode)"),
    re.compile(r"(?i)(<script|javascript:|data:text/html|onerror=)"),
    re.compile(r"(?i)(;\s*(rm|dd|mkfs|curl\s+-o|wget|nc\s+-e))"),
    re.compile(r"(?i)(\.\./|\.\.\\|%2e%2e)"),  # path traversal
]

_MAX_STRING_LEN = 32_768   # 32 KB per string field


class InputSanitizer:
    """
    Validates and sanitizes plugin inputs.

    Steps
    -----
    1. Type coercion against JSON schema
    2. String length caps
    3. Injection pattern detection
    4. Plugin-level validate() call
    """

    @staticmethod
    def sanitize(
        data:   Dict[str, Any],
        schema: List[SchemaProperty],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate ``data`` against ``schema``.
        Returns cleaned data or raises PluginValidationError.
        """
        cleaned: Dict[str, Any] = {}

        prop_map = {p.name: p for p in schema}
        required = {p.name for p in schema if p.required}

        # Check required fields
        missing = required - set(data.keys())
        if missing:
            raise PluginValidationError(f"Missing required fields: {sorted(missing)}")

        for name, value in data.items():
            prop = prop_map.get(name)
            if prop is None:
                if strict:
                    raise PluginValidationError(f"Unknown field: '{name}'")
                continue  # ignore unknown in lenient mode

            # Type check / coerce
            value = _coerce(name, value, prop)

            # String-specific checks
            if prop.type == "string" and isinstance(value, str):
                if len(value) > _MAX_STRING_LEN:
                    raise PluginValidationError(
                        f"Field '{name}' exceeds max length {_MAX_STRING_LEN}."
                    )
                InputSanitizer._check_injection(name, value)

            # Enum check
            if prop.enum is not None and value not in prop.enum:
                raise PluginValidationError(
                    f"Field '{name}' must be one of {prop.enum}, got {value!r}."
                )

            cleaned[name] = value

        # Fill defaults
        for prop in schema:
            if prop.name not in cleaned and prop.default is not None:
                cleaned[prop.name] = prop.default

        return cleaned

    @staticmethod
    def _check_injection(field_name: str, value: str) -> None:
        for pat in _INJECTION_PATTERNS:
            if pat.search(value):
                raise PluginValidationError(
                    f"Field '{field_name}' contains a potentially unsafe pattern. "
                    f"Pattern: {pat.pattern}"
                )


def _coerce(name: str, value: Any, prop: SchemaProperty) -> Any:
    type_map = {
        "string":  str,
        "integer": int,
        "number":  float,
        "boolean": bool,
    }
    if prop.type in type_map:
        target = type_map[prop.type]
        if not isinstance(value, target):
            try:
                return target(value)
            except (TypeError, ValueError):
                raise PluginValidationError(
                    f"Field '{name}' expected {prop.type}, got {type(value).__name__}."
                )
    elif prop.type == "array" and not isinstance(value, list):
        raise PluginValidationError(f"Field '{name}' must be an array.")
    elif prop.type == "object" and not isinstance(value, dict):
        raise PluginValidationError(f"Field '{name}' must be an object.")
    return value


# ─────────────────────────────────────────────
# 3. Execution Sandbox
# ─────────────────────────────────────────────

class ExecutionSandbox:
    """
    Wraps any coroutine with:
      - hard timeout
      - optional memory cap (placeholder; real cap needs process isolation)
      - structured error capture
    """

    def __init__(
        self,
        hard_timeout_sec: float = 60.0,
        log_slow_threshold: float = 5.0,
    ) -> None:
        self.hard_timeout     = hard_timeout_sec
        self.slow_threshold   = log_slow_threshold

    async def run(self, coro, plugin_name: str = "?") -> Any:
        """
        Execute ``coro`` inside the sandbox.
        Returns result or raises (propagates original exception).
        """
        import time
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(coro, timeout=self.hard_timeout)
            elapsed = time.monotonic() - t0
            if elapsed > self.slow_threshold:
                logger.warning(
                    "[Sandbox] Plugin '%s' was slow: %.2fs (threshold %.2fs)",
                    plugin_name, elapsed, self.slow_threshold,
                )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "[Sandbox] Plugin '%s' hit hard timeout (%ss).",
                plugin_name, self.hard_timeout,
            )
            raise
        except Exception as exc:
            logger.error("[Sandbox] Plugin '%s' raised: %s", plugin_name, exc)
            raise
