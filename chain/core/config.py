"""
==============
Per-chain configuration controlling retries, timeouts, and fallbacks.

Example
-------
>>> cfg = ChainConfig(retries=3, timeout=5.0)
>>> cfg = ChainConfig(retries=2, fallback_chain=my_backup_chain)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import BaseChain


@dataclass
class ChainConfig:
    """
    Configuration object attached to every BaseChain instance.

    Attributes
    ----------
    retries        : How many times to retry on failure (0 = no retry).
    retry_delay    : Seconds to wait between retries.
    timeout        : Maximum seconds allowed for a single execution.
                     None means no timeout.
    fallback_chain : Chain to execute if all retries are exhausted.
    verbose        : Emit debug logging when True.
    """

    retries: int = 0
    retry_delay: float = 0.5
    timeout: Optional[float] = None
    fallback_chain: Optional["BaseChain"] = field(default=None, repr=False)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be > 0 or None")
