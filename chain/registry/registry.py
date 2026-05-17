"""
====================
Plugin / registry system for dynamic chain registration and lookup.

Example
-------
>>> ChainRegistry.register("transform", TransformChain)
>>> ChainRegistry.register("parallel", ParallelChain)

>>> cls = ChainRegistry.get("transform")
>>> chain = cls(my_fn)

Decorator shorthand
-------------------
>>> @ChainRegistry.chain("my_custom")
... class MyCustomChain(BaseChain):
...     ...
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Type
from ..core.base import BaseChain
logger = logging.getLogger(__name__)


class ChainRegistry:
    """
    Global registry mapping string aliases to chain classes.
    """

    _registry: Dict[str, Type[BaseChain]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, alias: str, chain_cls: Type[BaseChain]) -> None:
        """
        Register a chain class under the given alias.

        Parameters
        ----------
        alias     : Unique string key.
        chain_cls : Class (not instance) that extends BaseChain.

        Raises
        ------
        TypeError if chain_cls is not a subclass of BaseChain.
        """
        if not (isinstance(chain_cls, type) and issubclass(chain_cls, BaseChain)):
            raise TypeError(f"{chain_cls!r} is not a BaseChain subclass")
        if alias in cls._registry:
            logger.warning("ChainRegistry: overwriting alias '%s'", alias)
        cls._registry[alias] = chain_cls
        logger.debug("ChainRegistry: registered '%s' → %s", alias, chain_cls.__name__)

    @classmethod
    def chain(cls, alias: str) -> Callable[[Type[BaseChain]], Type[BaseChain]]:
        """
        Class decorator that registers on definition.

        Example
        -------
        >>> @ChainRegistry.chain("echo")
        ... class EchoChain(BaseChain):
        ...     ...
        """

        def decorator(chain_cls: Type[BaseChain]) -> Type[BaseChain]:
            cls.register(alias, chain_cls)
            return chain_cls

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, alias: str) -> Type[BaseChain]:
        """
        Return the class registered under *alias*.

        Raises
        ------
        KeyError if the alias is not registered.
        """
        if alias not in cls._registry:
            raise KeyError(
                f"No chain registered as '{alias}'. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[alias]

    @classmethod
    def build(cls, alias: str, *args: Any, **kwargs: Any) -> BaseChain:
        """
        Instantiate the chain registered under *alias*.

        Example
        -------
        >>> chain = ChainRegistry.build("transform", my_fn)
        """
        return cls.get(alias)(*args, **kwargs)

    @classmethod
    def list_all(cls) -> Dict[str, str]:
        """Return a dict of {alias: class_name}."""
        return {alias: klass.__name__ for alias, klass in cls._registry.items()}

    @classmethod
    def unregister(cls, alias: str) -> None:
        """Remove an alias (useful in tests)."""
        cls._registry.pop(alias, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry (useful in tests)."""
        cls._registry.clear()


# ------------------------------------------------------------------
# Auto-register built-in chains
# ------------------------------------------------------------------

def _register_builtins() -> None:
    from ..chains.sequential import SequentialChain
    from ..chains.parallel import ParallelChain
    from ..chains.conditional import ConditionalChain, TransformChain

    ChainRegistry.register("sequential", SequentialChain)
    ChainRegistry.register("parallel", ParallelChain)
    ChainRegistry.register("conditional", ConditionalChain)
    ChainRegistry.register("transform", TransformChain)


_register_builtins()
