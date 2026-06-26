"""Generic name->class registries (one factory, reused for each component type).

Mirrors ``Standard/standard/core/registry.py`` but factored so synthesizers and
matchers share the same implementation:

    SYNTH_REGISTRY, register_synth, get_synth = make_registry("synth")

Register a component by decorating its class and importing its module from the
package ``__init__`` so registration runs at import time.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple, Type


def make_registry(kind: str) -> Tuple[Dict[str, Type], Callable, Callable]:
    """Return ``(REGISTRY, register_decorator, getter)`` for one component kind."""
    registry: Dict[str, Type] = {}

    def register(name: str):
        def _decorator(cls):
            if name in registry:
                raise ValueError(f"{kind} '{name}' is already registered")
            registry[name] = cls
            cls.registered_name = name
            return cls
        return _decorator

    def get(name: str):
        if name not in registry:
            raise KeyError(
                f"Unknown {kind} '{name}'. Registered: {sorted(registry)}"
            )
        return registry[name]

    return registry, register, get
