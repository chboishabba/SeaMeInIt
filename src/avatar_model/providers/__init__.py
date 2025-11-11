"""Body model provider registry and default implementations."""

from __future__ import annotations

from .base import (
    BodyModelProvider,
    ProviderFactory,
    available_providers,
    create_provider,
    get_provider_factory,
    register_provider,
)

# Ensure built-in providers are registered on import.
from . import smplx as _smplx_provider  # noqa: F401
from . import smplerx as _smplerx_provider  # noqa: F401

__all__ = [
    "BodyModelProvider",
    "ProviderFactory",
    "available_providers",
    "create_provider",
    "get_provider_factory",
    "register_provider",
]
