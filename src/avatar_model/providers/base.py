"""Provider interface and registry for avatar body models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor

try:  # pragma: no cover - optional dependency typing help
    from smplx.output import SMPLXOutput as _SMPLXOutput
except Exception:  # pragma: no cover - fallback for runtime without smplx
    _SMPLXOutput = Any

try:  # pragma: no cover - typing guard to avoid circular import at runtime
    from avatar_model.body_model import BodyModelConfig
except Exception:  # pragma: no cover - runtime import is optional for typing
    BodyModelConfig = Any  # type: ignore[assignment]


@runtime_checkable
class BodyModelProvider(Protocol):
    """Minimal contract that concrete body model providers must fulfil."""

    config: BodyModelConfig
    device: torch.device
    dtype: torch.dtype
    batch_size: int

    def parameter_shapes(self) -> Mapping[str, torch.Size]: ...

    def parameters(self, as_numpy: bool = False) -> Mapping[str, Tensor | np.ndarray]: ...

    def set_parameters(self, updates: Mapping[str, np.ndarray | Tensor]) -> None: ...

    def adjust_parameters(self, deltas: Mapping[str, np.ndarray | Tensor]) -> None: ...

    def set_shape(self, betas: np.ndarray | Tensor) -> None: ...

    def adjust_shape(self, delta_betas: np.ndarray | Tensor) -> None: ...

    def set_body_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None: ...

    def adjust_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None: ...

    def forward(self, **kwargs: object) -> _SMPLXOutput: ...

    def vertices(self) -> Tensor: ...

    def joints(self) -> Tensor: ...

    @property
    def model(self) -> torch.nn.Module: ...


ProviderFactory = Callable[[BodyModelConfig], BodyModelProvider]


class ProviderRegistry:
    """Runtime registry for body model provider factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.lower()
        self._factories[key] = factory

    def get(self, name: str) -> ProviderFactory | None:
        return self._factories.get(name.lower())

    def names(self) -> Sequence[str]:
        return tuple(sorted(self._factories))


_registry = ProviderRegistry()


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a new body model provider factory."""

    _registry.register(name, factory)


def get_provider_factory(name: str) -> ProviderFactory | None:
    """Return the factory for the requested provider name."""

    return _registry.get(name)


def available_providers() -> Sequence[str]:
    """Return the list of registered provider names."""

    return _registry.names()


def create_provider(name: str, config: BodyModelConfig) -> BodyModelProvider:
    """Instantiate the provider identified by ``name``."""

    factory = get_provider_factory(name)
    if factory is None:
        msg = f"Unknown body model provider '{name}'."
        raise KeyError(msg)
    return factory(config)
