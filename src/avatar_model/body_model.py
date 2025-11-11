"""High level wrapper around avatar body model providers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from .providers import BodyModelProvider, available_providers, create_provider

if TYPE_CHECKING:
    from smplx.output import SMPLXOutput as _SMPLXOutput
else:  # pragma: no cover - fallback when smplx is unavailable
    _SMPLXOutput = Any

__all__ = ["BodyModel", "BodyModelConfig"]


@dataclass(slots=True)
class BodyModelConfig:
    """Configuration options for :class:`BodyModel`."""

    model_path: Path
    """Directory containing the body model assets."""

    model_type: str = "smplx"
    """Registered provider name to instantiate."""

    gender: str = "neutral"
    """Gendered model variant to load (``neutral``, ``male``, ``female``)."""

    device: torch.device | str = "cpu"
    """Device on which to instantiate the model."""

    batch_size: int = 1
    """Number of body instances to evaluate per forward pass."""

    num_betas: int = 10
    """Number of shape blend weights to expose."""

    num_expression_coeffs: int = 10
    """Number of facial expression coefficients."""

    dtype: torch.dtype = torch.float32
    """Floating point precision for model parameters."""


class BodyModel:
    """User-facing facade that delegates to registered model providers."""

    def __init__(self, config: BodyModelConfig | None = None, **kwargs: object) -> None:
        """Instantiate a body model wrapper backed by a provider."""

        if config is None:
            if "model_path" not in kwargs:
                msg = "Either a BodyModelConfig or a model_path must be provided."
                raise ValueError(msg)
            config = BodyModelConfig(model_path=Path(kwargs.pop("model_path")), **kwargs)  # type: ignore[arg-type]
        elif kwargs:
            msg = "Provide configuration through either a BodyModelConfig or keyword arguments, not both."
            raise ValueError(msg)

        if not isinstance(config.model_path, Path):
            config = replace(config, model_path=Path(config.model_path))

        self.config = config

        try:
            self._provider: BodyModelProvider = create_provider(config.model_type, config)
        except KeyError as exc:
            options = ", ".join(available_providers()) or "<none>"
            msg = (
                f"Unknown body model provider '{config.model_type}'. "
                f"Available providers: {options}."
            )
            raise ValueError(msg) from exc
        except NotImplementedError as exc:
            raise NotImplementedError(str(exc)) from exc

    # ------------------------------------------------------------------
    @property
    def provider(self) -> BodyModelProvider:
        """Return the underlying provider instance."""

        return self._provider

    @property
    def model(self) -> torch.nn.Module:
        """Expose the low-level model object from the provider."""

        return self._provider.model

    @property
    def device(self) -> torch.device:
        """Device used by the provider."""

        return self._provider.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating point precision configured on the provider."""

        return self._provider.dtype

    @property
    def batch_size(self) -> int:
        """Batch size configured on the provider."""

        return self._provider.batch_size

    # ------------------------------------------------------------------
    def parameter_shapes(self) -> Mapping[str, torch.Size]:
        return self._provider.parameter_shapes()

    def parameters(self, as_numpy: bool = False) -> Mapping[str, Tensor | np.ndarray]:
        return self._provider.parameters(as_numpy=as_numpy)

    def set_parameters(self, updates: Mapping[str, np.ndarray | Tensor]) -> None:
        self._provider.set_parameters(updates)

    def adjust_parameters(self, deltas: Mapping[str, np.ndarray | Tensor]) -> None:
        self._provider.adjust_parameters(deltas)

    def set_shape(self, betas: np.ndarray | Tensor) -> None:
        self._provider.set_shape(betas)

    def adjust_shape(self, delta_betas: np.ndarray | Tensor) -> None:
        self._provider.adjust_shape(delta_betas)

    def set_body_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None:
        self._provider.set_body_pose(
            body_pose=body_pose, global_orient=global_orient, transl=transl
        )

    def adjust_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None:
        self._provider.adjust_pose(body_pose=body_pose, global_orient=global_orient, transl=transl)

    def forward(self, **kwargs: object) -> _SMPLXOutput:
        return self._provider.forward(**kwargs)

    def vertices(self) -> Tensor:
        return self._provider.vertices()

    def joints(self) -> Tensor:
        return self._provider.joints()
