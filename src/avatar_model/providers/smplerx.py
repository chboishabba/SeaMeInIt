"""Scaffolding provider for the upcoming SMPLer-X body model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import BodyModelProvider, register_provider

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from torch import Tensor
    from avatar_model.body_model import BodyModelConfig


class SmplerxProvider(BodyModelProvider):
    """Placeholder implementation until the SMPLer-X API is published."""

    def __init__(self, config: "BodyModelConfig") -> None:  # pragma: no cover - placeholder
        self.config = config
        self.device = torch.device(getattr(config, "device", "cpu"))
        self.dtype = getattr(config, "dtype", torch.float32)
        self.batch_size = getattr(config, "batch_size", 1)
        raise NotImplementedError(
            "SmplerxProvider is registered for discovery but does not yet implement the body model API."
        )

    # The remaining protocol methods are intentionally left as stubs that raise
    # the same error. This keeps the provider discoverable without exposing an
    # unusable partial implementation.
    def _not_implemented(self, *args, **kwargs):  # pragma: no cover - placeholder
        raise NotImplementedError(
            "SmplerxProvider is registered for discovery but does not yet implement the body model API."
        )

    parameters = _not_implemented
    parameter_shapes = _not_implemented
    set_parameters = _not_implemented
    adjust_parameters = _not_implemented
    set_shape = _not_implemented
    adjust_shape = _not_implemented
    set_body_pose = _not_implemented
    adjust_pose = _not_implemented
    forward = _not_implemented
    vertices = _not_implemented
    joints = _not_implemented

    @property
    def model(self):  # pragma: no cover - placeholder
        self._not_implemented()


register_provider("smplerx", SmplerxProvider)
