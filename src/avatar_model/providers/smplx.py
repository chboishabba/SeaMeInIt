"""Provider implementation that wraps the official SMPL-X model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

import smplx

from .base import BodyModelProvider, register_provider

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from avatar_model.body_model import BodyModelConfig
    from smplx.output import SMPLXOutput as _SMPLXOutput
else:  # pragma: no cover - fallback when typing module unavailable
    _SMPLXOutput = Any


class SmplxProvider(BodyModelProvider):
    """Concrete provider that exposes the SMPL-X parametric model."""

    _PARAMETER_KEYS: tuple[str, ...] = (
        "betas",
        "expression",
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "transl",
    )

    _DEFAULT_DIMS: Mapping[str, int] = {
        "betas": 10,
        "expression": 10,
        "global_orient": 3,
        "body_pose": 63,
        "left_hand_pose": 45,
        "right_hand_pose": 45,
        "jaw_pose": 3,
        "leye_pose": 3,
        "reye_pose": 3,
        "transl": 3,
    }

    def __init__(self, config: "BodyModelConfig") -> None:
        if not isinstance(config.model_path, Path):
            config = replace(config, model_path=Path(config.model_path))

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.batch_size = config.batch_size

        self._verify_assets()

        self._model = smplx.create(
            model_path=str(config.model_path),
            model_type=config.model_type,
            gender=config.gender,
            batch_size=config.batch_size,
            num_betas=config.num_betas,
            num_expression_coeffs=config.num_expression_coeffs,
        ).to(self.device)
        self._model.eval()

        self._parameters: Dict[str, Tensor] = {}
        self._initialise_parameters()

    def parameter_shapes(self) -> Mapping[str, torch.Size]:
        return {name: tensor.shape for name, tensor in self._parameters.items()}

    def parameters(self, as_numpy: bool = False) -> Mapping[str, Tensor | np.ndarray]:
        if as_numpy:
            return {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in self._parameters.items()
            }
        return {name: tensor.clone() for name, tensor in self._parameters.items()}

    def set_parameters(self, updates: Mapping[str, np.ndarray | Tensor]) -> None:
        for name, value in updates.items():
            self._set_parameter(name, value, additive=False)

    def adjust_parameters(self, deltas: Mapping[str, np.ndarray | Tensor]) -> None:
        for name, value in deltas.items():
            self._set_parameter(name, value, additive=True)

    def set_shape(self, betas: np.ndarray | Tensor) -> None:
        self._set_parameter("betas", betas, additive=False)

    def adjust_shape(self, delta_betas: np.ndarray | Tensor) -> None:
        self._set_parameter("betas", delta_betas, additive=True)

    def set_body_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None:
        if body_pose is not None:
            self._set_parameter("body_pose", body_pose, additive=False)
        if global_orient is not None:
            self._set_parameter("global_orient", global_orient, additive=False)
        if transl is not None:
            self._set_parameter("transl", transl, additive=False)

    def adjust_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None:
        if body_pose is not None:
            self._set_parameter("body_pose", body_pose, additive=True)
        if global_orient is not None:
            self._set_parameter("global_orient", global_orient, additive=True)
        if transl is not None:
            self._set_parameter("transl", transl, additive=True)

    @torch.no_grad()
    def forward(self, **kwargs: object) -> _SMPLXOutput:
        return self._model(**self._parameters, **kwargs)

    @torch.no_grad()
    def vertices(self) -> Tensor:
        output = self.forward(return_verts=True)
        return output.vertices

    @torch.no_grad()
    def joints(self) -> Tensor:
        output = self.forward(return_verts=True)
        return output.joints

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    # ------------------------------------------------------------------
    def _verify_assets(self) -> None:
        assets_root = self.config.model_path
        if not assets_root.exists():
            msg = (
                "SMPL-X assets are required but were not found at"
                f" {assets_root!s}. Download the official archive and extract it"
                " with `python tools/download_smplx.py --dest assets/smplx`."
            )
            raise FileNotFoundError(msg)

        model_dir = assets_root / self.config.model_type
        gender_suffix = self.config.gender.upper()
        model_name = f"{self.config.model_type.upper()}_{gender_suffix}.npz"
        model_file = model_dir / model_name
        if not model_file.exists():
            msg = (
                f"Expected SMPL-X asset {model_name} in {model_dir!s}, but it was not"
                " found. Ensure you have extracted the official SMPL-X release for"
                f" the {self.config.gender} model variant."
            )
            raise FileNotFoundError(msg)

    def _initialise_parameters(self) -> None:
        for name in self._PARAMETER_KEYS:
            template = getattr(self._model, name, None)
            if isinstance(template, Tensor):
                dim = template.shape[-1]
            else:
                dim = self._DEFAULT_DIMS[name]
            shape = (self.batch_size, dim)
            self._parameters[name] = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _set_parameter(self, name: str, value: np.ndarray | Tensor, *, additive: bool) -> None:
        if name not in self._parameters:
            msg = f"Unknown SMPL-X parameter '{name}'."
            raise KeyError(msg)

        target = self._parameters[name]
        tensor = self._coerce_to_tensor(value, target.dtype, target.device)
        if tensor.shape != target.shape:
            expected_elements = int(torch.prod(torch.tensor(target.shape, device="cpu")))
            if tensor.numel() != expected_elements:
                msg = (
                    f"Parameter '{name}' has shape {tensor.shape}, expected {target.shape} "
                    "or a compatible number of elements."
                )
                raise ValueError(msg)
            tensor = tensor.reshape(target.shape)

        if additive:
            target.add_(tensor)
        else:
            target.copy_(tensor)

    def _coerce_to_tensor(
        self, value: np.ndarray | Tensor, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
        return tensor


register_provider("smplx", SmplxProvider)
