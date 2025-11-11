"""High level wrapper around the SMPL-X body model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

import smplx

if TYPE_CHECKING:
    from smplx.output import SMPLXOutput as _SMPLXOutput
else:
    _SMPLXOutput = Any

__all__ = ["BodyModel", "BodyModelConfig"]


@dataclass(slots=True)
class BodyModelConfig:
    """Configuration options for :class:`BodyModel`."""

    model_path: Path
    """Directory containing the SMPL-compatible model files."""

    model_type: str = "smplx"
    """SMPL family model type to instantiate."""

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


def _load_manifest(root: Path) -> dict[str, object] | None:
    """Return the asset manifest for ``root`` if one exists."""

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def _bundle_name(manifest: dict[str, object] | None, assets_root: Path) -> str:
    """Heuristic for the short name of the asset bundle."""

    if manifest is not None:
        value = manifest.get("model")
        if isinstance(value, str) and value:
            return value
    return assets_root.name or "smplx"


class BodyModel:
    """User-friendly interface for loading and manipulating SMPL-X models.

    The wrapper keeps track of pose and shape parameters as ``torch.Tensor``
    instances while accepting input as either numpy arrays or tensors.  It also
    provides convenience helpers for generating mesh vertices and joints from
    the underlying SMPL-X module.
    """

    #: Parameter names managed by the wrapper.
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

    #: Fallback dimensionalities for parameters when not discoverable from the
    #: SMPL-X module.  Shapes are interpreted as ``(batch_size, dim)``.
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

    def __init__(self, config: BodyModelConfig | None = None, **kwargs: object) -> None:
        """Instantiate a SMPL-X body model wrapper.

        Parameters can either be supplied via a :class:`BodyModelConfig` or as
        keyword arguments mirroring the dataclass fields.  ``model_path`` is the
        only required option and should point to the folder containing SMPL
        family assets (for example ``assets/smplx`` or ``assets/smplerx``).  The
        provisioning helper writes a ``manifest.json`` alongside the assets,
        which is used to provide clearer error messages when files are missing.
        """

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
        self._verify_assets()

        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.batch_size = config.batch_size

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

    def _verify_assets(self) -> None:
        """Ensure the configured SMPL-X assets are present before loading."""

        assets_root = self.config.model_path
        manifest = _load_manifest(assets_root)
        bundle_name = _bundle_name(manifest, assets_root)

        if not assets_root.exists():
            msg = (
                "SMPL-compatible assets are required but were not found at"
                f" {assets_root!s}. Provision them with"
                f" `python tools/download_smplx.py --model {bundle_name}`"
                " (add --dest to override the default location)."
            )
            raise FileNotFoundError(msg)

        model_type = str(manifest.get("model_type", self.config.model_type)) if manifest else self.config.model_type
        model_dir = assets_root / model_type
        gender_suffix = self.config.gender.upper()
        model_name = f"{model_type.upper()}_{gender_suffix}.npz"
        model_file = model_dir / model_name
        if not model_file.exists():
            msg = (
                f"Expected SMPL-X asset {model_name} in {model_dir!s}, but it was not"
                " found. Ensure you have extracted the correct asset bundle"
                f" (try `python tools/download_smplx.py --model {bundle_name}`)"
                " and that the manifest matches the desired gender."
            )
            raise FileNotFoundError(msg)

    @property
    def model(self) -> torch.nn.Module:
        """Return the underlying SMPL-X module."""

        return self._model

    def parameter_shapes(self) -> Mapping[str, torch.Size]:
        """Return the tensor shapes for each managed parameter."""

        return {name: tensor.shape for name, tensor in self._parameters.items()}

    def parameters(self, as_numpy: bool = False) -> Mapping[str, Tensor | np.ndarray]:
        """Return a copy of the currently stored parameter tensors."""

        if as_numpy:
            return {name: tensor.detach().cpu().numpy().copy() for name, tensor in self._parameters.items()}
        return {name: tensor.clone() for name, tensor in self._parameters.items()}

    def set_parameters(self, updates: Mapping[str, np.ndarray | Tensor]) -> None:
        """Replace the stored tensors with new values."""

        for name, value in updates.items():
            self._set_parameter(name, value, additive=False)

    def adjust_parameters(self, deltas: Mapping[str, np.ndarray | Tensor]) -> None:
        """Increment stored parameters by the provided deltas."""

        for name, value in deltas.items():
            self._set_parameter(name, value, additive=True)

    def set_shape(self, betas: np.ndarray | Tensor) -> None:
        """Overwrite the body shape coefficients (``betas``)."""

        self._set_parameter("betas", betas, additive=False)

    def adjust_shape(self, delta_betas: np.ndarray | Tensor) -> None:
        """Add a delta to the shape coefficients."""

        self._set_parameter("betas", delta_betas, additive=True)

    def set_body_pose(
        self,
        body_pose: np.ndarray | Tensor | None = None,
        global_orient: np.ndarray | Tensor | None = None,
        transl: np.ndarray | Tensor | None = None,
    ) -> None:
        """Update the primary kinematic pose parameters."""

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
        """Add pose offsets to the current parameter values."""

        if body_pose is not None:
            self._set_parameter("body_pose", body_pose, additive=True)
        if global_orient is not None:
            self._set_parameter("global_orient", global_orient, additive=True)
        if transl is not None:
            self._set_parameter("transl", transl, additive=True)

    @torch.no_grad()
    def forward(self, **kwargs: object) -> _SMPLXOutput:
        """Execute a forward pass through the SMPL-X model."""

        return self._model(**self._parameters, **kwargs)

    @torch.no_grad()
    def vertices(self) -> Tensor:
        """Return the generated mesh vertices for the current parameters."""

        output = self.forward(return_verts=True)
        return output.vertices

    @torch.no_grad()
    def joints(self) -> Tensor:
        """Return joint locations for the current parameters."""

        output = self.forward(return_verts=True)
        return output.joints

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

    def _coerce_to_tensor(self, value: np.ndarray | Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
        return tensor
