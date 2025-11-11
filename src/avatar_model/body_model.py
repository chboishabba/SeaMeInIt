"""High level wrapper around avatar body model providers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
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
    """Directory containing the SMPL-compatible model files."""

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
    """User-facing facade that delegates to registered model providers."""

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
