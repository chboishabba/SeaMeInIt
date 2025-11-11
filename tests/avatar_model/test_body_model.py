"""Tests for the BodyModel wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from avatar_model.body_model import BodyModel, BodyModelConfig


def test_body_model_unknown_provider(tmp_path: Path) -> None:
    """Helpful error when requesting an unregistered provider."""

    config = BodyModelConfig(model_path=tmp_path, model_type="unknown")

    with pytest.raises(ValueError, match="Unknown body model provider"):
        BodyModel(config=config)


def test_body_model_missing_asset_directory(tmp_path: Path) -> None:
    """A helpful error is raised when the asset root is absent."""

    config = BodyModelConfig(model_path=tmp_path / "missing-assets")

    with pytest.raises(FileNotFoundError, match="SMPL-X assets are required"):
        BodyModel(config=config)


def test_body_model_missing_gender_file(tmp_path: Path) -> None:
    """Surface a clear error when the gendered asset archive is missing."""

    assets_root = tmp_path / "assets"
    model_dir = assets_root / "smplx"
    model_dir.mkdir(parents=True)

    config = BodyModelConfig(model_path=assets_root)

    with pytest.raises(FileNotFoundError, match="Expected SMPL-X asset"):
        BodyModel(config=config)


def test_smplerx_provider_not_implemented(tmp_path: Path) -> None:
    """The placeholder SMPLer-X provider advertises its incomplete state."""

    config = BodyModelConfig(model_path=tmp_path, model_type="smplerx")

    with pytest.raises(NotImplementedError, match="SmplerxProvider"):
        BodyModel(config=config)
