"""Pipeline utilities for fitting SMPL-X models."""

from .fit_from_measurements import FitResult, fit_smplx_from_measurements
from .fit_from_scan import (
    ICPSettings,
    RegistrationResult,
    create_parametric_mesh,
    fit_scan_to_smplx,
)

__all__ = [
    "FitResult",
    "fit_smplx_from_measurements",
    "ICPSettings",
    "RegistrationResult",
    "create_parametric_mesh",
    "fit_scan_to_smplx",
]
