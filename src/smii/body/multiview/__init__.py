"""Multi-view camera models, synthetic rigs, and shape identifiability referee."""

from __future__ import annotations

from .camera import (
    CameraModel,
    FloatArray,
)
from .fit_referee import (
    RecoveryResult,
    run_shape_recovery_test,
)
from .identifiability import (
    IdentifiabilitySpectrum,
    analyze_identifiability,
    compute_shape_jacobian,
)
from .synthetic_rig import (
    SyntheticRigConfig,
    build_circular_rig,
    build_eight_view_ring_rig,
    build_four_view_orthogonal_rig,
    build_frontal_rig,
    project_multiview,
)

__all__ = [
    "CameraModel",
    "FloatArray",
    "IdentifiabilitySpectrum",
    "RecoveryResult",
    "SyntheticRigConfig",
    "analyze_identifiability",
    "build_circular_rig",
    "build_eight_view_ring_rig",
    "build_four_view_orthogonal_rig",
    "build_frontal_rig",
    "compute_shape_jacobian",
    "project_multiview",
    "run_shape_recovery_test",
]
