"""Multi-view camera models, synthetic rigs, and shape identifiability referee."""

from __future__ import annotations

from .angular_reservoir import (
    AngularReservoir,
    AngularReservoirConfig,
    StreamObservation,
)
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
from .video_orbit import (
    LandmarkExtractor,
    MediaPipeLandmarkExtractor,
    StopReason,
    StreamConfig,
    StreamResult,
    VideoOrbitStreamer,
)
from .yaw_tracker import (
    CircularYawTracker,
    YawMeasurement,
    YawState,
    wrap_angle_pi,
)

__all__ = [
    "AngularReservoir",
    "AngularReservoirConfig",
    "CameraModel",
    "CircularYawTracker",
    "FloatArray",
    "IdentifiabilitySpectrum",
    "LandmarkExtractor",
    "MediaPipeLandmarkExtractor",
    "RecoveryResult",
    "StopReason",
    "StreamConfig",
    "StreamObservation",
    "StreamResult",
    "SyntheticRigConfig",
    "VideoOrbitStreamer",
    "YawMeasurement",
    "YawState",
    "analyze_identifiability",
    "build_circular_rig",
    "build_eight_view_ring_rig",
    "build_four_view_orthogonal_rig",
    "build_frontal_rig",
    "compute_shape_jacobian",
    "project_multiview",
    "run_shape_recovery_test",
    "wrap_angle_pi",
]
