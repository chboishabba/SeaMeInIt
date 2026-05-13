"""Utilities for working with fitted body meshes."""

from .body_carrier_receipt import (
    BodyCarrierReceipt,
    DEFAULT_BODY_BLOCKED_CONSUMERS,
    Promotion,
    can_consume_receipt,
    load_body_carrier_receipt,
    normalize_promotion,
    with_blocked_consumers,
    with_promotion,
)
from .body_record import load_body_record
from .correspondence_receipt import (
    CorrespondenceReceipt,
    DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS,
    TransformReceipt,
    can_consume_correspondence_receipt,
    is_diagnostic_nn_collapse,
    load_correspondence_receipt,
)
from .repair import repair_body_mesh_for_export, repair_mesh_with_pymeshfix

__all__ = [
    "BodyCarrierReceipt",
    "CorrespondenceReceipt",
    "DEFAULT_BODY_BLOCKED_CONSUMERS",
    "DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS",
    "Promotion",
    "TransformReceipt",
    "can_consume_correspondence_receipt",
    "can_consume_receipt",
    "is_diagnostic_nn_collapse",
    "load_body_carrier_receipt",
    "load_body_record",
    "load_correspondence_receipt",
    "normalize_promotion",
    "repair_body_mesh_for_export",
    "repair_mesh_with_pymeshfix",
    "with_blocked_consumers",
    "with_promotion",
]
