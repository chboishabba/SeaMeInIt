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
from .body_carrier_receipt_v2 import BodyCarrierReceiptV2
from .body_carrier_v2_builder import (
    build_body_carrier_receipt_v2,
    can_consume_body_receipt_v2,
)
from .body_carrier_v2_io import load_body_carrier_receipt_v2
from .body_carrier_v2_policy import (
    DEFAULT_BLOCKED_CONSUMERS as DEFAULT_BODY_BLOCKED_CONSUMERS_V2,
    decide_body_authorization,
)
from .body_record import load_body_record
from .correspondence_receipt import (
    CorrespondenceReceipt,
    DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS,
    TransformReceipt,
    TRANSFER_MODES,
    can_consume_correspondence_receipt,
    is_diagnostic_nn_collapse,
    load_correspondence_receipt,
)
from .repair import repair_body_mesh_for_export, repair_mesh_with_pymeshfix

__all__ = [
    "BodyCarrierReceipt",
    "BodyCarrierReceiptV2",
    "CorrespondenceReceipt",
    "DEFAULT_BODY_BLOCKED_CONSUMERS",
    "DEFAULT_BODY_BLOCKED_CONSUMERS_V2",
    "DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS",
    "Promotion",
    "TRANSFER_MODES",
    "TransformReceipt",
    "build_body_carrier_receipt_v2",
    "can_consume_body_receipt_v2",
    "can_consume_correspondence_receipt",
    "can_consume_receipt",
    "decide_body_authorization",
    "is_diagnostic_nn_collapse",
    "load_body_carrier_receipt",
    "load_body_carrier_receipt_v2",
    "load_body_record",
    "load_correspondence_receipt",
    "normalize_promotion",
    "repair_body_mesh_for_export",
    "repair_mesh_with_pymeshfix",
    "with_blocked_consumers",
    "with_promotion",
]
