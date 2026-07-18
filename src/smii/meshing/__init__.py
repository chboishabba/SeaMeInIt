"""Utilities for working with fitted body meshes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, TypeAlias

from . import body_carrier_receipt as _legacy_body_receipt_module
from .body_carrier_receipt import (
    BodyCarrierReceipt as LegacyBodyCarrierReceipt,
    DEFAULT_BODY_BLOCKED_CONSUMERS,
    Promotion,
    normalize_promotion,
    with_blocked_consumers,
    with_promotion,
)
from .body_carrier_compat import BodyCarrierReceipt
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

BodyCarrierReceiptLike: TypeAlias = LegacyBodyCarrierReceipt | BodyCarrierReceiptV2
_legacy_can_consume_receipt = _legacy_body_receipt_module.can_consume_receipt


def load_body_carrier_receipt(path: str | Path) -> BodyCarrierReceiptLike:
    """Load v2 when declared, otherwise validate the legacy v1 contract."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("BodyCarrierReceipt JSON must contain an object.")
    if payload.get("schema_version") == "smii.body_carrier_receipt.v2":
        return load_body_carrier_receipt_v2(target)
    return LegacyBodyCarrierReceipt.from_mapping(payload)


def can_consume_receipt(
    receipt: BodyCarrierReceiptLike,
    consumer: str | None = None,
) -> bool:
    """Apply the receipt-version-specific authorization boundary."""

    if isinstance(receipt, BodyCarrierReceiptV2):
        return can_consume_body_receipt_v2(receipt, consumer)
    return _legacy_can_consume_receipt(receipt, consumer)


# Direct imports of smii.meshing.body_carrier_receipt are common in the
# orchestration code. Patch the shared module functions so those consumers also
# prefer v2 without changing their import sites in this compatibility rung.
_legacy_body_receipt_module.load_body_carrier_receipt = load_body_carrier_receipt
_legacy_body_receipt_module.can_consume_receipt = can_consume_receipt

__all__ = [
    "BodyCarrierReceipt",
    "BodyCarrierReceiptLike",
    "BodyCarrierReceiptV2",
    "CorrespondenceReceipt",
    "DEFAULT_BODY_BLOCKED_CONSUMERS",
    "DEFAULT_BODY_BLOCKED_CONSUMERS_V2",
    "DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS",
    "LegacyBodyCarrierReceipt",
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
