"""Receipt-level seam derivation orchestration.

This module does not solve production seam topology. It records whether an
existing body/ROM/fabric/seam/panel/correction/manufacturing chain is a
promotable adaptive body atlas serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from .cut_topology_receipt import CutTopologyReceipt, can_consume_cut_topology_receipt
from .manufacturing_receipt import ManufacturingReceipt, can_consume_manufacturing_receipt
from .metric_correction_receipt import (
    MetricCorrectionReceipt,
    can_consume_metric_correction_receipt,
)
from .panel_unwrap_receipt import PanelUnwrapReceipt, can_consume_panel_unwrap_receipt
from .seam_cost_receipt import SeamCostReceipt, can_consume_seam_cost_receipt
from .solver_promotion_receipt import SolverPromotionReceipt, can_consume_solver_promotion_receipt

Promotion = Literal[-1, 0, 1]

SCHEMA_VERSION = "smii.finished_seam_receipt.v1"
CLAIM_BOUNDARY = "finished_pattern_is_serialization_of_body_rom_fabric_atlas_not_geometry_truth"
CLAIM_BOUNDARY_FLAGS = {
    "export_is_geometry_truth": False,
    "claims_global_optimum": False,
    "claims_isometry": False,
    "claims_true_inverse": False,
    "claims_manufacturing_authority_without_gate": False,
}
DEFAULT_BLOCKED_CONSUMERS = ("manufacturing", "pattern_export")

__all__ = [
    "CLAIM_BOUNDARY",
    "CLAIM_BOUNDARY_FLAGS",
    "DEFAULT_BLOCKED_CONSUMERS",
    "FinishedSeamReceipt",
    "Promotion",
    "SCHEMA_VERSION",
    "can_consume_finished_seam_receipt",
    "derive_finished_seams",
    "load_finished_seam_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    if isinstance(value, bool):
        raise ValueError("FinishedSeamReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("FinishedSeamReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("FinishedSeamReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"FinishedSeamReceipt is missing required field '{key}'.")


def _str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"FinishedSeamReceipt field '{key}' must be non-empty.")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        return _str_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return _str_value(value, key)


def _non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"FinishedSeamReceipt field '{key}' must be non-negative.")
    return coerced


def _non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        return _non_negative_int_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be a list.")
    return [str(item) for item in value]


def _string_mapping_value(value: object, key: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be an object.")
    return {
        str(name): _str_value(receipt_hash, f"{key}.{name}") for name, receipt_hash in value.items()
    }


def _claim_boundary_label(value: object) -> str:
    if isinstance(value, Mapping):
        for key, expected in CLAIM_BOUNDARY_FLAGS.items():
            if value.get(key) is not expected:
                raise ValueError(
                    "FinishedSeamReceipt claim_boundary must keep every overclaim flag false."
                )
        return CLAIM_BOUNDARY
    return _str_value(value, "claim_boundary")


def _int_mapping_value(value: object, key: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FinishedSeamReceipt field '{key}' must be an object.")
    return {
        str(name): _non_negative_int_value(count, f"{key}.{name}") for name, count in value.items()
    }


def _promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        return normalize_promotion(payload["promotion"])
    except KeyError as exc:
        raise _missing("promotion") from exc


def _blocked_consumers_for_promotion(
    promotion: Promotion,
    blocked_consumers: list[str],
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class FinishedSeamReceipt:
    """Final seam/pattern atlas receipt over existing stage receipts."""

    body_receipt_hash: str
    rom_receipt_hash: str
    fabric_receipt_hash: str
    basis_receipt_hash: str
    stage_receipt_hashes: dict[str, str]
    selected_seam_count: int
    panel_count: int
    shaping_operator_counts: dict[str, int]
    allowance_policy: str
    atlas_boundary: str
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    manufacturing_exports_hash: str | None = None
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "body_receipt_hash",
            "rom_receipt_hash",
            "fabric_receipt_hash",
            "basis_receipt_hash",
            "allowance_policy",
            "atlas_boundary",
            "claim_boundary",
        ):
            object.__setattr__(self, key, _str_value(getattr(self, key), key))
        object.__setattr__(
            self,
            "stage_receipt_hashes",
            _string_mapping_value(self.stage_receipt_hashes, "stage_receipt_hashes"),
        )
        object.__setattr__(
            self,
            "selected_seam_count",
            _non_negative_int_value(self.selected_seam_count, "selected_seam_count"),
        )
        object.__setattr__(
            self,
            "panel_count",
            _non_negative_int_value(self.panel_count, "panel_count"),
        )
        object.__setattr__(
            self,
            "shaping_operator_counts",
            _int_mapping_value(self.shaping_operator_counts, "shaping_operator_counts"),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _string_list_value(self.blocked_consumers, "blocked_consumers")
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        object.__setattr__(self, "blocker_log", _string_list_value(self.blocker_log, "blocker_log"))
        if self.manufacturing_exports_hash is not None:
            object.__setattr__(
                self,
                "manufacturing_exports_hash",
                _str_value(self.manufacturing_exports_hash, "manufacturing_exports_hash"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FinishedSeamReceipt":
        return cls(
            body_receipt_hash=_required_str(payload, "body_receipt_hash"),
            rom_receipt_hash=_required_str(payload, "rom_receipt_hash"),
            fabric_receipt_hash=_required_str(payload, "fabric_receipt_hash"),
            basis_receipt_hash=_required_str(payload, "basis_receipt_hash"),
            stage_receipt_hashes=_string_mapping_value(
                payload.get("stage_receipt_hashes", {}),
                "stage_receipt_hashes",
            ),
            selected_seam_count=_non_negative_int(payload, "selected_seam_count"),
            panel_count=_non_negative_int(payload, "panel_count"),
            shaping_operator_counts=_int_mapping_value(
                payload.get("shaping_operator_counts", {}),
                "shaping_operator_counts",
            ),
            allowance_policy=_required_str(payload, "allowance_policy"),
            atlas_boundary=_required_str(payload, "atlas_boundary"),
            promotion=_promotion(payload),
            blocked_consumers=_string_list_value(
                payload.get("blocked_consumers", []),
                "blocked_consumers",
            ),
            blocker_log=_string_list_value(payload.get("blocker_log", []), "blocker_log"),
            manufacturing_exports_hash=_optional_str(payload, "manufacturing_exports_hash"),
            claim_boundary=_claim_boundary_label(payload.get("claim_boundary", CLAIM_BOUNDARY)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FinishedSeamReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("FinishedSeamReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "body_receipt_hash": self.body_receipt_hash,
            "rom_receipt_hash": self.rom_receipt_hash,
            "fabric_receipt_hash": self.fabric_receipt_hash,
            "basis_receipt_hash": self.basis_receipt_hash,
            "input_hashes": {
                "body": self.body_receipt_hash,
                "rom": self.rom_receipt_hash,
                "fabric": self.fabric_receipt_hash,
                "basis": self.basis_receipt_hash,
            },
            "body_gate": {
                "receipt_hash": self.body_receipt_hash,
                "promotion": _promotion_name(self.promotion),
            },
            "rom_gate": {
                "receipt_hash": self.rom_receipt_hash,
                "promotion": _promotion_name(self.promotion),
            },
            "fabric_gate": {
                "receipt_hash": self.fabric_receipt_hash,
                "promotion": _promotion_name(self.promotion),
            },
            "basis_gate": {
                "receipt_hash": self.basis_receipt_hash,
                "promotion": _promotion_name(self.promotion),
            },
            "stage_receipt_hashes": dict(self.stage_receipt_hashes),
            "seam_atlas": {
                "selected_seam_count": int(self.selected_seam_count),
                "stage_hash": self.stage_receipt_hashes.get("solver"),
            },
            "panel_atlas": {
                "panel_count": int(self.panel_count),
                "stage_hash": self.stage_receipt_hashes.get("cut_topology"),
            },
            "flattening": {
                "panel_unwrap_hash": self.stage_receipt_hashes.get("panel_unwrap"),
            },
            "flattening_metrics": {
                "panel_count": int(self.panel_count),
            },
            "correction_ops": [
                {"type": name, "count": int(count)}
                for name, count in sorted(self.shaping_operator_counts.items())
                if int(count) > 0
            ],
            "allowance_fields": {
                "policy": self.allowance_policy,
                "stage_hash": self.stage_receipt_hashes.get("manufacturing"),
            },
            "manufacturing_exports": {
                "hash": self.manufacturing_exports_hash,
                "stage_hash": self.stage_receipt_hashes.get("manufacturing"),
            },
            "selected_seam_count": int(self.selected_seam_count),
            "panel_count": int(self.panel_count),
            "shaping_operator_counts": dict(self.shaping_operator_counts),
            "allowance_policy": self.allowance_policy,
            "atlas_boundary": self.atlas_boundary,
            "promotion": int(self.promotion),
            "promotion_state": _promotion_name(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": dict(CLAIM_BOUNDARY_FLAGS),
            "claim_boundary_label": self.claim_boundary,
        }
        if self.manufacturing_exports_hash is not None:
            payload["manufacturing_exports_hash"] = self.manufacturing_exports_hash
        return payload

    def to_json(self, path: str | Path) -> Path:
        receipt_path = Path(path)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt_path


def load_finished_seam_receipt(path: str | Path) -> FinishedSeamReceipt:
    return FinishedSeamReceipt.from_json(path)


def with_promotion(receipt: FinishedSeamReceipt, promotion: object) -> FinishedSeamReceipt:
    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion,
            receipt.blocked_consumers,
        ),
    )


def can_consume_finished_seam_receipt(
    receipt: FinishedSeamReceipt,
    consumer: str | None = None,
) -> bool:
    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers


def _promotion_name(promotion: Promotion) -> str:
    if promotion == 1:
        return "promoted"
    if promotion == 0:
        return "diagnostic_only"
    return "rejected"


def derive_finished_seams(
    *,
    body_receipt_hash: str,
    rom_receipt_hash: str,
    fabric_receipt_hash: str,
    basis_receipt_hash: str,
    seam_cost_receipt: SeamCostReceipt,
    solver_receipt: SolverPromotionReceipt,
    cut_topology_receipt: CutTopologyReceipt,
    panel_unwrap_receipt: PanelUnwrapReceipt,
    metric_correction_receipt: MetricCorrectionReceipt | None,
    manufacturing_receipt: ManufacturingReceipt | None,
    allowance_policy: str = "variable_boundary_field",
    manufacturing_exports_hash: str | None = None,
) -> FinishedSeamReceipt:
    """Compose stage receipts into a finished seam/pattern atlas receipt."""

    stage_hashes = {
        "seam_cost": seam_cost_receipt.costs_hash,
        "solver": solver_receipt.seam_hash,
        "cut_topology": cut_topology_receipt.seam_edges_hash,
        "panel_unwrap": panel_unwrap_receipt.uv_hash,
    }
    if metric_correction_receipt is not None and metric_correction_receipt.correction_payload_hash:
        stage_hashes["metric_correction"] = metric_correction_receipt.correction_payload_hash
    if manufacturing_receipt is not None:
        stage_hashes["manufacturing"] = manufacturing_receipt.cutting_artifacts_hash

    blocker_log: list[str] = []
    if not can_consume_seam_cost_receipt(seam_cost_receipt, "solver_promotion"):
        blocker_log.append("seam_cost_receipt_not_promoted")
    if not can_consume_solver_promotion_receipt(solver_receipt, "panel_unwrap"):
        blocker_log.append("solver_receipt_not_promoted")
    if not can_consume_cut_topology_receipt(cut_topology_receipt, "panel_unwrap"):
        blocker_log.append("cut_topology_receipt_not_promoted")
    if metric_correction_receipt is not None and not can_consume_metric_correction_receipt(
        metric_correction_receipt,
        "panel_unwrap",
    ):
        blocker_log.append("metric_correction_receipt_not_promoted")
    if not can_consume_panel_unwrap_receipt(panel_unwrap_receipt, "manufacturing"):
        blocker_log.append("panel_unwrap_receipt_not_promoted")
    if manufacturing_receipt is None:
        blocker_log.append("manufacturing_receipt_missing")
    elif not can_consume_manufacturing_receipt(manufacturing_receipt):
        blocker_log.append("manufacturing_receipt_not_promoted")

    shaping_counts = {
        "dart": cut_topology_receipt.typed_dart_count,
        "gusset": cut_topology_receipt.typed_gusset_count,
        "relief_cut": cut_topology_receipt.typed_relief_cut_count,
        "ease": cut_topology_receipt.typed_ease_count,
        "stretch_zone": cut_topology_receipt.typed_stretch_zone_count,
    }
    if metric_correction_receipt is not None:
        for correction in metric_correction_receipt.corrections:
            shaping_counts[correction.correction_type] = (
                shaping_counts.get(correction.correction_type, 0) + 1
            )

    promotion: Promotion = 1 if not blocker_log else 0
    return FinishedSeamReceipt(
        body_receipt_hash=body_receipt_hash,
        rom_receipt_hash=rom_receipt_hash,
        fabric_receipt_hash=fabric_receipt_hash,
        basis_receipt_hash=basis_receipt_hash,
        stage_receipt_hashes=stage_hashes,
        selected_seam_count=solver_receipt.seam_edge_count,
        panel_count=panel_unwrap_receipt.panel_count,
        shaping_operator_counts=shaping_counts,
        allowance_policy=allowance_policy,
        atlas_boundary="adaptive_body_rom_fabric_seam_atlas",
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blocker_log,
        manufacturing_exports_hash=manufacturing_exports_hash,
    )
