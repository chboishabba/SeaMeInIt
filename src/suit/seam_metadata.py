"""Helpers for authoring seam partner metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SeamPartner:
    """Explicit seam pairing metadata for a panel boundary edge."""

    edge: tuple[int, int]
    partner_panel: str
    partner_edge: tuple[int, int]
    role: str | None = None
    zone: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SeamPartner":
        edge = _parse_edge(payload.get("edge"), "edge")
        partner_edge = _parse_edge(payload.get("partner_edge"), "partner_edge")
        partner_panel = payload.get("partner_panel")
        if partner_panel is None:
            raise KeyError("seam_partners entries require partner_panel.")
        role = payload.get("role")
        zone = payload.get("zone")
        return cls(
            edge=edge,
            partner_panel=str(partner_panel),
            partner_edge=partner_edge,
            role=str(role) if role is not None else None,
            zone=str(zone) if zone is not None else None,
        )

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "edge": list(self.edge),
            "partner_panel": self.partner_panel,
            "partner_edge": list(self.partner_edge),
        }
        if self.role is not None:
            data["role"] = self.role
        if self.zone is not None:
            data["zone"] = self.zone
        return data


def normalize_seam_metadata(
    seams: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Normalize seam metadata and derive seam-aware split ranges."""

    if seams is None:
        return None

    normalized: dict[str, dict[str, Any]] = {}
    for panel_name, metadata in seams.items():
        entry = dict(metadata)
        partners_raw = entry.get("seam_partners")
        if partners_raw is not None:
            if not isinstance(partners_raw, list):
                raise TypeError("seam_partners must be a list of entries.")
            partners = [SeamPartner.from_mapping(item) for item in partners_raw]
            entry["seam_partners"] = [partner.to_mapping() for partner in partners]
            if "seam_avoid_ranges" not in entry:
                entry["seam_avoid_ranges"] = [partner.edge for partner in partners]
            if "seam_partner" not in entry and partners:
                primary = next(
                    (partner for partner in partners if partner.role == "primary"),
                    None,
                )
                chosen = primary if primary is not None else partners[0]
                entry["seam_partner"] = chosen.partner_panel
        normalized[str(panel_name)] = entry

    return normalized


def _parse_edge(value: object, label: str) -> tuple[int, int]:
    if value is None:
        raise KeyError(f"seam_partners entries require {label}.")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-item list or tuple.")
    return (int(value[0]), int(value[1]))


__all__ = ["SeamPartner", "normalize_seam_metadata"]
