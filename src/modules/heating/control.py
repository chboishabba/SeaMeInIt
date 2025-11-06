"""Control logic for distributing power to heating zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from .heating_mesh import HeatingMeshPlan


@dataclass(frozen=True)
class ZoneConfig:
    """Safety and power envelope for a thermal zone."""

    zone_id: str
    max_watt_density: float
    duty_cycle_limit: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("Zone identifier must not be empty.")
        if self.max_watt_density < 0.0:
            raise ValueError("Maximum watt density cannot be negative.")
        if not 0.0 <= self.duty_cycle_limit <= 1.0:
            raise ValueError("Duty cycle limits must lie in the [0, 1] range.")


@dataclass(frozen=True)
class ZoneBudget:
    """Computed power envelope for a specific zone."""

    zone_id: str
    installed_power: float
    max_safe_power: float
    recommended_power: float
    recommended_duty_cycle: float
    zone_area: float

    def to_payload(self) -> Dict[str, float | str]:
        return {
            "zone_id": self.zone_id,
            "installed_power": self.installed_power,
            "max_safe_power": self.max_safe_power,
            "recommended_power": self.recommended_power,
            "recommended_duty_cycle": self.recommended_duty_cycle,
            "zone_area": self.zone_area,
        }


@dataclass(slots=True)
class HeatingController:
    """Derive zone power budgets using heating placement plans."""

    zone_configs: Mapping[str, ZoneConfig]
    global_safety_margin: float = 0.9

    @classmethod
    def with_default_limits(
        cls, zone_ids: Iterable[str], *, max_watt_density: float = 220.0, duty_cycle: float = 0.85
    ) -> "HeatingController":
        configs = {
            zone_id: ZoneConfig(zone_id=zone_id, max_watt_density=max_watt_density, duty_cycle_limit=duty_cycle)
            for zone_id in zone_ids
        }
        return cls(zone_configs=configs)

    def derive_zone_budgets(self, plan: HeatingMeshPlan) -> Dict[str, ZoneBudget]:
        budgets: Dict[str, ZoneBudget] = {}
        for zone_id, config in self.zone_configs.items():
            area = plan.zone_area(zone_id)
            installed_power = plan.zone_wattage(zone_id)
            max_safe_power = area * config.max_watt_density
            safety_limited = max_safe_power * self.global_safety_margin
            recommended_power = min(installed_power, safety_limited)
            if max_safe_power <= 0.0:
                recommended_power = 0.0
            duty_cycle = 0.0
            if installed_power > 0.0:
                duty_cycle = recommended_power / installed_power
                duty_cycle = min(duty_cycle, config.duty_cycle_limit)
                duty_cycle = max(duty_cycle, 0.0)

            budgets[zone_id] = ZoneBudget(
                zone_id=zone_id,
                installed_power=installed_power,
                max_safe_power=max_safe_power,
                recommended_power=recommended_power,
                recommended_duty_cycle=duty_cycle,
                zone_area=area,
            )

        return budgets

    def describe(self) -> Dict[str, object]:
        return {
            "global_safety_margin": self.global_safety_margin,
            "zones": [
                {
                    "zone_id": config.zone_id,
                    "max_watt_density": config.max_watt_density,
                    "duty_cycle_limit": config.duty_cycle_limit,
                    "label": config.label,
                }
                for config in self.zone_configs.values()
            ],
        }


__all__ = ["HeatingController", "ZoneBudget", "ZoneConfig"]
