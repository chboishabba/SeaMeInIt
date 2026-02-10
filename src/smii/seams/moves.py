"""Local seam edit moves for PDA-based seam optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

Edge = tuple[int, int]

__all__ = [
    "MoveResult",
    "reroute_edge",
    "shorten_seam",
    "split_panel",
    "merge_panels",
    "switch_fabric",
    "rotate_grain",
]


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Result of applying a move to a set of edges."""

    edges: tuple[Edge, ...]
    changed_edges: tuple[Edge, ...]
    description: str
    fabric_id: str | None = None
    grain_rotation_deg: float | None = None


def _sorted_edge(edge: Edge) -> Edge:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _edge_cost(edge: Edge, costs: Mapping[Edge, float]) -> float:
    return float(costs.get(_sorted_edge(edge), 0.0))


def reroute_edge(
    selected_edges: Sequence[Edge],
    candidate_edges: Sequence[Edge],
    costs: Mapping[Edge, float],
) -> MoveResult:
    """Replace the highest-cost edge with the lowest-cost available alternative sharing a vertex."""

    normalized_selected = [_sorted_edge(edge) for edge in selected_edges]
    normalized_candidates = {_sorted_edge(edge) for edge in candidate_edges}
    if not normalized_selected or not normalized_candidates:
        return MoveResult(edges=tuple(normalized_selected), changed_edges=tuple(), description="no-op")

    worst_edge = max(normalized_selected, key=lambda edge: _edge_cost(edge, costs))
    worst_vertices = set(worst_edge)
    alternatives = [
        edge
        for edge in normalized_candidates
        if edge not in normalized_selected and (edge[0] in worst_vertices or edge[1] in worst_vertices)
    ]
    if not alternatives:
        return MoveResult(edges=tuple(normalized_selected), changed_edges=tuple(), description="no alt edge")

    best_alt = min(alternatives, key=lambda edge: _edge_cost(edge, costs))
    if _edge_cost(best_alt, costs) >= _edge_cost(worst_edge, costs):
        return MoveResult(edges=tuple(normalized_selected), changed_edges=tuple(), description="no better edge")

    updated = tuple(sorted([edge for edge in normalized_selected if edge != worst_edge] + [best_alt]))
    changed = tuple(sorted({worst_edge, best_alt}))
    return MoveResult(edges=updated, changed_edges=changed, description="reroute_edge")


def shorten_seam(selected_edges: Sequence[Edge], costs: Mapping[Edge, float]) -> MoveResult:
    """Remove the highest-cost edge to shorten a seam."""

    normalized_selected = [_sorted_edge(edge) for edge in selected_edges]
    if len(normalized_selected) <= 1:
        return MoveResult(edges=tuple(normalized_selected), changed_edges=tuple(), description="no-op")

    worst_edge = max(normalized_selected, key=lambda edge: _edge_cost(edge, costs))
    updated = tuple(edge for edge in normalized_selected if edge != worst_edge)
    return MoveResult(edges=tuple(sorted(updated)), changed_edges=(worst_edge,), description="shorten_seam")


def split_panel(
    selected_edges: Sequence[Edge],
    candidate_edges: Sequence[Edge],
    costs: Mapping[Edge, float],
) -> MoveResult:
    """Add the cheapest missing edge to increase separability."""

    normalized_selected = {_sorted_edge(edge) for edge in selected_edges}
    missing = [_sorted_edge(edge) for edge in candidate_edges if _sorted_edge(edge) not in normalized_selected]
    if not missing:
        return MoveResult(edges=tuple(sorted(normalized_selected)), changed_edges=tuple(), description="no missing edges")

    best_new = min(missing, key=lambda edge: _edge_cost(edge, costs))
    updated = tuple(sorted(normalized_selected | {best_new}))
    return MoveResult(edges=updated, changed_edges=(best_new,), description="split_panel")


def merge_panels(panel_a_edges: Sequence[Edge], panel_b_edges: Sequence[Edge]) -> MoveResult:
    """Merge two edge sets."""

    merged = {_sorted_edge(edge) for edge in panel_a_edges} | {_sorted_edge(edge) for edge in panel_b_edges}
    return MoveResult(edges=tuple(sorted(merged)), changed_edges=tuple(sorted(merged)), description="merge_panels")


def switch_fabric(selected_edges: Sequence[Edge], fabrics: Sequence[str], current: str | None) -> MoveResult:
    """Switch panel fabric assignment to the next available option."""

    normalized_selected = tuple(_sorted_edge(edge) for edge in selected_edges)
    if not fabrics:
        return MoveResult(edges=normalized_selected, changed_edges=tuple(), description="no fabrics")
    if current in fabrics:
        current_index = fabrics.index(current)
        target = fabrics[(current_index + 1) % len(fabrics)]
    else:
        target = fabrics[0]
    return MoveResult(edges=normalized_selected, changed_edges=tuple(), description="switch_fabric", fabric_id=target)


def rotate_grain(selected_edges: Sequence[Edge], current_rotation: float, step_deg: float, *, limit: float | None = None) -> MoveResult:
    """Rotate panel grain assignment by +/- step within an optional limit."""

    normalized_selected = tuple(_sorted_edge(edge) for edge in selected_edges)
    target = current_rotation + step_deg
    if limit is not None:
        target = float(max(-limit, min(limit, target)))
    return MoveResult(
        edges=normalized_selected,
        changed_edges=tuple(),
        description="rotate_grain",
        grain_rotation_deg=target,
    )
