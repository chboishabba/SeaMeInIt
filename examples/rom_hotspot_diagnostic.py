"""Generate a stub hotspot plot from synthetic ROM aggregation results.

Run locally to sanity check diagnostics without touching real assets:

    python examples/rom_hotspot_diagnostic.py

Outputs land in `outputs/rom/` and remain git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smii.rom.aggregation import RomSample, aggregate_fields
from smii.rom.basis import KernelBasis, KernelProjector


def main() -> None:
    basis = KernelBasis.from_arrays(np.eye(8))
    projector = KernelProjector(basis)
    rng = np.random.default_rng(7)
    samples = [
        RomSample(pose_id=f"pose_{idx}", coeffs={"demo": rng.normal(size=8)}) for idx in range(12)
    ]

    aggregation = aggregate_fields(samples, projector, diagnostics_top_k=5)
    diagnostics = aggregation.diagnostics["demo"]

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional diagnostic dependency
        print("matplotlib is required for plotting hotspots. Install it and re-run the script.")
        return

    indices = [hotspot.index for hotspot in diagnostics.vertex_hotspots]
    variances = [hotspot.variance for hotspot in diagnostics.vertex_hotspots]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(indices, variances)
    ax.set_xlabel("Vertex index")
    ax.set_ylabel("Variance (normalized units)")
    ax.set_title("ROM Hotspot Diagnostic (demo)")
    ax.grid(True, alpha=0.3)

    output = Path("outputs/rom/hotspots_demo.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved demo hotspot plot to {output}")


if __name__ == "__main__":
    main()
