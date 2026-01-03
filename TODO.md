# TODO

- Decide PDF tiling format (A4/A0/Letter) and implement multi-page pattern layouts.
- Add Panel data model validation/quality gates (distortion thresholds, boundary curvature, seam compatibility).
- Implement boundary regularization stages R3-R6 (turning budget, min feature suppression, seam reconciliation, spline fit) that consume `PanelBudgets` and emit structured issues.
- Expose outline cleanup parameters (outlier threshold, smoothing iterations, simplify tolerance)
  in the pattern export CLI so garment makers can tune output fidelity.
