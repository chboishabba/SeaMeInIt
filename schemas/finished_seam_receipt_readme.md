# Finished Seam Receipt Schema

`finished_seam_receipt.schema.json` validates the final SeaMeInIt atlas
serialization receipt. The receipt wraps promoted body, ROM, fabric, basis,
seam, panel, metric-correction, and manufacturing evidence.

The schema intentionally requires claim-boundary flags proving the exported
pattern is not treated as geometry truth, a global optimum, an isometry, a true
inverse, or an ungated manufacturing authority.
