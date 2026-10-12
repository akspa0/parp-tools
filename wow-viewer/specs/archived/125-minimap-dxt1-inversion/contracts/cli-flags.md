# Contract: `synthetic-minimap` CLI Surface

**Tool**: `WowViewer.Tool.Harvest` | **File**: `Program.cs`

## New flags

| Flag | Requires | Effect |
|------|----------|--------|
| `--dxt1-parity` | — | Emit a `*_dxt1.png` parity companion per tile alongside the pristine render (FR-015). |
| `--lighting-baseline` | `--authored-reference` | Survey authored tiles for a shared lighting baseline and account for it when scoring (FR-016). |
| `--encoding-survey` | — | Report the per-build/map distribution of encodings (FR-013). |

## Modified behaviour

- `--score --authored-reference` now reports parity-adjusted agreement alongside unadjusted, and
  states which encoding was applied (FR-003). When `--dxt1-parity` is set, the parity companion is
  used directly (no comparison-time encode).
- Degenerate (single flat colour) tiles are excluded from aggregate scores and reported as excluded
  (FR-004).
- Every comparison report and corpus row records its parity status, including "none" (FR-005).
- Era-gating: an unrecognised build is flagged, never silently defaulted (FR-006).

## Outputs

- `*_dxt1.png` — parity companion per tile (with `--dxt1-parity`).
- `authored-comparison.csv` — parity-adjusted + unadjusted agreement (with `--score`).
- Console report — lighting-baseline result (with `--lighting-baseline`).
- Console report — encoding distribution (with `--encoding-survey`).
