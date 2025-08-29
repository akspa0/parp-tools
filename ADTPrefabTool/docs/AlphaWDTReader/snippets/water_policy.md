# Water Policy (MH2O-only)

- Always convert from `MCLQ` to `MH2O` during Alpha→3.x.
- Do not emit legacy `MCLQ` in outputs.
- Zero legacy `MCLQ` offsets; ensure writer never mixes formats.
- Validate post-write: scan headers to assert no `MCLQ` remains.
- Keep config simple: no toggle; MH2O-only is the default and only mode.

## WLW-based Rescue Policy (optional)

- **Toggle**: `--rescue-liquids-from-wlw` (default off). When enabled, WLW source geometry may be rasterized to fill missing/broken MH2O regions.
- **Threshold**: `--wlw-rescue-tri-threshold <N>` to ignore tiny/degenerate faces.
- **Behavior**:
  - Additive-only. Do not overwrite existing valid MH2O layers.
  - WLW is authoritative in 0.5.3; WLQ/WLM are optional in 0.6.0–3.0 and used only if validated.
  - MH2O-only output remains enforced (no MCLQ).
- **Validation**: Use real test data to compare pre/post masks and inspect in noggit3.

## Legacy MCLQ (very optional)

- **Output mode**: `--output-adt-version pre3x` switches to MCLQ-only output.
- **Mutual exclusivity**: Do not emit MH2O when targeting pre-3.x.
- **Fill strategies**:
  - `--mclq-fill preserve` (default): copy original MCLQ verbatim if present; otherwise error with guidance.
  - `--mclq-fill from-wlw`: rasterize WLW to MCLQ height/flags using `--wlw-rescue-tri-threshold <N>` as needed.
- **Integrity**: No silent fallback between strategies; user must opt in to regeneration.
- **Testing**: Use real tiles; for `preserve` require byte-accurate equality; for `from-wlw` validate in viewer/tools.
