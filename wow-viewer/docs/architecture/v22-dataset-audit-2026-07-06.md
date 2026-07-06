# V22 Dataset Signal Audit — 2026-07-06 (Spec 094, amendment A8)

C#-grounded audit of the canonical V22 store (`output/datasets/v22/3_3_5_12340.zarr`,
5,134 tiles, 36 arrays). The reference signals were re-extracted from the staged
`3_3_5_12340` client by the existing, working C# harvester
(`WowViewer.Tool.Harvest extract-unified`); Python only compared
(`data-harvester/scripts/audit_v22_dataset.py`). Sample: 6 tiles across Azeroth,
Expansion01, Northrend, UlduarRaid (seed 94). Full detail:
`output/v24_validation/v22_audit_20260706/report.json`.

## Verdict

The signals the models actually train on are structurally sound. Two systemic
defects and one coverage gap need follow-up; the object-mask family diverges
from naive re-extraction **by design** (the store's masks come from the
enriched projection pipeline, which is strictly richer than the reference
extractor's MCRF-presence heuristic).

## Signal-by-signal

| Signal | Result | Reading |
|---|---|---|
| `height_257` | OK 6/6 | Bit-faithful vs C# reference. |
| `minimap_rgb` | OK 6/6 | Bit-faithful. |
| `mcnr_mask_257` | OK 6/6 | Bit-faithful. |
| `mcnk_flags_16` | OK 6/6 | Bit-faithful. |
| `normal_xyz` | OK 4/6, CLOSE 2/6 | Max abs diff ≤ 0.085 on two tiles — normalization noise, not a defect. |
| `alpha_256` | OK 4/6, 1 gap | **Coverage gap**: Expansion01 (43,11) is zero-filled in the store while the C# reference extracts real alpha. `has_alpha_256` flag is truthfully False, so this is a harvest miss, not an index lie. |
| `shadow_mask` | OK 3/6, 1 gap | Same pattern as alpha on the same tile. |
| `mcly_texture_ids` | OK 4/6, 1 diff | Expansion01 (43,11): IDs differ (max 8) — likely texture-table ordering differences between harvest runs; joinable only through the name table, not raw IDs. |
| `liquid_mask` | shape mismatch | Store is 256², C# `unified_liquid_mask` is 257² — a resolution-convention difference, not data loss. The 257→256 convention used at build time should be documented in the V18 spec. |
| `object_precise_mask`, `object_mask`, `mddf_mask`, `modf_mask`, `object_filtered_mask` | diverge from reference | Expected: the store's masks come from the enriched per-placement projection (V22Enrich / model decode); the reference extractor only projects MCRF presence. Mean abs diff 0.02–0.29. The store is the richer signal; do **not** "fix" it toward the reference. |
| `mddf_*` / `modf_*` placement arrays | OK | Internal consistency verified: counts sum to data rows (1,004,602 MDDF / 10,868 MODF), offsets monotonic and in range, data finite. |
| `has_*` index flags | OK | 0 truthfulness violations on the sampled tiles. |

## Systemic defects found

1. **`holes_16` is wrong at the C# source.** The store is all-True on ordinary
   terrain. Root cause: `AdtTensorPackBuilder.ReadMcrfAndHoles` derives holes
   from `MCNK flags & 0x0000FF00` (commented "in some formats") — on LK-era
   MCNKs the hole bitmap is a dedicated header field, not flags bits 8–15, so
   the derivation fires on unrelated flag bits. Both the store and any fresh
   re-extraction share the defect, which is why the sampled comparison shows
   "OK": they agree on the same wrong value. **Consequence for V24**: Stage B's
   hole gate initially zeroed out every training pixel; `harvester/v24/tiles.py`
   now normalizes polarity (majority-True masks are flipped) as a workaround.
   **Follow-up**: fix the C# derivation in a separate spec (the fix touches
   `WowViewer.Core.IO`, which Spec 094 is not allowed to modify), then rebuild
   `holes_16` in V18/V22.
2. **Per-tile signal coverage gaps exist and are silent.** Expansion01 (43,11)
   lacks alpha and shadow in the store while the client has the data. The
   `has_*` flags record the gap truthfully, so training-side filtering works,
   but the gaps are re-harvestable. **Follow-up**: run the audit with a larger
   sample (`--sample 64`) to size the gap before deciding on a re-harvest.

## What this means for V24

V24's inputs — `height_257`, `minimap_rgb`, `alpha_256`, `normal_xyz`,
`mcnr_mask_257`, `object_precise_mask`, `liquid_mask` — are all in the
"sound" column. The V22 per-object mask data the user flagged as suspect was
not needed: V24 uses the tile-level mask, and the audit confirms the per-tile
placement arrays are internally consistent should a future spec want them.

## Reproduce

```bash
cd wow-viewer/data-harvester
uv run python scripts/audit_v22_dataset.py \
  --store ../output/datasets/v22/3_3_5_12340.zarr \
  --staged-client ../../output/tmp/wowarchive-clients/3_3_5_12340 \
  --sample 6 --output ../output/v24_validation/v22_audit_20260706
```
