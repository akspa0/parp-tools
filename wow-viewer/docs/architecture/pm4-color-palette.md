# PM4 Color Palette

**Status**: Active | **Owner**: WoWViewer

## Rule

Three color families, three jobs. Never mix them.

| Family | Use | Saturation |
|---|---|---|
| **Light pastels** | Containers, borders, bounds | Low — light values |
| **Dark pastels** | Mesh lines, mesh triangles | Low-medium — dark values |
| **Saturated (free)** | Markers, highlights, selection | High |

The pastel families are reserved — never put a marker or highlight color in pastel. The saturated family is reserved for **interactive signals** that must be unmistakable against the mesh.

## Light Pastels — Containers / Bounds

| Token | RGB | Use |
|---|---|---|
| `Pm4ColorObjectBounds` | `(1.00, 0.75, 0.80)` | Per-object bounds (sub-merged) |
| `Pm4ColorCk24Bounds` | varies by ck24Type (0.65-0.95 all channels) | Per-CK24 merged bounds, type-keyed |
| `Pm4ColorSelectedBounds` | `(1.00, 1.00, 0.95)` | Selected object inner bounds |
| `Pm4ColorMddfBounds` | `(1.00, 0.70, 1.00)` | MDDF (M2) instance bounds |
| `Pm4ColorModfBounds` | `(0.75, 0.95, 1.00)` | MODF (WMO) instance bounds |

## Dark Pastels — Mesh

| Token | RGB | Use |
|---|---|---|
| `GetPm4TypeColor(0x40)` | `(0.85, 0.55, 0.30)` | Ck24Type 0x40 (M2) |
| `GetPm4TypeColor(0x80)` | `(0.80, 0.40, 0.25)` | Ck24Type 0x80 (M2 ext) |
| `GetPm4TypeColor(other)` | `(0.80, 0.50, 0.30)` | Other Ck24Type |
| `GetTypeFlagColor(0x03)` | `(0.30, 0.65, 0.45)` | TypeFlag 0x03 (M2 top) |
| `GetTypeFlagColor(0x10)` | `(0.30, 0.55, 0.65)` | TypeFlag 0x10 (interior floor) |
| `GetTypeFlagColor(0x12)` | `(0.80, 0.45, 0.45)` | TypeFlag 0x12 (exterior solid) |
| `ColorFromHeight` | gradient `(0.45,0.55,0.80)` → `(0.80,0.50,0.45)` | Height color mode |

## Markers / Free Saturated

| Token | RGB | Use |
|---|---|---|
| `Pm4ColorMscn` | `(0.10, 0.95, 1.00)` | MSCN scene-graph connector anchor. One cube per MSUR surface (placed via `MSUR.MscnRefIndex`). Saturated cyan — pops against pastel mesh. See `pm4-chunk-semantics.md`. |
| `Pm4ColorMspv` | `(1.00, 0.20, 0.80)` | MSPV path-vertex position. One cube per `MSPI` index reached from an `MSLK` link's path-vertex chain. Only present when surfaces are connected. Saturated magenta — pops against pastel mesh and against MSCN. See `pm4-chunk-semantics.md`. |
| `Pm4ColorCentroid` | `(0.70, 0.55, 0.50)` | Per-object centroid pin (dark pastel) |
| `Pm4ColorMprl` | `(0.40, 0.80, 0.85)` | MPRL position reference (medium pastel teal) |
| `Pm4ColorHighlight` | `(0.20, 1.00, 0.95)` | **Search/highlight match — saturated teal** |
| `Pm4ColorSelection` | `(1.00, 0.95, 0.20)` | **Group selection — saturated yellow** |

## Anti-Patterns

- ❌ Bright neon mesh (`(0.10, 1.00, 0.40)`) — clashes with the saturated signal family
- ❌ White or near-white mesh — kills contrast with selection
- ❌ Saturated highlights inside the pastel mesh palette — they don't pop
- ❌ Saturated mesh that's also a selection color — can't tell what's selected

## Source of Truth

All PM4 color constants live in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` near the `ColorFromHeight` / `GetPm4TypeColor` helpers, in the `// PM4 color system` block.

When adding a new PM4 visualization, pick a family first. If the new visual is a **container**, use a light pastel. If it's a **mesh** (line/triangle), use a dark pastel. If it's a **signal** (marker/highlight/selection), use saturated.

For the canonical reading of what each PM4 stream actually is (MSUR, MSCN, MSLK, MSPI, MSPV, MSVI, MSVT, MPRL) and the TypeFlags classification (`0x03` walkable M2-top, `0x10` walkable interior floor, `0x12` structural exterior solid), see `wow-viewer/docs/architecture/pm4-chunk-semantics.md`. The "MSCN = polygon centroids / MSPV = shared vertices" reading previously used in the palette doc is dead.
