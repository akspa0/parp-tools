# PM4 OBJ Export Feature Coverage Matrix

This matrix compares the capabilities of the legacy PM4 OBJ exporter and the current **Core.v2** implementation (`Pm4ObjExporter`).

| # | Feature / Data Element | Legacy Exporter | Core.v2 Exporter | Parity Status |
|---|------------------------|-----------------|------------------|---------------|
| 1 | Vertex positions (`v`) | ✅ | ✅ | ✅ Complete |
| 2 | Triangle faces (`f`) | ✅ | ✅ | ✅ Complete |
| 3 | Material-library reference line (`mtllib …`) | ✅ | ❌ | ❌ Missing |
| 4 | `usemtl` + per-object material assignment | ✅ (`default`, `positionData`, `commandData`) | ❌ | ❌ Missing |
| 5 | `g` / `o` group naming for mesh | ✅ (`g` + `o`) | ✅ (`o` only) | ⚠️ Partial (no `g`) |
| 6 | Separate objects for position-data points | ✅ | ❌ | ❌ Missing |
| 7 | Separate objects for command-data points | ✅ | ❌ | ❌ Missing |
| 8 | Bounding-box & dimension comments | ✅ | ❌ | ❌ Missing |
| 9 | Terrain-coordinate comment header | ✅ | ❌ | ❌ Missing |
|10 | Consolidated multi-file OBJ exporter | ✅ | ❌ | ❌ Missing |
|11 | Vertex normals (`vn`) | ❌ | ❌ | N/A (not in legacy) |
|12 | Texture coordinates (`vt`) | ❌ | ❌ | N/A (not in legacy) |
|13 | Smoothing groups (`s`) | ❌ | ❌ | N/A (not in legacy) |

Legend: ✅ = implemented; ❌ = not implemented; ⚠️ = partially implemented.

## Summary
The Core.v2 exporter currently covers only the basic geometry (`v`/`f`) and object name. All other features present in the legacy exporter are absent or partial. Implementation work should focus on rows 3–10 to reach full parity.
