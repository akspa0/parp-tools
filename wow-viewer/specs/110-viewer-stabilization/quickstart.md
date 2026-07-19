# Viewer Stabilization Quickstart

## Phase 1 automated proof

From `I:\parp\parp-tools`:

```powershell
dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainLightingMathTests|FullyQualifiedName~TerrainMinimapCompositorTests"
dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

## Terrain-derived minimap export

The following is a user-run map export; it reads every occupied terrain tile and can take several
minutes. It writes per-tile PNGs, one stitched PNG, and `synthesis-manifest.json` under the output
directory. Do not treat the derived PNG as a raw client minimap. Production synthesis always uses
one 12:00 achromatic global light; map LIT and Light DBC runtime profiles are viewer-only inputs.

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -- \
  synthetic-minimap --client-root "H:\CLIENTS\<build>" --map "<MapName>" \
  --output-dir "wow-viewer\output\synthesized-minimaps\<MapName>" \
  --per-tile --whole-map
```

For the post-fidelity-correction visual check, start with `--limit 1 --per-tile` and inspect that
single PNG for stable material regions, correct Alpha MCAL/MCLY layer alignment, smoothly
interpolated terrain lighting (no dense-MCNR checkerboard), and no repeated-texture moire or blurred
interpolation bands. Only then increase to `--limit 4` or add `--whole-map`. Record configured
client root, build identity, and the manifest's fixed-noon-white lighting evidence. Normal synthesized RGB omits
MCSH, as ordinary minimaps do; `--bake-mcsh` is an explicit exceptional-history diagnostic preview,
not a normal export or training-label mode.

If the command reports `specular_companion_rgb_proxy`, it recovered a missing diffuse BLP from a
verified same-stem `_s.blp` companion. If it reports `related_diffuse_rgb_proxy`, that companion
was unavailable and it scanned the loaded archive/listfile catalog for the first successfully
decoded ordinary `.blp`. Exact and strongly similar basenames rank before shared directory/theme
tokens, which lets moved historical assets repair stale ADT links; material-only suffixes such as
`_s`, `_n`, or `_h` are excluded from that second tier. The manifest/dataset sidecar retains both
paths and the resolution kind. The live terrain viewer follows this same order and logs its selected
proxy. This is an RGB material proxy, not proof that native specular/alpha material behavior was
reproduced.

For an authored-minimap dataset stream, use `--stream-profile v22`. It performs the complete
terrain-texture decode needed to attach `minimap_lighting` metadata and emits texture sidecars only
when every MTEX entry stays name-aligned. Treat `estimated_time_of_day_hours` as a bucket candidate
only when `time_of_day_evidence` explicitly says so; it is never proof of the image's capture time.

## Phase 1 user visual proof

Run the Debug viewer with an approved configured client root. For one map with LIT and one map without:

1. Open Lighting and set a custom fog start/end range.
2. Change time of day through dawn, noon, dusk, and night.
3. Verify the active range says `User override`, survives lighting updates, and terrain stays visible.
4. On the LIT map, enable `Show LIT minimap markers`; verify positional markers appear in both the normal minimap and the full-screen minimap (`M`) with the same selected entry highlight.
5. In Lighting, select a positional LIT entry and double-click it; verify the camera frames that entry from above and the active fog range has not changed.
6. Reset the override; verify the source returns to `Lighting recommendation` or `Fallback`.
7. Open **Tools > Export > Synthesized Terrain Minimap...** and run one bounded per-tile export
   first. Verify its terrain is correctly projected, blended, and free of static-like texture noise.
   Verify the paired `_liquid` image does not paint narrow strips over dry terrain-cell boundaries,
   does not paint through terrain, and renders water blue, slime green, and WLM/WLL magma/lava
   through the orange magma palette where those sources exist.
   Then choose per-tile and whole-map output. Verify the manifest says `NoonWhiteGlobal` with
   `synthetic_minimap_fixed_noon_global_white`, terrain material colours are not LIT/DBC-tinted,
   and relief is north-lit rather than south-lit (basins must not read as mountains). The fixed
   north-west bearing remains asymmetric at noon. Confirm all successful tile PNGs are present and the combined PNG has
   the recorded bounds and transparent missing areas. If a tile fails, rerun only that coordinate with
   `--tile-x <x> --tile-y <y>`; the error must include the decode stage and first relevant source
   frame before any additional bounds fix is attempted.
8. Confirm Lighting's Fog Start and Fog End each show a visible slider track and grab. Change both
   endpoints and confirm the active range updates normally.
9. Open **Tools > Archeology**. Start playback, select Range, Layers, Playback, and Capture in
   turn, then pause and stop the run. Confirm the parent remains Tools > Archeology and the World
   surface contains no UniqueId range controls. In legacy UI, use the Tools menu's **UniqueId
   Archeology** action and confirm its dedicated window contains the same playback transport.

Record configured client root, build identity, and fingerprint with screenshots.

## Native M2 user proof (after Phase 2)

Open named representative M2 files from 1.0.0, 1.12.1, 2.4.3, and WotLK-or-later clients. Record:

- source path and build fingerprint;
- the displayed format/profile and renderer route;
- visible geometry/materials or the full capability diagnostic.

Do not treat an exported MDX file as M2 rendering proof.
