> **Reverted 2026-09-27** after the operator reported worse performance; code restored to `c2e9517`. This
> receipt records what was tried. See the spec amendment "R-39 code reverted".

# Receipt — R-39 CASC minimap streaming cost (2026-09-27)

Operator-approved R-39a/b/c (spec amendment 2026-09-27). Covers **R39-T001–T004**. No FPS, lag, GC or
visual result is claimed; that is **R39-T005** (operator, CASC client).

## Files changed

| File | Change |
|---|---|
| `src/viewer/WoWViewer/Rendering/MinimapRenderer.cs` | **a)** after a tile's first successful data-source read its BLP bytes are written to `<cacheRoot>/<sha1>.blp` (temp file + move); loads check that file first, then the legacy `<sha1>.png`. The unused `TrySaveCachedBitmap` (PNG writer, no caller) became `TrySaveCachedTile`. **b)** `DecodeTile`: DXT BLP2 tiles keep level 0 compressed (format chosen exactly as SereniaBLPLib's decoder: alpha depth > 1 → DXT5 if pixel format 7 else DXT3; otherwise RGBA-DXT1) and upload with `glCompressedTexImage2D` when `GL_EXT_texture_compression_s3tc` is present; other BLP2 encodings decode with `BlpFile.GetPixels` into one array; anything else (or a header Core's `BlpSummaryReader` rejects) takes the previous `GetImage` route. Level 0 only, same linear/clamp parameters, so display sampling is unchanged. **c)** the worker is a dedicated `Thread` at `ThreadPriority.Lowest` (was a pool task); before each tile it waits in 50 ms steps while `worldAssetsLoading` is set, at most 1 s per tile; on `CascDataSource` it tries `world/minimaps/<map>/mapXX_YY.blp` (the existing candidate string) first. A tile read that throws is logged and recorded as failed instead of ending the worker (on a dedicated thread it would end the process). |
| `src/viewer/WoWViewer/ViewerApp.cs` | existing `ProcessPendingLoads` call passes `worldAssetsLoading: (_worldScene?.PendingAssetLoadCount ?? 0) > 0` (argument only; no new member) |

Not changed: the BLP library (`libs/…/SereniaBLPLib`), any format reader, MPQ candidate order, texture
filtering.

## Verification

| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true --no-incremental` | 0 | 0 errors; compiler warnings identical to baseline (`NU1903` restore warning excluded, run-to-run variable) |
| `dotnet test WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 1 | same 26 pre-existing environmental failures |
| `dotnet build src/viewer/WoWViewer/WoWViewer.csproj …` after the final one-line change (summary-reader catch widened to fall back on any header error) | 0 | 0 errors (viewer has no test project) |
| scratch equivalence check (agent scratchpad `blpcheck`, Core.IO + SereniaBLPLib; synthetic 256² and 512² BLP2s) | 0 | see below |

Equivalence check results (synthetic data, not client files):

| Check | 256² | 512² |
|---|---|---|
| `BlpSummaryReader` level-0 slice = expected DXT1 size, in bounds | 32,768 B at 148 | 131,072 B at 148 |
| GL RGBA-DXT1 reference decode (EXT_texture_compression_s3tc rules) of the uploaded slice vs SereniaBLPLib decode, random blocks incl. 2,065 / 8,265 three-colour/transparent blocks | 0 bytes differ | 0 bytes differ |
| `GetPixels(bgra: Uncompressed)` vs `GetImage` + copy, DXT1 BLP2 | identical | identical |
| same, hand-built ARGB8888 BLP2 | identical | identical |

Caveat: GL implementations may round DXT interpolants differently from the reference by ±1; the reference
itself matches the library bit for bit.

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| R-39a cache written and read | code; second-session behaviour | code — **operator check owed** |
| R-39b fewer per-tile allocations | DXT: file bytes only (no decode arrays); others: one array instead of image + copy | code + equivalence check |
| R-39c throttle | lowest-priority thread, bounded deferral, CASC path first | code |
| No whole-program lag on CASC while tiles stream | — | **not claimed (R39-T005)** |
| Minimap looks unchanged | equivalence check on synthetic tiles | **operator visual check owed** |

## Operator check (R39-T005, PowerShell 7)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Open a CASC client, load a large map, fly with WDL far terrain and the minimap window visible. Watch the
sidebar heap line (`GC=a/b/c`: the third number should climb far more slowly than before) and whether the
program still stalls. Compare the minimap with the previous build. Restart and reload the same map:
tiles should appear quickly from the cache (`<viewer bin>/output/cache/minimap/<segment>/` gains `.blp` files).
Note: tiles fill in more slowly while world assets are loading — by design.
