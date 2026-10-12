# Receipt — T080 missing textures bind the error texture (2026-09-27)

Operator direction (verbatim): "we should just point to the missing or error blp for missing textures, instead
of searching for missing textures, that's what the real engine does. good god, why is that literally wasting
time..." — recorded as the plan amendment of the same date.

## Files changed
- `src/viewer/WoWViewer/Rendering/M2Renderer.cs`: `ResolveReplaceableTexture` is database-only (naming probes,
  `.blp` list scan, hardcoded defaults, `FindExistingTexturePath`, `StripVariantSuffix` removed);
  `TryLoadMaterialTexture` binds the missing texture instead of retrying slots 11/1; `TryReadTextureBytes`
  retries inside the model directory only for bare file names.
- `src/viewer/WoWViewer/Rendering/M2Renderer.TextureStreaming.cs`: fallback stages removed; no candidate / no
  candidate loaded / failed upload → `ApplyMissingTexture`.
- `src/viewer/WoWViewer/Rendering/M2Renderer.MissingTexture.cs` (new, 77 lines): `Textures\ShaneCube.blp` when
  the data source has it, else a generated 8×8 magenta/black checker; one GL texture per data source, shared and
  reference-counted through `M2TextureCache`.
- `src/viewer/WoWViewer/Rendering/M2TextureStreamer.cs`: `Candidate.FallbackSlot` and `Request.Stage` removed.
- `src/viewer/WoWViewer/Rendering/ReplaceableTextureResolver.cs`: `Resolve` no longer falls back to
  `ResolveFromCreatureDirectory` / `ResolveFromCharacterDirectory` (both removed); the diagnostic
  `GetReplaceableResolutionCandidates` listing is unchanged.
- Not changed: the legacy MDX `ModelRenderer` keeps its own search.

## Verification
| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors; compiler warnings identical to base (NU1903 excluded) |
| `dotnet test WowViewer.slnx -c Debug` | 1 | 26 failures (Core.Tests 18, PM4 8), identical to the pre-existing E4 environmental set |

## Criterion → evidence
| Criterion | Evidence |
|---|---|
| No search for missing textures in the native M2 path | Code: the only replaceable resolution is `_texResolver.Resolve` (database); no `GetFileList(".blp")` remains in `M2Renderer*.cs` |
| Missing textures point at the error texture | Code: every failure branch calls `ApplyMissingTexture` / `AcquireMissingTexture` |
| Load time / visuals improved | **Not proven.** Operator capture owed (T081); build and tests do not show runtime behavior |
