# Evidence Receipt: Creature Variant Texture Resolution & Replaceable Texture Binding Fix

**Date**: 2026-09-13  
**Feature Branch**: `235-legacy-mdx-m2-rendering`  
**Owner Spec**: [spec.md](../spec.md)  
**Governance Standard**: `AGENTS.md` §9.2 (Receipts required)

---

## 1. Problem Statements & Root Causes

### 1.1 Defect A: Creature Variants (`DragonSpawnArmored.mdx`) Render as Pure White Meshes
- **Symptom**: 0x100 creature model variants (e.g. `Creature\DragonSpawn\DragonSpawnArmored.mdx`) load geometry but render completely untextured / flat white in `M2Renderer`.
- **Root Cause 1 (`WowViewerM2RuntimeBridge.cs:BuildEra100Material`)**:
  In `BuildEra100Material`, the texture binding loop ran `for (int stage = 0; stage < batch.TextureCount; stage++)`. When `batch.TextureCount` was 0 (or uninitialized), `bindings` was empty, `primary` was null, and both `TexturePath` and `ReplaceableId` remained null/0. Additionally, when `lookupIndex >= geometry.TextureLookup.Count`, it collapsed to `lookupValue = -1` without falling back to direct texture indexing `geometry.Textures[lookupIndex]`.
- **Root Cause 2 (`M2Renderer.cs` & `ReplaceableTextureResolver.cs` Directory Search)**:
  In models like `DragonSpawnArmored.mdx`, `modelBase` is `"DragonSpawnArmored"`. In the MPQ/directory `Creature\DragonSpawn\`, textures are named after the creature family/folder (`DragonSpawnSkin.blp`, `DragonSpawn.blp`, etc.).
  The previous resolver logic only tested `modelBase + suffix` (`DragonSpawnArmoredSkin.blp`) and filtered directory scans with `fName.StartsWith(modelBaseLower)`. Because `"dragonspawnskin".StartsWith("dragonspawnarmored")` was `false`, directory scans yielded zero candidates and failed to find the valid skin textures.
- **Root Cause 3 (`M2Renderer.cs:TryLoadMaterialTexture` Untextured Mesh Fallback)**:
  If candidate metadata in a model's batch had no texture path and `replaceableId == 0`, `TryLoadMaterialTexture` simply skipped it and returned `false`, leaving the shader with `uHasTexture = 0` (flat white) instead of attempting a directory-based creature skin fallback.

### 1.2 Defect B: Character Models Rendered Without Default Underwear / Inappropriate Geosets
- **Symptom**: Naked character models in the viewer rendered naked without underwear or had robe skirts and cloaks enabled.
- **Root Causes**:
  - `DefaultCharacterSelectionGroups` previously included robe skirts (`1201`), cloaks (`1101`), tabards (`1301`), and shoulders (`1401`/`1501`), which should be item-equipped, not active by default on naked base characters.
  - Underwear resolution queried `CharSections.dbc` `baseSection: 0` before checking `baseSection: 5` (Underwear) and `baseSection: 4`, missing the real `NakedPelvisSkin` / `NakedTorsoSkin` underwear textures.

---

## 2. Remediation & Architecture Changes

1. **`WowViewerM2RuntimeBridge.cs`**:
   - Clamped batch stage count: `int stageCount = Math.Max(1, (int)batch.TextureCount);`.
   - Added direct texture index fallback: when `lookupIndex >= geometry.TextureLookup.Count` but `lookupIndex < geometry.Textures.Count`, use `geometry.Textures[lookupIndex]`.
   - Defaulted negative/unresolved lookups to `replaceableId = 1`, and empty-filename texture entries to creature skin slot `11`.

2. **`M2Renderer.cs`**:
   - Added `CreatureVariantSuffixes` and `StripVariantSuffix` helper to recognize creature variants (`Armored`, `Caster`, `Overlord`, `Rider`, etc.).
   - In `ResolveReplaceableTexture`, search against `candidateBases = [ modelBase, folderBase, strippedBase ]`.
   - In Strategy 3 (Directory Scan), evaluate all `.blp` files in `_modelDir` with a scoring heuristic matching skin (`1, 11-13`), detail/underwear (`2, 8`), hair (`6, 10`), and facial hair (`7`).
   - In `TryLoadMaterialTexture`, added a safety net: if no candidate texture was loaded, attempt resolving creature skin slots (`11`, then `1`) from directory heuristics.

3. **`ReplaceableTextureResolver.cs`**:
   - Updated `ResolveFromCreatureDirectory` to check `[ modelBase, folderBase, strippedBase ]` and directory BLP scoring.
   - Updated `FindModelId` to fall back to `folderBase` (e.g. `DragonSpawn` for `DragonSpawnArmored`) and `strippedBase` in `_modelFileNameToId` and `_modelPathToId`.
   - Refined `CharSections` underwear queries to prioritize `baseSection: 5` (Underwear) and added client filenames (`NakedPelvisSkin`, `NakedTorsoSkin`, etc.).
   - Pruned robe (`1201`), cloak (`1101`), tabard (`1301`), and shoulders (`1401`/`1501`) from `DefaultCharacterSelectionGroups`.

4. **`ModelRenderer.cs`**:
   - Aligned Strategy 2 and Strategy 3 with `folderBase`, `strippedBase`, and scored candidate resolution.

---

## 3. Files Changed

| Component | File Path | Description of Change |
|---|---|---|
| M2 Runtime Bridge | [WowViewerM2RuntimeBridge.cs](../../../src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs) | Enforced `stageCount >= 1`, added direct texture lookup fallback, defaulted empty filenames to replaceable slot 11. |
| Native M2 Renderer | [M2Renderer.cs](../../../src/viewer/WoWViewer/Rendering/M2Renderer.cs) | Added variant suffix stripping, folderBase matching, dir-scan scoring, and safety-net fallback in `TryLoadMaterialTexture`. |
| Texture Resolver | [ReplaceableTextureResolver.cs](../../../src/viewer/WoWViewer/Rendering/ReplaceableTextureResolver.cs) | Added folderBase and strippedBase to `FindModelId` and `ResolveFromCreatureDirectory`; underwear prioritization; pruned robe/cloak geosets. |
| Model Renderer | [ModelRenderer.cs](../../../src/viewer/WoWViewer/Rendering/ModelRenderer.cs) | Added folderBase, strippedBase, and scored dir-scan resolution. |
| Unit Tests | [M2EmbeddedProfileRealDataTests.cs](../../../tests/WowViewer.Core.Tests/M2EmbeddedProfileRealDataTests.cs) | Added unit test `BuildEra100StaticRenderModel_TransfersTextureBindingsAndFallbackSlots`. |

---

## 4. Verification Commands & Outputs

### 4.1 Solution Compilation
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: `0`
- **Errors**: `0`

### 4.2 Targeted Era100 Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Era100"
```
- **Exit Code**: `0`
- **Output**: `Passed! - Failed: 0, Passed: 18, Skipped: 0, Total: 18`

### 4.3 Static Render Model Texture Transfer Test
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~BuildEra100StaticRenderModel_TransfersTextureBindingsAndFallbackSlots"
```
- **Exit Code**: `0`
- **Output**: `Passed! - Failed: 0, Passed: 1, Skipped: 0, Total: 1`
