# Ghidra Findings Report: WoW Alpha 0.5.3 MDX Animation (Build 3368)

Date: 2026-02-13  
Target binary: `WoWClient.exe` (with `Wowae.pdb`)  
Method: live decompilation and call-path tracing in Ghidra

---

## Executive Summary (What Was Wrong)

The critical parser mismatch is confirmed:

- `KGTR/KGRT/KGSC` track headers are **`[tag + keyCount + interpolation + globalSeqId]`**.
- They are **not** `[tag + byteSize + ...]`.
- `KGRT` keys in the raw BONE stream are consumed as **compressed quaternions**, not float4 keys.

If your loader reads raw `KGRT` as `(time + x,y,z,w float)`, rotation data is garbage and animation appears static/incorrect.

---

## 1) Verified Load/Build Pipeline

### `BuildModelFromMdxData` (`0x00421fb0`)
Observed call order:

1. `MdxLoadGlobalProperties`
2. `MdxReadTextures`
3. `MdxReadMaterials`
4. `MdxReadGeosets`
5. `MdxReadAttachments`
6. `MdxReadAnimation`
7. `MdxReadRibbonEmitters`
8. `MdxReadEmitters2`
9. `MdxReadNumMatrices`
10. `MdxReadHitTestData` (flagged)
11. `MdxReadLights` (flagged)
12. `CollisionDataCreate`
13. `MdxReadExtents`
14. `MdxReadPositions` (includes PIVT handling)
15. `MdxReadCameras`

### `MdxReadAnimation` (`0x004221b0`)
- Calls `AnimCreate(...)` and stores animation handle.

### `AnimCreate` (`0x0073a850`)
- Counts section entries.
- Calls `AnimBuild(...)`.

### `AnimBuild` (resolved in session)
Observed animation build sequence:

1. `AnimAddSequences`
2. `AnimAddCameras`
3. `AnimAddGeosets`
4. `AnimAddTextureAnims`
5. `AnimAddMaterialLayers`
6. `AnimBuildObjectIdTranslation`
7. `IAnimCreateObjects`
8. `BuildHierarchy`
9. `AnimInit`

This confirms there *is* a post-parse wiring stage (`BuildHierarchy`, `AnimInit`).

---

## 2) Object Chunk Parse Order (Animation Objects)

From `IAnimCreateObjects`:

- `BONE` (`0x454E4F42`)
- `HTST` (optional)
- `LITE` (optional)
- `HELP`
- `ATCH`
- `PRE2`
- `RIBB`
- `EVTS`

So BONE object creation occurs before some other object types, but pivots are supplied via positions/pivot data paths and object-id remapping.

---

## 3) Node and BONE Binary Layout (Verified)

From `GenericHandlerAnim` and `CreateBone`:

### Shared Node Header
- `+0x00` `uint32 nodeSize` (inclusive; node end = `nodeStart + nodeSize`)
- `+0x04` `char name[0x50]`
- `+0x54` `int32 objectId`
- `+0x58` `int32 parentId`
- `+0x5C` `uint32 flags`
- `+0x60...` animation sub-chunks until node end

### BONE Post-Node Fields
After node data is consumed, `CreateBone` reads:
- `int32 geosetId`
- `int32 geosetAnimId`

So `geosetId/geosetAnimId` are **outside** `nodeSize`.

---

## 4) Track Header/Body Layout (Decisive)

Raw-stream parsers called by `GenericHandlerAnim`:

- `AnimObjectSetTranslation` @ `0x0074e260`
- `AnimObjectSetRotation` @ `0x0074e930`
- `AnimObjectSetScaling` @ `0x0074ef10`

### Common Track Header (all 3)
- `+0x00` `uint32 tag` (`KGTR/KGRT/KGSC`)
- `+0x04` `uint32 keyCount`
- `+0x08` `uint32 interpType`
- `+0x0C` `int32 globalSeqId` (`-1` = none)
- `+0x10` key data

### KGTR / KGSC key payload
- Linear (`interp 0/1` effective): stride `0x10`
  - `int32 time`
  - `float3 value`
- Hermite/Bezier (`interp 2/3`): stride `0x28`
  - `int32 time`
  - `float3 value`
  - `float3 inTan`
  - `float3 outTan`

### KGRT key payload
- Linear: stride `0x0C`
  - `int32 time`
  - `C4QuaternionCompressed` (8 bytes)
- Hermite/Bezier: stride `0x1C`
  - `int32 time`
  - `C4QuaternionCompressed value`
  - `C4QuaternionCompressed inTan`
  - `C4QuaternionCompressed outTan`

This is the single highest-impact finding for your viewer.

---

## 5) Quaternion Compression Details (Verified)

### Functions
- `C4QuaternionCompressed::Set` @ `0x0075bad0`
- `C4QuaternionCompressed::GetW` @ `0x0075ba30`
- `C4QuaternionCompressed::operator C4Quaternion` @ `0x0074d690`

### Important behavior
- Internal representation is packed signed components, reconstructed with scale constants.
- `operator C4Quaternion` writes fields in a non-obvious order in decompilation; rely on function behavior, not guessed component ordering.
- Runtime interpolation for rotation uses compressed track types:
  - `InterpolateLinear` @ `0x0075de20`
  - `InterpolateHermite` @ `0x0075dc00`
  - `InterpolateBezier` @ `0x0075de00` (stubbed error path in this build)

---

## 6) SEQS and GLBS Layout (Verified)

From `AnimAddSequences`:

### SEQS
- Has leading `uint32 count` in this build path.
- Entry size is `0x8C` (140 bytes), matching your current alpha assumption.
- Parser advances by `0x23` dwords per sequence.

### GLBS
- Read as plain `uint32[]` durations (`chunkBytes / 4`).
- Stored into global sequence length arrays.

---

## 7) Time Units and Advancement

From `AnimAdvanceTime`, `AdvanceTime`, `SetGlobalSequenceTime`, `SetSequenceTime`, `CKeyFrameTrackBase::SetAnimTime`:

- Engine time source is async millisecond clock (`OsGetAsyncTimeMsPrecise` path).
- Global sequences are advanced as `elapsed = (elapsed + delta) % duration`.
- Sequence/local tracks use sequence elapsed time + per-track key search.
- Global-seq tracks are selected by `globalSeqId != -1` and use global elapsed arrays.

Conclusion: timing is millisecond-based in the animation system path traced.

---

## 8) Bone/Object Transform Evaluation Path

Core animation evaluation path:

1. `ModelAnimate` -> `IModelAnimate`
2. `AnimAnimateModel` -> `IAnimAnimateModel`
3. `PrepareObjectHierarchyViews` (recursive hierarchy walk)
4. `TransformObjectView`
   - `TranslateView`
   - `RotateView` (+ billboard/face-dir cases)
   - `ScaleView`
5. `PlaceObject`
   - Bone objects write world matrix snapshot to bone matrix array.

`PrepareObjectHierarchyViews` uses world-matrix push/pop recursion, so parent transform context is active for children.

---

## 9) Matrix Math Convention and Composition Evidence

From NTempest matrix operators:

- `C34Matrix::Translate` updates `d0/d1/d2` using current basis (local-space translation accumulation).
- `NTempest::operator*=(C3Vector, C34Matrix)` shows transformed position:
  - `x' = x*a0 + y*b0 + z*c0 + d0`
  - `y' = x*a1 + y*b1 + z*c1 + d1`
  - `z' = x*a2 + y*b2 + z*c2 + d2`

This confirms row-vector style usage in this path and is consistent with world-stack incremental composition.

---

## 10) Skinning/Geoset Matrix Data Flow (Runtime)

### Loading
`LoadGeosetTransformGroups` reads:
- `GNDX` -> vertex group ids
- `MTGC` -> matrix-group counts
- `MATS` -> matrix indices
- `BIDX` -> bone indices array
- `BWGT` -> bone weights array

### Render-side usage confirmed
Matrix group path actively used:
- `AddGeosetMatrixGroups` (`0x004421b0`)
- `AddMatrixGroupRangeToSet` (`0x00442310`)
- `SetGeosetMatrix` / `BuildPrimBones` / `BuildPrimBone`
- `RenderGeosetPrep` -> `GxXformSetBones(numGroups, matrixPtr)` for weighted geosets

### Important nuance
In traced primary render chain, geoset palette setup is driven by group/matrix tables (`GNDX/MTGC/MATS`-derived), while `BIDX/BWGT` are loaded/copied but were not observed as direct drivers in the main geoset draw path inspected.

---

## 11) Practical Fix Guidance for Viewer

1. Keep Node parse exactly as verified (`size/name/objectId/parentId/flags`, sub-chunks from `+0x60`).
2. Parse BONE tail fields (`geosetId`, `geosetAnimId`) after node end.
3. Parse track headers as `[tag + keyCount + interp + globalSeqId]`.
4. For `KGRT`, parse compressed quaternion key payloads (not float4).
5. Use millisecond timeline semantics for sequences/global sequences.
6. Ensure hierarchy update uses parent transform context before child evaluation.
7. Keep geoset matrix group path (`GNDX/MTGC/MATS`) intact for skinning palette construction.

---

## 12) Most Impactful Confirmations (Checklist)

- [x] `KGTR/KGRT/KGSC` second field is `keyCount`, not `byteSize`
- [x] Node `nodeSize` is inclusive from size field start
- [x] BONE `geosetId/geosetAnimId` are outside node block
- [x] KGRT raw keys are compressed quaternions
- [x] Post-parse animation wiring exists (`BuildHierarchy`, `AnimInit`)
- [x] Global sequences are duration arrays with modulo-time advancement

---

## Appendix A: Key Addresses Used

- `BuildModelFromMdxData` `0x00421fb0`
- `MdxReadAnimation` `0x004221b0`
- `AnimCreate` `0x0073a850` / `0x0073cb10` overloads
- `IAnimCreateObjects` (resolved in session)
- `GenericHandlerAnim` `0x0073b110`
- `AnimObjectSetTranslation` raw `0x0074e260`
- `AnimObjectSetRotation` raw `0x0074e930`
- `AnimObjectSetScaling` raw `0x0074ef10`
- `C4QuaternionCompressed::Set` `0x0075bad0`
- `C4QuaternionCompressed::GetW` `0x0075ba30`
- `C4QuaternionCompressed::operator C4Quaternion` `0x0074d690`
- `InterpolateLinear` `0x0075de20`
- `InterpolateHermite` `0x0075dc00`
- `InterpolateBezier` `0x0075de00`
- `AnimAdvanceTime` / `AdvanceTime` / `SetSequenceTime` / `SetGlobalSequenceTime` (resolved in session)
- `IModelAnimate` `0x00439ea0`
- `SetGeosetMatrices` `0x00439880`
- `BuildPrimBones` `0x004399c0`
- `BuildPrimBone` `0x00439a40`
- `SetUnanimatedGeosetMatrices` `0x00439bd0`
- `SetUnanimatedGeosetMatrix` `0x00439cb0`
- `RenderGeosetPrep` `0x00430030`

---

If needed, next step is to convert this into a C# parser contract document (structs + exact read strides + validation asserts) and then patch `MdxViewer` accordingly.