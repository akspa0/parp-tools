# Ghidra Investigation: MDX Animation System — WoW Alpha 0.5.3 (Build 3368)

## Context

We have a working MDX viewer that loads and renders static MDX models from WoW Alpha 0.5.3. We have a PDB (`Wowae.pdb`) for this binary, so all function names and many type names are available.

**What works:** Sequence names parse correctly (SEQS chunk). Bone hierarchy is built. Keyframe interpolation code exists (linear, hermite, bezier). Bone matrices are calculated each frame.

**What does NOT work:** No model actually animates. Bone matrices appear to be identity or wrong. Vertices never move. We suspect our binary parsing of the BONE chunk's animation tracks (KGTR/KGRT/KGSC) may be reading wrong data, or our bone matrix composition is wrong, or the skinning data (vertex→bone mapping) is being discarded.

**Binary:** `WoWClient.exe` — Build 3368, Dec 11 2003, with `Wowae.pdb`

---

## TASK 1: MDX File Loading Pipeline (CRITICAL)

### Goal
Trace the complete MDX loading pipeline to understand the exact order chunks are parsed and how animation data is wired to the rendering system.

### Entry Point
```
BuildModelFromMdxData @ 0x00421fb0
```
This is the 15-step MDX loading pipeline. We need the EXACT order of operations.

### Questions
1. **What is the complete list of chunk FourCCs processed, in order?** List every chunk tag this function (and its callees) handles.
2. **Does BONE come before or after PIVT?** Our parser reads PIVT to assign pivot points to bones by ObjectId. If BONE is parsed first, pivots wouldn't be available yet — we'd need a second pass.
3. **After all chunks are parsed, what final wiring/initialization happens?** Is there a `BuildHierarchy()` or `AnimInit()` call that connects bones to geosets?
4. **Is there a separate "animation build" step after geometry loading?** The deep dive mentions `AnimBuild @ 0x0073cb10` — is this called from `BuildModelFromMdxData` or separately?

### Key Functions to Trace
| Function | Address | What to look for |
|----------|---------|-----------------|
| `BuildModelFromMdxData` | `0x00421fb0` | Chunk dispatch loop, order of chunk processing |
| `MdxReadAnimation` | `0x004221b0` | How animation handle is created from MDX data |
| `AnimBuild` | `0x0073cb10` | How CAnimData is populated from raw MDX bytes |
| `AnimCreate` | `0x0073a850` | Animation instance creation |

---

## TASK 2: BONE Chunk Binary Layout (CRITICAL)

### Goal
Determine the EXACT binary layout of a BONE entry, including the Node header and animation sub-chunks.

### Current Parser Assumption
```
BONE chunk:
  uint32 count
  BoneEntry[count]:
    Node:
      uint32 nodeSize          // inclusive size of entire node block
      char[0x50] name          // 80-byte fixed string
      int32 objectId
      int32 parentId
      uint32 flags
      // Animation sub-chunks follow until nodeEnd:
      KGTR sub-chunk (optional) — translation keys
      KGRT sub-chunk (optional) — rotation keys  
      KGSC sub-chunk (optional) — scaling keys
    // After node:
    int32 geosetId
    int32 geosetAnimId
```

### Questions
1. **Is `nodeSize` the first field, and does it include itself?** i.e., does `nodeEnd = position_before_nodeSize + nodeSize` or `position_after_nodeSize + nodeSize`?
2. **Is the name field exactly 0x50 (80) bytes?** Or could it be shorter/longer in Alpha?
3. **Are `objectId`, `parentId`, `flags` in that exact order after the name?**
4. **Do `geosetId` and `geosetAnimId` come AFTER the node block (outside nodeSize), or INSIDE it?**
5. **Are there any fields between `flags` and the first animation sub-chunk?** Some MDX variants have a `billboarded` flag or extra padding here.

### Key Function
```
Look for the BONE chunk handler inside BuildModelFromMdxData or AnimBuild.
Search for FourCC 0x454E4F42 ("BONE") in the chunk dispatch.
The function that creates CAnimBoneObj entries is:
  AnimObjectCreateBone @ 0x0074d8e0
```

### Verification Method
Pick a known MDX file (e.g., `Creature/DireWolf/DireWolf.mdx`) and:
1. Find the BONE chunk offset in the hex dump
2. Trace the binary reader through the Ghidra code
3. Confirm each field offset matches

---

## TASK 3: Animation Track Sub-Chunk Layout (CRITICAL)

### Goal
Determine the EXACT binary layout of KGTR, KGRT, and KGSC sub-chunks inside a Node.

### Current Parser Assumption
```
KGTR (Translation) sub-chunk:
  char[4] tag = "KGTR"
  uint32 keyCount
  uint32 interpolationType    // 0=None, 1=Linear, 2=Hermite, 3=Bezier
  int32 globalSeqId           // -1 if not global
  KeyData[keyCount]:
    int32 frame               // time in milliseconds(?)
    float x, y, z             // translation value
    [if hermite/bezier:]
    float inTanX, inTanY, inTanZ
    float outTanX, outTanY, outTanZ

KGRT (Rotation) sub-chunk:
  char[4] tag = "KGRT"
  uint32 keyCount
  uint32 interpolationType
  int32 globalSeqId
  KeyData[keyCount]:
    int32 frame
    float x, y, z, w          // quaternion (4 floats? or compressed?)
    [if hermite/bezier:]
    float inTanX, inTanY, inTanZ, inTanW
    float outTanX, outTanY, outTanZ, outTanW

KGSC (Scaling) sub-chunk:
  char[4] tag = "KGSC"
  uint32 keyCount
  uint32 interpolationType
  int32 globalSeqId
  KeyData[keyCount]:
    int32 frame
    float x, y, z
    [if hermite/bezier:]
    float inTanX, inTanY, inTanZ
    float outTanX, outTanY, outTanZ
```

### Questions
1. **Is the sub-chunk header `[tag(4) + keyCount(4)]` or `[tag(4) + size(4)]`?** Our parser reads `keyCount` after the tag. But some formats use a byte-size instead, and keyCount is inside the data.
2. **Is `interpolationType` a uint32 immediately after keyCount?** Or is it packed differently?
3. **Is `globalSeqId` present in every track, or only when a global sequence flag is set?**
4. **Are rotation quaternions stored as 4 floats (X,Y,Z,W) or compressed?** The deep dive doc mentions "C4QuaternionCompressed" with 10-bit components. Which format does Alpha 0.5.3 actually use for KGRT?
5. **What is the quaternion component order?** `(X,Y,Z,W)` or `(W,X,Y,Z)`?
6. **Are keyframe times in milliseconds?** Or some other unit (frames at 30fps, ticks)?

### Key Functions
```
Look for KGTR/KGRT/KGSC FourCC constants:
  KGTR = 0x5254474B
  KGRT = 0x5452474B  
  KGSC = 0x4353474B

Track reading functions (from the deep dive):
  InterpolateLinear @ 0x0075de20
  InterpolateHermite @ 0x0075dc00
  InterpolateBezier @ 0x0075de00

Also look for the track READER functions — the ones that parse binary data into
the CAnimTrack structure. These will definitively show the layout.
```

### Critical Sub-Question: Track Tag as Size vs Count
If the field after the 4-byte tag is a **byte size** rather than a **key count**, our entire track parsing is wrong — we'd be reading `interpolationType` as the first keyframe's time, etc. This would explain why animations appear static (garbage data in keys).

---

## TASK 4: SEQS Entry Layout Verification (HIGH)

### Goal
Confirm the exact SEQS entry size and field layout for Alpha 0.5.3.

### Current Parser Assumption
```
SEQS chunk:
  uint32 count
  Entry[count] where entrySize = (chunkSize - 4) / count
  
  Entry (140 bytes for Alpha):
    char[0x50] name           // 80 bytes
    uint32 intervalStart      // start time (ms)
    uint32 intervalEnd        // end time (ms)
    float moveSpeed
    uint32 flags
    float frequency
    int32 replayStart
    int32 replayEnd
    uint32 blendTime
    float boundsRadius
    float boundsMinX, boundsMinY, boundsMinZ
    float boundsMaxX, boundsMaxY, boundsMaxZ
```

### Questions
1. **Is there a leading `uint32 count` before the entries?** Standard WC3 MDX has no count — entries are `chunkSize / entrySize`. Alpha may differ.
2. **Is the name field 80 bytes (0x50)?** We confirmed "Stand - Var1" reads correctly with this assumption for DireWolf.mdx, but verify in the binary.
3. **Are `intervalStart` and `intervalEnd` in milliseconds?** This is critical — if they're in some other unit, our keyframe time matching is broken.
4. **What is the exact entry size?** We infer 140 from `(chunkSize - 4) / count`. Confirm from the struct definition in the binary.

### Key Function
```
AnimAddSequences @ 0x00754000
Search for SEQS FourCC: 0x53514553
```

---

## TASK 5: Bone Matrix Composition Order (CRITICAL)

### Goal
Determine the EXACT matrix composition order the client uses to transform a bone.

### Current Implementation
```csharp
// Our code (MdxAnimator.cs):
localMatrix = T(-pivot) * S(scale) * R(rotation) * T(pivot) * T(translation)
worldMatrix = localMatrix * parentMatrix
```

### Questions
1. **What is the exact local matrix composition?** Is it:
   - `T(-pivot) * S * R * T(pivot) * T(translation)` (our current code)
   - `T(translation) * T(pivot) * R * S * T(-pivot)` (reversed)
   - `T(translation + pivot) * R * S * T(-pivot)` (combined)
   - Something else entirely?
2. **Is the world matrix `local * parent` or `parent * local`?** This depends on row-major vs column-major convention.
3. **Does the client use row-major or column-major matrices?** DirectX typically uses row-major (row vectors), OpenGL uses column-major. System.Numerics uses row-major.
4. **Is the pivot subtracted before or after rotation?** The standard approach is: translate to origin (subtract pivot), apply rotation/scale, translate back (add pivot), then apply translation.
5. **Are there any additional transforms?** Billboard flags, inverse bind pose, etc.?

### Key Functions
```
The bone evaluation function that computes the final matrix per bone.
Look for calls to:
  AnimGetObjectPosition @ 0x00741c00
  AnimApplyObjectFaceDir @ 0x007414a0

Also look for the per-frame bone update loop — the function that iterates
all bones and computes their world matrices. This is the most important
function for this task.

Matrix multiplication functions in NTempest namespace:
  C34Matrix::Multiply or similar
  C44Matrix operations
```

---

## TASK 6: Vertex Skinning — How Bones Apply to Vertices (CRITICAL)

### Goal
Understand how the client applies bone matrices to vertex positions at render time.

### Current Problem
Our parser reads GNDX (vertex groups), MTGC (matrix groups), and BIDX/BWGT (bone indices/weights) but **discards all of them**. We only store MATS (MatrixIndices). Without the vertex→bone mapping, skinning cannot work.

### Questions
1. **What is the exact vertex→bone mapping data flow?**
   - `GNDX[vertexIndex]` → group index (1 byte per vertex)
   - `MTGC[groupIndex]` → bone count for that group
   - `MATS[offset]` → actual bone indices
   - Is this correct? Or does Alpha use a different scheme?

2. **Does Alpha 0.5.3 use BIDX/BWGT (per-vertex bone indices + weights) instead of GNDX/MTGC/MATS?**
   - BIDX and BWGT appear in some Alpha MDX files
   - Are these an alternative to the GNDX/MTGC system?
   - If BIDX exists, how many bones per vertex? 1? 4?

3. **Is skinning done on CPU or GPU in the Alpha client?**
   - If CPU: the client transforms vertices before uploading to D3D
   - If GPU: the client uploads bone matrices as shader constants
   - Alpha uses D3D8 which has limited vertex shader support — likely CPU skinning

4. **What is the actual skinning formula?**
   ```
   skinnedPos = sum(boneMatrix[i] * bindPos * weight[i]) for each bone
   ```
   Or does it use a simpler single-bone assignment (no blending)?

5. **Is there an inverse bind pose matrix?** Or are vertices already in bone-local space?

### Key Functions
```
The rendering function that draws geosets:
  Look for D3D DrawPrimitive/DrawIndexedPrimitive calls
  
The vertex transform function:
  Look for where vertex positions are multiplied by bone matrices
  This might be in a function called from the geoset render path

Search for references to the GNDX/MTGC data after it's loaded.
The function that uses VertexGroups to look up bone indices is key.
```

---

## TASK 7: Global Sequences (MEDIUM)

### Goal
Understand how global sequences interact with bone animation tracks.

### Current Implementation
```csharp
// If track.GlobalSeqId >= 0, use global sequence time instead of animation time
if (globalSeqId >= 0 && globalSeqId < _globalSeqFrames.Length)
{
    frame = _globalSeqFrames[globalSeqId];
    from = 0;
    to = (int)_mdx.GlobalSequences[globalSeqId];
}
```

### Questions
1. **Is `GlobalSeqId` stored per-track or per-bone?** Our parser reads it per-track (inside KGTR/KGRT/KGSC). Is this correct?
2. **Is the GLBS chunk just an array of uint32 durations?** `chunkSize / 4 = count`, each entry is a duration in ms?
3. **When a track uses a global sequence, does the keyframe time range always start at 0?** i.e., keys go from `0` to `globalSeqDuration`?
4. **How does the client advance global sequence time?** Is it wall-clock time modulo duration? Or tied to the animation system's delta time?

### Key Function
```
Look for GLBS FourCC: 0x53424C47
Look for how globalSeqId is used during track evaluation.
```

---

## TASK 8: Geoset Animation (ATSQ/GEOA) — Visibility & Alpha (MEDIUM)

### Goal
Understand how geoset visibility and alpha animation works, since some geosets may be hidden/shown per-sequence.

### Questions
1. **What is the ATSQ chunk layout?** Our parser reads:
   ```
   Entry[]:
     uint32 entrySize
     uint32 geosetId
     float defaultAlpha
     float defaultColorR, G, B
     uint32 unknown
     Sub-chunks: KGAO (alpha keys), KGAC (color keys)
   ```
   Is this correct?

2. **How does geoset visibility work?** Is there a separate visibility track, or is alpha=0 used for hidden?

3. **Is `geosetId` in ATSQ the same as the geoset's index in the GEOS array?** Or is it the `SelectionGroup` field?

4. **Does the client skip rendering geosets with alpha=0?** Or does it always render and rely on alpha blending?

### Key Function
```
AnimAddGeosets @ 0x00754c00
KGAO FourCC: 0x4F41474B
KGAC FourCC: 0x4341474B
```

---

## TASK 9: Animation Time Units and Playback Speed (HIGH)

### Goal
Definitively determine what time units are used throughout the animation system.

### Questions
1. **Are SEQS intervalStart/intervalEnd in milliseconds?**
2. **Are keyframe `Frame` values in the same units as SEQS intervals?**
3. **What deltaTime does the client pass to the animation update function?** Is it milliseconds since last frame? Seconds?
4. **Is there a global time scale or playback speed multiplier?**
5. **What frame rate does the animation system assume?** 30fps? 60fps? Variable?

### Key Functions
```
AnimGetObjectTimeScale @ 0x0074b680
UpdateBaseAnimation @ 0x005f56a0
ChooseAnimation @ 0x005fbd10
PlayBaseAnimation @ 0x005f53c0

Look for the main game loop's timing — how deltaTime is calculated
and passed to the animation update.
```

---

## TASK 10: Complete Node Structure Verification (HIGH)

### Goal
The "Node" is the shared base structure for BONE, HELP, LITE, ATCH, PRE2, RIBB, EVTS, CAMS. Verify its exact layout.

### Current Assumption
```
Node:
  uint32 inclusiveSize        // total size of node including this field
  char[0x50] name             // 80-byte null-terminated string
  int32 objectId              // unique ID for this node
  int32 parentId              // parent node ID (-1 for root)
  uint32 flags                // node flags (billboard, etc.)
  // Followed by animation sub-chunks (KGTR, KGRT, KGSC) until inclusiveSize reached
```

### Questions
1. **Is `inclusiveSize` measured from the start of the size field itself?** i.e., `nodeEnd = positionOfSizeField + inclusiveSize`?
2. **Total fixed header size = 4 + 80 + 4 + 4 + 4 = 96 bytes?** Confirm there are no extra fields.
3. **Are the animation sub-chunks (KGTR/KGRT/KGSC) optional?** A node with no animation would have `inclusiveSize = 96` (just the header)?
4. **Can animation sub-chunks appear in any order?** Or is it always KGTR → KGRT → KGSC?
5. **Are there other sub-chunk types that can appear inside a Node?** e.g., KATV (visibility), or any Alpha-specific tags?

### Key Function
```
IAnimCreateObjects (called from AnimBuild)
Look for the node parsing code that reads the shared header.
All node types (bone, helper, light, etc.) should share this code path.
```

---

## TASK 11: D3D8 Rendering Path — How Animated Geometry Reaches the Screen (MEDIUM)

### Goal
Trace from bone matrices → vertex transformation → D3D8 draw call.

### Questions
1. **Does the client use software vertex processing (CPU skinning)?** D3D8 on 2003 hardware likely uses CPU transforms.
2. **Where are transformed vertices stored?** In a dynamic vertex buffer? System memory array?
3. **Is there a `TransformVertices()` or `SkinMesh()` function?** Find it and document its signature.
4. **Does the client use `DrawIndexedPrimitive` or `DrawPrimitive`?** And with what vertex format (FVF flags)?

### Key Functions
```
Look for D3D8 API calls:
  IDirect3DDevice8::DrawIndexedPrimitive
  IDirect3DDevice8::DrawPrimitive
  IDirect3DDevice8::SetTransform
  IDirect3DDevice8::SetStreamSource

Trace backwards from the draw call to find where vertex data is prepared.
```

---

## TASK 12: Hex Dump Verification — DireWolf.mdx (VALIDATION)

### Goal
Use a known MDX file to validate our parsing against the binary's expectations.

### File
`Creature/DireWolf/DireWolf.mdx` — confirmed: SEQS count=17, entry size=140, first sequence "Stand - Var1"

### Steps
1. Open the file in a hex editor
2. Find the BONE chunk (FourCC "BONE" = `42 4F 4E 45`)
3. Read the first bone entry manually:
   - Note the nodeSize value
   - Read the name (should be readable ASCII)
   - Read objectId, parentId, flags
   - Check if KGTR/KGRT/KGSC sub-chunks follow
   - Note the keyCount and first few keyframe values
4. Compare with what our parser produces (add verbose logging)
5. Find the GNDX chunk inside the first GEOS entry
   - Note how many bytes per vertex (1 or 4?)
   - Cross-reference with MTGC and MATS

### Expected Output
A byte-by-byte annotation of the first BONE entry showing exactly where each field starts and what value it contains.

---

## Priority Order

| Priority | Task | Why |
|----------|------|-----|
| 🔴 CRITICAL | Task 3 (Track Layout) | If track header is size-not-count, ALL keyframe data is garbage |
| 🔴 CRITICAL | Task 2 (BONE Layout) | Node size interpretation affects everything |
| 🔴 CRITICAL | Task 5 (Matrix Composition) | Wrong order = wrong transforms even with correct data |
| 🔴 CRITICAL | Task 6 (Vertex Skinning) | Without this, bones move but vertices don't |
| 🟡 HIGH | Task 1 (Loading Pipeline) | Chunk ordering and initialization |
| 🟡 HIGH | Task 9 (Time Units) | Wrong units = animation plays at wrong speed or not at all |
| 🟡 HIGH | Task 10 (Node Structure) | Shared base for all animated objects |
| 🟡 HIGH | Task 4 (SEQS Layout) | Already partially confirmed but need full verification |
| 🟢 MEDIUM | Task 7 (Global Sequences) | Affects looping ambient animations |
| 🟢 MEDIUM | Task 8 (Geoset Animation) | Affects visibility toggling per sequence |
| 🟢 MEDIUM | Task 11 (D3D8 Path) | Confirms CPU vs GPU skinning approach |
| 🔵 VALIDATION | Task 12 (Hex Dump) | Ground truth verification |

---

## Quick-Reference: Known Addresses

### MDX Loading
| Function | Address |
|----------|---------|
| `BuildModelFromMdxData` | `0x00421fb0` |
| `MdxReadAnimation` | `0x004221b0` |

### Animation System
| Function | Address |
|----------|---------|
| `AnimCreate` | `0x0073a850` |
| `AnimBuild` | `0x0073cb10` |
| `AnimAddSequences` | `0x00754000` |
| `AnimAddGeosets` | `0x00754c00` |
| `AnimAddMaterialLayers` | `0x00755800` |
| `AnimBuildObjectIdTranslation` | `0x0073a4d0` |
| `AnimEnumObjects` | `0x0073e610` |

### Object Creation
| Function | Address |
|----------|---------|
| `AnimObjectCreateBone` | `0x0074d8e0` |
| `AnimObjectCreateHelper` | `0x0074d7a0` |
| `AnimObjectCreateLight` | `0x0074d800` |
| `AnimObjectCreateEmitter2` | `0x0074d970` |
| `AnimObjectCreateRibbon` | `0x0074d9e0` |

### Interpolation
| Function | Address |
|----------|---------|
| `InterpolateLinear` | `0x0075de20` |
| `InterpolateHermite` | `0x0075dc00` |
| `InterpolateBezier` | `0x0075de00` |

### Playback
| Function | Address |
|----------|---------|
| `ChooseAnimation` | `0x005fbd10` |
| `PlayBaseAnimation` | `0x005f53c0` |
| `UpdateBaseAnimation` | `0x005f56a0` |
| `AnimGetObjectTimeScale` | `0x0074b680` |
| `AnimGetObjectPosition` | `0x00741c00` |
| `AnimLockObjectSequence` | `0x00741840` |
| `AnimEnableBlending` | `0x00741590` |

### FourCC Constants
| Chunk | Hex | ASCII |
|-------|-----|-------|
| BONE | `0x454E4F42` | `BONE` |
| SEQS | `0x53514553` | `SEQS` |
| GLBS | `0x53424C47` | `GLBS` |
| PIVT | `0x54564950` | `PIVT` |
| GEOS | `0x534F4547` | `GEOS` |
| ATSQ | `0x51535441` | `ATSQ` |
| KGTR | `0x5254474B` | `KGTR` |
| KGRT | `0x5452474B` | `KGRT` |
| KGSC | `0x4353474B` | `KGSC` |
| KGAO | `0x4F41474B` | `KGAO` |
| KGAC | `0x4341474B` | `KGAC` |
| GNDX | `0x58444E47` | `GNDX` |
| MTGC | `0x4347544D` | `MTGC` |
| MATS | `0x5354414D` | `MATS` |
| BIDX | `0x58444942` | `BIDX` |
| BWGT | `0x54475742` | `BWGT` |

---

## What We Need Back

For each task, provide:
1. **Ghidra pseudocode** of the relevant function(s)
2. **Exact struct layouts** with byte offsets
3. **Field sizes and types** (uint32, int32, float, etc.)
4. **Any Alpha-specific differences** from standard WC3 MDX format
5. **Concrete values** from tracing DireWolf.mdx through the code path if possible

The single most impactful finding would be confirming whether the KGTR/KGRT/KGSC sub-chunk header is `[tag + keyCount]` or `[tag + byteSize]` — if we have this wrong, every keyframe we read is garbage.
