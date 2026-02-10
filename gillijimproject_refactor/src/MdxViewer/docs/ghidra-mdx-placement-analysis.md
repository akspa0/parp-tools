# Ghidra Reverse Engineering Brief: MDX/M2 Doodad Placement Transform in WoW Alpha 0.5.3 (Build 3368)

## Objective

Reverse-engineer the **exact matrix construction** the WoW 0.5.3 Alpha client uses to place MDX (M2) doodad models in the 3D world. We need the complete transform pipeline from MDDF placement data + MDX model data → final world-space vertex positions. We have been unable to reproduce correct placement from file data alone and need the ground truth from the binary.

## Background & Problem Statement

We are building a renderer for WoW Alpha 0.5.3 terrain and objects. **WMO (World Map Object) placement works correctly**, but **MDX/M2 doodad placement is wrong** — models appear twisted/tilted on incorrect axes. These are small props like wooden boards, fences, stairs, and trees that are placed via MDDF entries in ADT files.

The models "swivel on an axis" — they rotate around some pivot but end up at wrong orientations. We suspect the MDX format has internal orientation data (pivot points, bone transforms, or model-space basis vectors) that the client uses *before* applying the MDDF world placement, and we haven't been able to figure out the correct combination.

We have tried:
- Simple Euler rotation: `RotX(rx) * RotY(ry) * RotZ(rz)` — wrong
- Wiki formula: `RotY(rot[1]-270) * RotZ(-rot[0]) * RotX(rot[2]-90)` — wrong
- Various axis remappings for our coordinate system — all wrong
- The models are consistently twisted/tilted, suggesting a systematic error in the rotation pipeline

## What We Need From Ghidra Analysis

### Priority 1: The MDDF → World Matrix Construction

Find the function that reads MDDF placement entries and constructs the world transform matrix. This is the critical path:

1. **How does the client read MDDF rotation values?** Are they Euler angles (degrees)? Which axes? What order?
2. **What is the exact matrix multiplication order?** (Scale × Rotation × Translation, or some other order?)
3. **What rotation convention is used?** (Intrinsic vs extrinsic Euler? Axis-angle? Quaternion intermediate?)
4. **Is there a coordinate system conversion step?** (The client renders in Direct3D left-handed coordinates)
5. **Are the rotation values modified before use?** (Negated? Offset by 90°/180°/270°? Swapped?)

### Priority 2: MDX Model-Space Transform

Find how the client loads an MDX model and whether any model-internal data affects world placement:

1. **MODL chunk bounds** — Does the client use the model's bounding box center as a pivot offset?
2. **PIVT (Pivot Points)** — Are pivot points applied to shift the model origin before world placement?
3. **Bone[0] / Root bone** — Is there a root bone transform that reorients the model?
4. **GEOS bounds** — Per-geoset bounds — are these used for anything placement-related?
5. **Is there a model-space basis change?** (MDX models might be authored Y-up or Z-up, and the client converts)

### Priority 3: The Rendering Pipeline

1. **What coordinate system does the client use internally?** (WoW wiki says X=north, Y=west, Z=up for world coords, but the D3D renderer may use something different)
2. **How does the client set up the model matrix for `DrawPrimitive` calls?** Find the `SetTransform(D3DTS_WORLD, ...)` call chain.
3. **Is there a global scene transform?** (Camera view matrix that implicitly converts coordinate systems)

## File Format Details

### MDDF Entry (36 bytes per entry in Alpha 0.5.3)

```
Offset  Size  Field
0x00    4     nameIndex (uint32) — index into MDNM string table
0x04    4     uniqueId (uint32)
0x08    4     position.X (float) — WoW world X
0x0C    4     position.Z (float) — WoW world Z (height)
0x10    4     position.Y (float) — WoW world Y
0x14    4     rotation.X (float) — degrees
0x18    4     rotation.Z (float) — degrees (note: Z before Y in file, same as position)
0x1C    4     rotation.Y (float) — degrees
0x20    2     scale (uint16) — 1024 = 1.0
0x22    2     flags (uint16)
```

**Key detail**: Position and rotation are stored as C3Vector with layout `(X, Z, Y)` in the file — the middle component is the vertical axis (height). This is the standard WoW C3Vector file layout.

### MDX Model Structure (relevant chunks)

```
MDLX (magic)
VERS — version (always 800 in Alpha)
MODL — model info:
  - Name (80 bytes, null-padded)
  - BoundsMin (C3Vector: 3 floats)
  - BoundsMax (C3Vector: 3 floats)
  - BlendTime (uint32)
SEQS — animation sequences (each has its own bounds)
PIVT — pivot points (C3Vector per bone/node)
BONE — bones (each has objectId, parentId, flags, translation/rotation/scaling tracks)
GEOS — geosets:
  - VRTX: vertex positions (C3Vector per vertex)
  - NRMS: vertex normals
  - UVBS: UV coordinates
  - PVTX: triangle indices
  - Per-geoset bounds (CMdlBounds: min, max, radius)
  - Per-animation extents
```

### WoW Coordinate System (World Space)

- **X** = North (increases going north)
- **Y** = West (increases going west)  
- **Z** = Up (increases going up)
- **Left-handed** (Direct3D convention)
- Map origin: `(17066.666, 17066.666, 0)` — center of tile grid

### What Works (WMO placement — for comparison)

WMO placement uses MODF entries with the same rotation format. Our working WMO transform is:

```
transform = RotZ(180°) * RotX(rx) * RotY(ry) * RotZ(rz) * Translate(pos)
```

Where:
- `rx, ry, rz` are the raw rotation values from MODF converted to radians
- Position is converted: `rendererX = 17066.666 - wowY`, `rendererY = 17066.666 - wowX`, `rendererZ = wowZ`
- `RotZ(180°)` compensates for triangle winding reversal (we swap CW→CCW indices for OpenGL)
- This is in System.Numerics row-major convention (A * B = apply A first, then B to vertex)

**WMOs render correctly with this transform.** MDX doodads use the same transform but appear twisted.

## Suggested Ghidra Search Strategy

### Entry Points to Find

1. **MDDF parser**: Search for the string "MDDF" or the FourCC `0x4644444D`. The function that processes this chunk will read the 36-byte entries and create placement structures.

2. **`CMapObj::LoadDoodadDef`** or similar — the function that takes an MDDF entry and creates a renderable doodad instance. This is where the transform matrix is built.

3. **`CM2Model::Render`** or **`CM2Instance::UpdateTransform`** — the function that sets up the D3D world matrix before drawing. Look for `IDirect3DDevice8::SetTransform` calls with `D3DTS_WORLD`.

4. **Matrix construction helpers**: Search for patterns like:
   - `sin`/`cos`/`sinf`/`cosf` calls near matrix construction
   - `D3DXMatrixRotationYawPitchRoll`
   - `D3DXMatrixRotationX/Y/Z`
   - `D3DXMatrixScaling`
   - `D3DXMatrixTranslation`
   - `D3DXMatrixMultiply`
   - Any 4x4 float array construction (16 consecutive float writes)

5. **String references**: Search for strings like:
   - `"MDDF"`, `"MODF"` — chunk parsers
   - `".mdx"`, `".m2"` — model loading
   - `"doodad"` — placement logic
   - `"CMapObjDef"`, `"CMapObj"`, `"CM2"` — class names (may appear in debug strings)

### Key Comparisons

Once you find the MDX placement transform, compare it with the WMO (MODF) placement transform. The wiki says they use "the same" rotation, but there may be subtle differences:
- Does MDX apply an additional model-space rotation that WMO doesn't?
- Does MDX use the PIVT/bone data to offset the origin?
- Is the MDX vertex data in a different coordinate basis than WMO vertex data?

## Expected Output

Please provide:

1. **The exact C pseudocode** for the function that builds the MDX world placement matrix from MDDF data
2. **The exact C pseudocode** for any model-space transform applied to MDX vertices before the world matrix
3. **The exact C pseudocode** for the equivalent WMO (MODF) placement for comparison
4. **Any D3D API calls** (`SetTransform`, `DrawPrimitive`, etc.) in the render path
5. **The coordinate system** used internally (is it the same as file coords, or converted?)
6. **Any constants** (90°, 180°, 270° offsets, axis negations, etc.) applied to rotation values

## Additional Context

- The binary is a **32-bit Windows PE** (WoW.exe, Alpha 0.5.3 build 3368)
- It uses **Direct3D 8** (not 9)
- The MDX format is Warcraft III MDX (not the later M2 format) — Alpha WoW reused the WC3 model format
- Models are loaded from MPQ archives
- The client was built with MSVC (likely VC6 or VC7)
- There may be debug symbols or RTTI information that helps identify classes

## Summary of the Mystery

The core question is: **What transform does the 0.5.3 client apply between "raw MDX vertex in model space" and "final vertex in world space"?**

We know it involves:
1. Some model-space adjustment (pivot? bone? bounds center offset?)
2. Scale from MDDF (uint16 / 1024)
3. Rotation from MDDF (3 floats, degrees, stored as X,Z,Y in file)
4. Translation from MDDF (3 floats, stored as X,Z,Y in file)

But we don't know the **exact order, axis mapping, and whether there's a model-internal pre-transform**. That's what we need from Ghidra.
