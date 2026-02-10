# MDX Rotation Analysis - Critical Findings

## The Rotation Mystery: Single Angle vs 3-Axis Euler

### Key Finding: Both MDX and WMO Use Single-Angle Z-Rotation

Analysis of [`CWorld::ObjectCreate`](specifications/outputs/053/mdx-alignment/01-overview.md) reveals that both **MDX doodads** and **WMO objects** use the **same rotation format**:

```cpp
uint __fastcall CWorld::ObjectCreate(
    char *param_1,      // Model path
    C3Vector *param_2,  // Position (X, Y, Z)
    float param_3,      // ROTATION ANGLE (single float, degrees!)
    int param_4,        // Flags
    int param_5,        // Snap flag
    __uint64 param_6    // Sound handle
)
```

**Both paths use the same `float param_3` as the rotation angle!**

### Matrix Construction is Identical

#### [`CMap::CreateDoodadDef`](specifications/outputs/053/mdx-alignment/01-overview.md) (MDX):
```cpp
// Initialize as identity
mat = IdentityMatrix();

// Translate to position
NTempest::C44Matrix::Translate(this, position);

// Rotate about Z-axis by single angle
C3Vector axis = {0, 0, 1};  // Z-axis
NTempest::C44Matrix::Rotate(this, angle, &axis, true);
```

#### [`CMap::CreateMapObjDef`](specifications/outputs/053/mdx-alignment/01-overview.md) (WMO):
```cpp
// Identical code!
mat = IdentityMatrix();
NTempest::C44Matrix::Translate(this, position);
C3Vector axis = {0, 0, 1};
NTempest::C44Matrix::Rotate(this, angle, &axis, true);

// WMO additionally computes inverse matrix
NTempest::C44Matrix::AffineInverse(this, &local_74);
```

### The Puzzle: MDDF Stores 3 Rotation Values

According to the file format:
```
MDDF Entry (36 bytes):
Offset  Size  Field
0x14    4     rotation.X (float) — degrees
0x18    4     rotation.Z (float) — degrees
0x1C    4     rotation.Y (float) — degrees
```

**But the placement code uses only a single angle!**

## Hypotheses for Rotation Conversion

### Hypothesis 1: The "X" rotation is the primary one
The X rotation might be the only one used (trees/grass might only need Y-axis rotation).

### Hypothesis 2: Euler angle conversion happens earlier
The 3 values are converted to a single angle BEFORE reaching `ObjectCreate()`.

### Hypothesis 3: Only the first value is used
The client might only read `rotation[0]` and ignore the other two.

### Hypothesis 4: Different coordinate convention
The file stores (X, Z, Y) but the client converts to Z-only.

## Path-Based Rendering Analysis

### Search Results: No Direct Path-Based Rendering Found

Extensive search for path-based rendering decisions (e.g., `"World\NoDxt\Detail\"`):
- **`No functions found`** matching "NoDxt", "Detail", or similar directory patterns
- **`EnableDoodadFullAlpha`** applies to ALL doodads, not path-specific
- **`IModelEnableFullAlpha`** modifies material blending, not path-dependent

### However, Path IS Used For:

1. **WMO Detection**: [`ObjectCreate`](specifications/outputs/053/mdx-alignment/01-overview.md):
   ```cpp
   pcVar3 = SStrStr(param_1, s__wmo);
   if (pcVar3 == (char *)0x0) {
       // Create doodad (MDX)
   } else {
       // Create WMO
   }
   ```

2. **Model Loading**: Path determines which loader handles the file:
   - `.mdx` → MDX loader
   - `.wmo` → WMO loader
   - `.mdl` → MDL loader (Warcraft III format)

3. **Caching**: Path used as cache key for loaded models

### Material/Texture Assignment

Materials appear to be loaded from:
- **MDX/MDL model data**: Internal material definitions
- **Texture references**: Models reference textures by path
- **Database lookups**: Some materials from DBC files

**No evidence of directory-based material override in Alpha 0.5.3.**

## Next Steps for Rotation Analysis

1. **Find MDDF chunk parsing**: Where are the 3 rotation values read?
2. **Find rotation conversion**: How are 3 values converted to 1 angle?
3. **Check C44Matrix::Rotate**: Does it handle axis-angle or Euler?
4. **Test with sample data**: Compare raw MDDF vs. actual rendered angle

## Related Functions

| Function | Purpose |
|----------|---------|
| `CWorld::ObjectCreate` | Main entry point for placement |
| `CMap::CreateDoodadDef` | Creates MDX doodad with matrix |
| `CMap::CreateMapObjDef` | Creates WMO with matrix |
| `CMap::UpdateDoodadDef` | Updates doodad transform |
| `SStrStr` | Checks if path contains ".wmo" |

## Conclusion

**The MDX placement uses single-angle Z-rotation, same as WMO.** The 3 rotation values in MDDF must be converted before reaching the placement functions. The conversion location is not yet found - this is the critical missing piece.

Path-based rendering (transparency, blend modes) was **not found** in this Alpha build. Models render based on their material data, not directory location.
