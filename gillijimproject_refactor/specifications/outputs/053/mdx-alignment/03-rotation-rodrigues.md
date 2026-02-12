# MDX Rotation Implementation - Rodrigues Formula

## The Complete Rotation Pipeline

### 1. Axis-Angle Rotation Function

The client uses **Rodrigues' rotation formula** for all rotations:

```cpp
/* NTempest::C34Matrix::Rotation */
C34Matrix * __fastcall
NTempest::C34Matrix::Rotation(
    C34Matrix *__return_storage_ptr__,
    float param_1,          // Angle in RADIANS
    C3Vector *param_2,      // Rotation axis
    bool param_3            // Is axis already normalized?
)
```

**Key Mathematical Operations:**
```cpp
// Extract axis components
axis_.x = param_2->x;
axis_.y = param_2->y;
axis_.z = param_2->z;

// Normalize axis if not already normalized
if (!param_3) {
    float magnitude = SQRT(axis.x² + axis.y² + axis.z²);
    axis = axis / magnitude;
}

// Compute sin/cos of angle (ANGLE MUST BE IN RADIANS!)
fVar4 = fsin(angle);  // sin(θ)
fVar5 = fcos(angle);  // cos(θ)

// Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
// Where K is the cross-product matrix of the axis vector
__return_storage_ptr__->a0 = axis.x * axis.x * (1-cos) + cos;
__return_storage_ptr__->a1 = axis.x * axis.y * (1-cos) + axis.z * sin;
__return_storage_ptr__->a2 = axis.x * axis.z * (1-cos) - axis.y * sin;
__return_storage_ptr__->b0 = axis.x * axis.y * (1-cos) - axis.z * sin;
__return_storage_ptr__->b1 = axis.y * axis.y * (1-cos) + cos;
__return_storage_ptr__->b2 = axis.y * axis.z * (1-cos) + axis.x * sin;
// ... continues for c row
```

### 2. Matrix Rotation Application

```cpp
/* C34Matrix::Rotate - applies rotation to existing matrix */
void __thiscall NTempest::C34Matrix::Rotate(
    C34Matrix *this,       // Matrix to modify
    float param_1,         // Angle in RADIANS  
    C3Vector *param_2,     // Rotation axis
    bool param_3           // Normalized?
) {
    C34Matrix rot;
    C34Matrix temp;
    
    // Create rotation matrix from axis-angle
    Rotation(&rot, angle, axis, normalized);
    
    // Multiply: result = rotation × current_matrix
    // This applies rotation AFTER the current transform
    operator*(&temp, rot, this);
    
    // Copy result back
    this = temp;
}
```

### 3. Doodad Placement Matrix Construction

```cpp
void __fastcall CMap::CreateDoodadDef(
    CMapDoodadDef *this,
    C3Vector *position,     // (X, Y, Z) in WoW coords
    float rotation_angle,    // ANGLE IN DEGREES!
    int flags
) {
    // Step 1: Identity matrix
    mat = Identity();
    
    // Step 2: Translation
    C44Matrix::Translate(&mat, position);
    
    // Step 3: Z-axis rotation
    C3Vector axis = {0, 0, 1};  // Z-axis
    C44Matrix::Rotate(&mat, rotation_angle, &axis, true);
}
```

## CRITICAL: Degree-to-Radian Conversion

**THE MISSING PIECE!** The rotation function expects **RADIANS** but the MDDF file stores **DEGREES**.

Where does the conversion happen?

### Possible Conversion Points:

1. **In the Rotation function itself** - No, it uses fsin/fcos directly
2. **Before calling Rotate** - Most likely!
3. **In the chunk parser** - Possible
4. **Implicit conversion** - Not in the C++ code

### What We've Searched (Not Found):
- `DegToRad` function
- Direct multiplication by π/180
- Any explicit conversion code

### What We've Found:
The angle is passed as `float param_3` directly to `Rotate()`. The conversion must happen before this point.

## The Mystery: 3 MDDF Rotations → 1 Angle

Recall from the analysis brief:
```
MDDF Entry (36 bytes):
0x14    4     rotation.X (float) — degrees
0x18    4     rotation.Z (float) — degrees  
0x1C    4     rotation.Y (float) — degrees
```

But `CreateDoodadDef` only takes **ONE** angle!

### Hypotheses:

1. **Only rot[0] (X) is used** - The other two might be unused or for future expansion
2. **Quaternion conversion** - 3 Euler angles → quaternion → axis-angle
3. **Client bug/oversight** - Alpha client might only use first rotation
4. **Different format in Alpha** - Maybe Alpha stored single rotation despite spec saying 3

## Working WMO Transform (For Reference)

From the analysis brief:
```cpp
transform = RotZ(180°) * RotX(rx) * RotY(ry) * RotZ(rz) * Translate(pos)
```

But Ghidra shows the client uses:
```cpp
mat = Identity();
mat = Translate(pos);
mat = RotateZ(single_angle);
```

**The transforms don't match!**

### Possible Explanations:

1. **Different interpretation** - The wiki formula might be for a different coordinate system
2. **OpenGL vs D3D** - The wiki formula accounts for GL/D3D differences
3. **Inverted multiplication order** - Matrix multiplication order matters

## Required: Degree-to-Radian + 3-to-1 Conversion

Based on Ghidra analysis, the client code appears to:
1. Read 3 rotation values from MDDF
2. Convert to a single angle somehow
3. Convert degrees to radians
4. Apply axis-angle rotation about Z-axis

**This conversion code has not been located in the binary.**

## Next Steps

1. [ ] Find MDDF chunk parsing function
2. [ ] Locate degree-to-radian conversion
3. [ ] Find 3-to-1 rotation conversion
4. [ ] Verify with test data from Alpha 0.5.3 client
5. [ ] Document exact formula for renderer implementation

## Related Functions Analyzed

| Function | Address | Purpose |
|----------|---------|---------|
| `C34Matrix::Rotation` | 0x00493eb0 | Axis-angle to rotation matrix (Rodrigues) |
| `C34Matrix::Rotate` | 0x004941d0 | Apply rotation to matrix |
| `CMap::CreateDoodadDef` | 0x00680300 | Construct doodad placement matrix |
| `CMap::CreateMapObjDef` | 0x00680f50 | Construct WMO placement matrix |
| `fsin`/`fcos` | - | Math library (radians input) |
