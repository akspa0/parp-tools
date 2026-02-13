# C4QuaternionCompressed — Exact Decompression Formula (Ghidra Verified)

Source functions (WoW Alpha 0.5.3 / build 3368):
- `NTempest::C4QuaternionCompressed::Set` @ `0x0075bad0`
- `NTempest::C4QuaternionCompressed::GetW` @ `0x0075ba30`
- `NTempest::C4QuaternionCompressed::operator C4Quaternion` @ `0x0074d690`
- Rotation key reader uses same unpack path in `AnimObjectSetRotation` @ `0x0074e930`

---

## Packed Layout

`C4QuaternionCompressed` is 64 bits:
- `data0` = low 32 bits
- `data1` = high 32 bits

Three signed components are packed as 21-bit signed integers (`xq`, `yq`, `zq`) and reconstructed as floats.

---

## Exact Unpack Equations

Let `u0 = data0 (uint32)`, `u1 = data1 (uint32)`.

```c
// arithmetic right shifts are required (signed)
int32 xq = ((int32)u1) >> 10;
int32 yq = ((int32)((u1 << 22) | (u0 >> 10))) >> 11;
int32 zq = ((int32)(u0 << 11)) >> 11;

float x = (float)xq * 0x1.0p-21f; // 2^-21 = 4.76837158203125e-7
float y = (float)yq * 0x1.0p-20f; // 2^-20 = 9.5367431640625e-7
float z = (float)zq * 0x1.0p-20f; // 2^-20
```

These constants correspond to what decomp showed as:
- `___real_35000000` = `2^-21`
- `___real_35800000` = `2^-20`

---

## Exact `w` Reconstruction

From `GetW` (`0x0075ba30`):

```c
float s = x*x + y*y + z*z;

// epsilon from decomp threshold 0x35800000f == 2^-20
if (fabsf(s - 1.0f) < 0x1.0p-20f)
    return 0.0f;

// otherwise positive branch only
return sqrtf(1.0f - s);
```

Important behavior:
- Reconstructed `w` is non-negative (canonicalized hemisphere).
- If norm is already very close to 1, code snaps `w` to `0`.

---

## Field Order in Returned Quaternion

`operator C4Quaternion` writes the unpacked XYZ and reconstructed W into `C4Quaternion` fields. In practical usage, this matches the same unpack path used by interpolation/read routines.

Use this as implementation intent:

```c
quat.x = x;
quat.y = y;
quat.z = z;
quat.w = reconstructedW;
```

(Names in decompiled struct fields are noisy; equations above are the reliable part.)

---

## Compression Side (for reference, from `Set`)

`Set` applies a sign canonicalization tied to input `w` and quantizes with:
- `x * 2^21`
- `y * 2^20`
- `z * 2^20`

Then packs with this exact bit logic:

```c
uint32 data0 = ((yq & 0x1FFFFF) << 21) | (zq & 0x1FFFFF);
uint32 data1 = (((uint32)(xq) >> 11) << 21)
             | (((xq * 0x200000) | (yq & 0x1FFFFF)) >> 11);
```

This is included only to mirror the binary exactly; for your parser, decompression equations are the critical requirement.

---

## Drop-in Decompress Pseudocode

```c
typedef struct { uint32_t data0, data1; } C4QuaternionCompressed;
typedef struct { float x, y, z, w; } Quat;

static inline Quat Decompress(const C4QuaternionCompressed c)
{
    int32_t xq = ((int32_t)c.data1) >> 10;
    int32_t yq = ((int32_t)((c.data1 << 22) | (c.data0 >> 10))) >> 11;
    int32_t zq = ((int32_t)(c.data0 << 11)) >> 11;

    float x = (float)xq * 0x1.0p-21f;
    float y = (float)yq * 0x1.0p-20f;
    float z = (float)zq * 0x1.0p-20f;

    float s = x*x + y*y + z*z;
    float w = (fabsf(s - 1.0f) < 0x1.0p-20f) ? 0.0f : sqrtf(1.0f - s);

    Quat q = { x, y, z, w };
    return q;
}
```

---

If you want, next step is I can generate the exact C# `ReadCompressedQuaternion()` implementation and wire it into your `KGRT` parser call-site(s).