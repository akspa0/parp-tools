# PD4

From wowdev

Jump to navigation Jump to search

PD4 files are the PM4 equivalent for WMOs. There is one file per root WMO. They are not supposed to be shipped to the client and are used by the server only.

## Contents

* 1 MVER
* 2 MCRC
* 3 MSHD
* 4 MSPV
* 5 MSPI
* 6 MSCN
* 7 MSLK
* 8 MSVT
* 9 MSVI
* 10 MSUR

# MVER

```
uint32_t version; enum { version_48 = 48, // seen in (6.0.1.18297), (6.0.1.18443) };
```

# MCRC

```
uint32_t _0x00; // Always 0 in version_48.u
```

# MSHD

```
struct { uint32_t _0x00; uint32_t _0x04; uint32_t _0x08; uint32_t _0x0c[5]; // Always 0 in version_48, likely placeholders.u } header;
```

# MSPV

```
C3Vectori msp_vertices[];
```

# MSPI

```
uint32_t msp_indices[]; // index into #MSPV
```

# MSCN

Not related to #MSPV and #MSLK: Seen to have one entry while #MSPV and #MSLK has none.

```
C3Vectori mscn[]; // n ≠ normals.u
```

# MSLK

```
struct { uint8_t _0x00; // earlier documentation has this as bitmask32 flagsu uint8_t _0x01; uint16_t _0x02; // Always 0 in version_48, likely padding.u uint32_t _0x04; // An index somewhere.u int24_t MSPI_first_index; // -1 if _0x0b is 0 uint8_t MSPI_index_count; uint32_t _0x0c; // Always 0xffffffff in version_48.u uint16_t _0x10; uint16_t _0x12; // Always 0x8000 in version_48.u } mslk[];
```

# MSVT

```
C3Vectori msvt[]; // t ≠ tangents. vt = vertices?u
```

For some reason the values are ordered YXZ, and must have the following formulas ran on them to get the ingame coordinates.

```
worldPos.y = 17066.666 - position.y; worldPos.x = 17066.666 - position.x; worldPos.z = position.z / 36.0f; // Divide by 36 is used to convert the internal inch height to yards u
```

# MSVI

Likely not triangles but quads, or an n-gon described somewhere, possibly #MSUR where _0x01 is count and _0x14 is offsetu.

```
uint32_t msv_indices[]; // index into #MSVT
```

# MSUR

```
struct { uint8_t _0x00; // earlier documentation has this as bitmask32 flagsu uint8_t _0x01; // count of indices in #MSVIu uint8_t _0x02; uint8_t _0x03; // Always 0 in version_48, likely padding.u float _0x04; float _0x08; float _0x0c; float _0x10; uint32_t MSVI_first_index; uint32_t _0x18; uint32_t _0x1c; } msur[];