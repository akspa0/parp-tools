# Data Model: Spec 263 — 1.60 MCCV Terrain Shadow Ground-Truth Validation & Synthesis

## 1. MCCV Vertex Layout (145 Vertices per MCNK Chunk)

In WoW ADTs, each MCNK chunk contains 145 vertices organized as:
- **Outer Grid (9x9 = 81 vertices)**:
  - Coordinate spacing: $d = 4.1666\,\text{yards}$ ($33.333 / 8$).
  - Vertex index range: `[0 .. 80]`.
  - $(u, v) \in \{0, 1/8, 2/8, \dots, 1\} \times \{0, 1/8, 2/8, \dots, 1\}$.
- **Inner Grid (8x8 = 64 vertices)**:
  - Coordinate spacing: offset by $(d/2, d/2)$ from outer grid.
  - Vertex index range: `[81 .. 144]`.
  - $(u, v) \in \{1/16, 3/16, \dots, 15/16\} \times \{1/16, 3/16, \dots, 15/16\}$.

## 2. In-Memory and On-Disk Binary Format

```
Offset   Size (bytes)   Type              Description
-----------------------------------------------------------------------------
0x00     4              uint32 (FourCC)   'VCCM' ('MCCV' in Little-Endian)
0x04     4              uint32            Size (145 * 4 = 580 bytes)
0x08     580            byte[580]         145 BGRA vertex colors (B, G, R, A)
```

In 1.60 modern terrain shading:
- Each vertex color channel $[0, 255]$ maps to $[0.0, 1.0]$.
- The terrain shader evaluates $\text{Light} \times (2.0 \times \text{MCCV})$, where $127 \approx 0.5$ represents neutral $1.0\times$ lighting multiplier.
- Channels $B, G, R$ are identical (achromatic) for terrain shadow and occlusion.

## 3. Comparison Metrics Schema

```json
{
  "tile_coords": [30, 30],
  "ncc": 0.842,
  "mae": 0.041,
  "ssim": 0.887,
  "ridge_coincidence_pct": 82.5,
  "passed": true
}
```
