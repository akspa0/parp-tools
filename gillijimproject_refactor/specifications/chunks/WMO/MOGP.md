# MOGP — WMO Group Header and Container

## Summary
Defines one WMO group and contains subordinate geometry/material/liquid chunks.

## Parent Chunk
Root-level WMO group region.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Group token validation in WMO group parser |
| 3.3.5 parser code (repo) | Uses 68-byte MOGP header before subchunks |
| 0.7.0.3694 | Inferred v14 continuity; exact offsets require direct confirmation |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | Group flags (`???` full bit map in 0.7) |
| 0x04 | ... | ... | Additional group metadata (`???`) |

Observed lineage note: group subchunk parsing commonly begins after a **68-byte header** in downstream tooling; verify for this exact build.

## Confidence
- Role and presence: **High**
- Full field map: **Medium/Low**
