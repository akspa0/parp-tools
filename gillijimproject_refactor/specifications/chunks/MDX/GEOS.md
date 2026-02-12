# GEOS — Geoset Geometry

## Summary
Contains mesh geometry data (vertices, normals, faces, UV linkage) for one geoset.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_0044d730` token `0x534f4547` (`GEOS`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | geosetBytes | GEOS section size |
| 0x04 | uint32 | geosetCount | Number of geosets |
| 0x08 | byte[] | geosetRecords | Variable-size records; each starts with `bytesThisGeo` |

Within each parsed geoset payload, the loader validates subchunk order:
- `VRTX` (`0x58545256`)
- `NRMS` (`0x534d524e`)
- `UVAS` (`0x53415655`), with expected channel count `1`
- `PTYP` (`0x50595450`)
- `PCNT` (`0x544e4350`)
- `PVTX` (`0x58545650`)

## Confidence
- Geoset framing and subchunk sequence: **High**
- Full semantic naming of every field: **Medium**
