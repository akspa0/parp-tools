# MHDR — ADT Root Header

## Summary
Root ADT header that stores offsets to top-level ADT chunks.

## Parent Chunk
Root-level ADT chunk.

## Build 0.7.0.3694 Evidence
- Confirmed in `FUN_006bd840` (`mIffChunk->token=='MHDR'`).
- Header is accessed as an array of 32-bit offsets (`piVar1[3..10]`).

## Structure — Build 0.7.0.3694 (confirmed offsets)
Offsets below are relative to MHDR data base (`mhdr + 0x10` in parser math):

| Header Index | Target Chunk | Runtime Pointer Assignment |
|---|---|---|
| `piVar1[3]` | `MCIN` | `param+0x674` |
| `piVar1[4]` | `MTEX` | used for texture load call |
| `piVar1[5]` | `MMDX` | `param+0x68C` |
| `piVar1[6]` | `MMID` | `param+0x694` |
| `piVar1[7]` | `MWMO` | `param+0x690` |
| `piVar1[8]` | `MWID` | `param+0x698` |
| `piVar1[9]` | `MDDF` | `param+0x67C` |
| `piVar1[10]` | `MODF` | `param+0x680` |

## Confidence
- **High**
