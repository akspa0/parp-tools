# Data Model: MCAL Alpha Map Decode Correctness

**Date**: 2026-09-01

All types in `WowViewer.Core.IO.Maps`. No type here requires a graphics context (FR-008).

## `AdtAlphaEncoding` (enum)

The format's encodings, and only those.

| Member | Payload |
|---|---|
| `Compressed` | RLE control-byte stream |
| `Packed4Bit` | 2048 bytes, two texels per byte |
| `FullByte` | 4096 bytes, one texel per byte |

There is no `Inferred`, `Guessed`, or `Residual` member. FR-002 is enforced by the absence of
a representable state for manufactured data.

## `AdtAlphaDecodeRule` (readonly record struct)

Resolved **before** reading any bytes, from era profile plus format flags (FR-004).

| Field | Type | Source |
|---|---|---|
| `Encoding` | `AdtAlphaEncoding` | `MCLY` compression flag; else era profile + `MPHD` big-alpha mask |
| `ApplyEdgeFixup` | `bool` | MCNK `doNotFixAlphaMap` (`0x8000`) inverted |
| `ExpectedBytes` | `int?` | 2048 / 4096; `null` for `Compressed`, whose length is data-driven |
| `Justification` | `string` | which profile and flags produced this rule — carried so a report can explain itself |

**Rule**: constructed only from era + flags. It never takes a neighbouring layer's offset as
input. That is the design flaw research.md R4 identifies.

## `AdtLayerDecodeOutcome`

| Field | Type | Notes |
|---|---|---|
| `LayerIndex` | `int` | |
| `Rule` | `AdtAlphaDecodeRule` | what was attempted |
| `Alpha` | `byte[]?` | 4096 texels on success; **null on failure, always** |
| `BytesConsumed` | `int` | 0 on failure |
| `Failure` | `AdtAlphaDecodeFailure?` | non-null exactly when `Alpha` is null |

**Validation**: `Alpha` and `Failure` are mutually exclusive and exactly one is set. Same
structural guarantee as spec 198's `PermutationRequest` — the fabrication this spec removes
was only possible because a failed decode could be replaced by a plausible array.

## `AdtAlphaDecodeFailure` (enum)

`OffsetOutOfRange`, `TruncatedPayload`, `CompressedStreamOverrun`, `NoAlphaFlag`,
`UnexplainedEncoding`.

`NoAlphaFlag` is a legitimate absence, not an error — 0.5.3 chunks produce it and must not be
counted as failures (spec edge case, FR-007).

## `AdtChunkDecodeOutcome`

| Field | Type | Notes |
|---|---|---|
| `Layers` | `IReadOnlyList<AdtLayerDecodeOutcome>` | |
| `McalPayloadBytes` | `int` | |
| `BytesAccountedFor` | `int` | sum of layer `BytesConsumed` |
| `UnexplainedBytes` | `int` | `McalPayloadBytes - BytesAccountedFor` |

`UnexplainedBytes == 0` is the per-chunk form of US2's gate (SC-002). A non-zero value is the
honest signal that the rule does not yet explain the file.

## `AdtAlphaDecodeReport`

Aggregatable across a corpus (FR-009).

| Field | Type |
|---|---|
| `ChunksByEra` | `IReadOnlyDictionary<string, int>` |
| `RuleCounts` | `IReadOnlyDictionary<(string Era, AdtAlphaEncoding), int>` |
| `FailureCounts` | `IReadOnlyDictionary<(string Era, AdtAlphaDecodeFailure), int>` |
| `FullyAccountedChunks` | `int` |
| `UnexplainedChunks` | `int` |
| `UnexplainedFiles` | `IReadOnlyList<string>` |

`UnexplainedFiles` is SC-006: the remaining unknown expressed as a list of paths rather than
a confidence level.

## Relationships

```
Era profile + MCLY/MPHD/MCNK flags
        │
        ▼
AdtAlphaDecodeRule ──▶ AdtMcalDecoder ──▶ AdtLayerDecodeOutcome
                              │                    │
                              ▼                    ▼
                      AdtChunkDecodeOutcome ──▶ AdtAlphaDecodeReport
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
   Renderer                                Dataset harvest
   (same bytes, FR-006)                    (same bytes, FR-006)
```
