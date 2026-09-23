# Contract: Canonical MCAL Decoder

**Date**: 2026-09-01

In-process contract between `AdtMcalDecoder` and every consumer. Stated as behaviour so
`AdtMcalDecoderTests` can pin it directly.

## C1 — Failure is representable; fabrication is not

```
AdtLayerDecodeOutcome is valid  ⟺  (Alpha is null) XOR (Failure is null)
```

There is no constructor, factory, or code path that produces an outcome carrying alpha the
decoder did not read from MCAL bytes. FR-002 is a type invariant, not a convention — the
fabrication being removed was possible precisely because "failed decode" and "here is an
array" were not mutually exclusive states.

## C2 — The rule is chosen before the bytes are read

```
ResolveRule(eraProfile, mclyFlags, mphdFlags, mcnkFlags) → AdtAlphaDecodeRule
```

Takes no offset, no payload length, and no neighbouring layer. Given identical era and flags
it returns an identical rule, regardless of file contents (FR-004). This is what makes the
decoder testable without a file and what forecloses the span-inference flaw.

## C3 — One owner

```
Repository-wide, exactly one implementation reads MCAL bytes into texels.
```

`Mcal`, `AlphaMapService`, `DecodeLayerBySpan` and the `StandardTerrainAdapter` fallback loop
either delegate or cease to exist (FR-001, SC-001).

## C4 — Same input, same bytes, every caller

```
Decode(mcal, rule, layer) is a pure function of its arguments.
```

The renderer and the harvest, given the same chunk and era inputs, receive byte-identical
alpha (FR-006, SC-005). No caller-specific defaults — in particular, no caller supplies its
own `bigAlpha` constant, which is the current divergence (research.md R2).

## C5 — Accounting is reported, never reconciled

```
UnexplainedBytes = McalPayloadBytes - Σ BytesConsumed
```

The decoder reports this value. It does not adjust a rule, extend a layer, or consume a
remainder to drive it to zero. A non-zero remainder is a finding (FR-005, SC-006), and the
temptation to absorb it is exactly how span inference became load-bearing.

## C6 — Absence is not failure

```
NoAlphaFlag ⇒ counted as absent, not as a failure
```

0.5.3 chunks carry no alpha. They must produce no alpha map, no error, and no entry in the
failure counts (FR-007). A decoder that reports the entire Alpha corpus as broken is as
useless as one that fabricates.

## C7 — Headless

```
No member of the decoder, the rule, or the report types touches a graphics context.
```

FR-008 — required for both the unit tests and the offline corpus sweep.
