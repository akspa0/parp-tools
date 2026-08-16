# Phase 0 Research: M2 Reader Era Parity

**Status**: Partially complete. Section 1 is measured; sections 2–5 are open and gate Phase 1.

## 1. Measured baseline (2026-08-15)

Read from staged clients via `tools/inspect`. Configured root is the operator's client library; no
client path is baked into source.

| Declared | Build | Model | Outcome |
|---|---|---|---|
| `MDLX` | 0.5.3.3368 | `Creature\HighElf\HighElfMale_Warrior.mdx` | Works — 54 bones, 106 sequences, 21 helpers, 32 attachments, 128 pivot points, 14 chunks all known |
| `MD20 0x100` | 2.0.0.5610 | `CHARACTER\BloodElf\Male\BloodElfMale.m2` | `bones=0`; geometry unavailable, fails at bone index 10 |
| `MD20 0x100` | 2.0.0.5610 | `Character\NightElf\Male\NightElfMale.m2` | Identical failure, identical index |
| `MD20 0x107` | 3.0.1.8303 | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | Unhandled refusal: "2.x TBC era, not yet supported" |
| `MD20 0x108` | 3.3.0.10958 | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | **Works** — 151 bones, 155 sequences, 14 aliases, 7 global loops, geometry available, 375 bone lookups |
| `MD20 0x109`+ | 4.0.0.11927 | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | Unhandled failure reading camera records |

### Decisions taken from this baseline

**Decision**: The reference implementation is `MD20 0x108`, evidenced at 3.3.0.10958.
**Rationale**: It is the only `MD20` route measured reading a complete model — skeleton, sequences and
geometry — on this date.
**Alternatives considered**: "3.3.5 through 4.0.0" as stated in the request. Rejected by measurement:
the 4.0.0 beta fails, so `0x109`+ cannot serve as a yardstick until its own defect is understood.

**Decision**: The broken range is exactly `0x100`–`0x107`.
**Rationale**: 3.0.1 declares `0x107` and 3.3.0 declares `0x108`; the boundary is crisp at the declared
version and coincides with the end of the existing refusal range.
**Alternatives considered**: Treating the range as approximate on the assumption that 3.0.1 and 3.3.5
share a version word. **That assumption was made in an earlier draft and measurement refuted it.**
Recorded here so it is not re-derived.

**Decision**: Layout knowledge for `0x100` is consumed from where it already lives, not rediscovered.
**Rationale**: `M2Era100Constants` already records this era's bone array position and element stride,
with native-client references, and explicitly contrasts them with the `0x108` values. The defect is
that no bone parser consumes it.
**Alternatives considered**: Deriving the layout afresh from the binary. Rejected — the constitution's
format-reader ownership rule forbids rewriting what exists, and the existing record is evidenced.

**Decision**: Spec 154 surfaces avoid `WowViewer.Core.Anim`.
**Rationale**: `PathNormalizer` throws on any path containing the staged client library root, pinned by
tests. Any survey routed through it would refuse the very corpus it must read.
**Alternatives considered**: Fixing `PathNormalizer` as part of this work. Rejected — the constitution
records it as a deliberately unbundled follow-up; bundling it here repeats the mistake it warns about.

## 2. Build enumeration — OPEN

Every staged build at or below 4.0.0 must be listed with its identity, then surveyed. **No build may
be skipped for looking redundant with a neighbour.**

Known present and not yet surveyed:

- Three separate 3.0.1 pre-release builds: **8303, 8334, 8391**. Only 8303 has been read. Under FR-011
  that says nothing about 8334 or 8391.
- A second 2.0.0 pre-release (5665) and a 2.4.3 retail build, neither read.
- A 3.3.5 retail build, not read — the reference measurement is from 3.3.0.
- Additional 1.x and 0.x builds present in the library.

**Why this is not bookkeeping**: structurally significant changes land in `0.0.1` patch releases
without a version-word bump. Two builds one patch apart can differ in ways that break a reader written
against the other. The three 3.0.1 builds are the standing test of whether the survey respects that.

## 3. The `0x100` disambiguation probe — OPEN

`0x100` covers two mutually incompatible layouts and the dispatcher already separates them by probing
the header rather than trusting the version. What the probe inspects, and why it is sufficient, must be
written down before more layouts are routed the same way — it is the existing precedent for
evidence-based layout selection required by FR-002.

Note the probe's real-world consequence: a 2.0.0 pre-release declares `0x100` and is therefore routed
by the probe, never reaching the `0x102`–`0x107` refusal at all. Version-word reasoning would have
mis-predicted that.

## 4. Coverage of `0x102`–`0x106` — OPEN

`0x107` is confirmed present at 3.0.1.8303. Whether any staged build declares `0x102` through `0x106`
is unknown. **This decides how much of US3 is real work** and must be answered before Phase 3 is
scoped. A range with no staged representative cannot be validated and must not be claimed as supported.

## 5. The 4.0.0.11927 contradiction — OPEN

The viewer is reported to render `0x109`+ models; the dispatcher path terminates on one. Two readings,
neither yet eliminated:

- **A**: 11927 is a pre-release outlier and later `0x109`+ builds read cleanly.
- **B**: The render path and the inspection path do not reach the same code, so "the viewer renders it"
  and "the reader parses it" are separate claims.

Reading B would explain the whole contradiction without either observation being wrong, and would mean
the reference route's health has never actually been tested by the tool used to measure it. Resolving
this is SC-002. **Do not adopt either reading without evidence.**

## Open Research Boundaries

- Nothing at or beyond 4.0.1 is read, surveyed, or referenced (SC-008). Later formats are not a
  deferred phase of this work; they are outside it.
- Whether a 3.3.0-era rig is close enough to a 2.x rig to answer the driving question is a separate
  question from whether the readers work. Phase 4 can produce a comparison before Phases 2–3 widen
  which builds may participate; the comparison's own validity across that gap is not assumed.
