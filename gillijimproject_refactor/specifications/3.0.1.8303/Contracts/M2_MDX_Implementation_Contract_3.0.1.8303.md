# M2/MDX Implementation Contract — 3.0.1.8303

## Purpose
Translate the reverse-engineered M2/MDX deltas for build `3.0.1.8303` into concrete, minimal repository changes.

This document is implementation-focused (what to change, where, and how to validate), not another RE pass.

---

## Inputs
- [specifications/3.0.1.8303/Contracts/M2_MDX_Contract_3.0.1.8303.md](specifications/3.0.1.8303/Contracts/M2_MDX_Contract_3.0.1.8303.md)
- [specifications/3.0.1.8303/Contracts/baseline-diff-3.0.1.8303.md](specifications/3.0.1.8303/Contracts/baseline-diff-3.0.1.8303.md)
- [src/MdxViewer/Terrain/FormatProfileRegistry.cs](src/MdxViewer/Terrain/FormatProfileRegistry.cs)
- [src/MdxViewer/Terrain/StandardTerrainAdapter.cs](src/MdxViewer/Terrain/StandardTerrainAdapter.cs)

---

## Required architectural decision
Treat build `3.0.1.8303` as **M2-profiled model parsing**, not as legacy `MdxProfile_091_3810` behavior.

### Decision rule
- If `buildVersion == 3.0.1.8303`, model pipeline must resolve to a profile that requires:
  - `MD20` root magic
  - strict typed offset/count span validation
  - version split handling around `0x108`
- Legacy MDX chunk-walk (`GEOS/TEXS`-centric assumptions) must not be auto-applied.

---

## New/updated profile contracts

## A) Keep existing `MdxProfile` for old builds
No behavior change for existing validated old builds (`0.6.x`–`0.9.x`).

## B) Add M2-specific profile contract (new)
Add a new profile type for the modern path (name can vary; example below):

```text
M2Profile
  ProfileId: string
  RequiredRootMagic: enum { MD20 }
  MinSupportedVersion: int   // 0x104
  MaxSupportedVersion: int   // 0x108
  UseTypedOffsetCountTable: bool
  StrictSpanValidation: bool
  VersionSplitThreshold: int // 0x108
  NestedRecordStrides:
    SkinLikeA: 0x70
    SkinLikeB: 0x2C
    EffectLikeA: 0xD4
    EffectLikeB: 0x7C
```

### Registry entries (required)
Add at least:
- `M2Profile_301_8303`
- `M2Profile_30x_Unknown` (strict fallback)

### Dispatch rule (required)
`ResolveModelProfile(buildVersion)`:
- `3.0.1.8303` -> `M2Profile_301_8303`
- `3.0.x.*` unknown -> `M2Profile_30x_Unknown`
- existing old versions -> existing MDX profile resolver path

---

## Code change map

## 1) Profile registry
Target: [src/MdxViewer/Terrain/FormatProfileRegistry.cs](src/MdxViewer/Terrain/FormatProfileRegistry.cs)

Required changes:
- Add new `M2Profile` definition + static profile instances.
- Add `ResolveModelProfile(string? buildVersion)`.
- Keep `ResolveMdxProfile` for old path compatibility.
- Do not alter existing old-build IDs.

## 2) Model parser entry guard
Target: `src/MdxViewer/Formats/Mdx/*` (exact file depends on actual model entrypoint)

Required behavior:
- At parser entry, branch by resolved model profile family:
  - `M2Profile` => enforce `MD20` and typed span table path.
  - legacy `MdxProfile` => keep current MDX parser flow.
- If profile-family mismatch for file/magic, fail fast with diagnostics (no silent coercion).

## 3) Renderer/animator consumption guards
Targets:
- `src/MdxViewer/Rendering/MdxRenderer.cs`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

Required behavior:
- Consume only data model produced by the resolved profile family.
- For unsupported mix (legacy parser output expected, M2 output provided, etc.), issue diagnostic and skip unsafe processing.

---

## Field-level mapping (RE -> profile)

| Ghidra evidence | Profile field | Required parser rule |
|---|---|---|
| `0x0079A8C0` checks `0x3032444D` | `RequiredRootMagic=MD20` | Reject non-MD20 in `M2Profile_301_8303` |
| `0x0079A8C0` checks `0x103 < ver < 0x109` | `MinSupportedVersion=0x104`, `MaxSupportedVersion=0x108` | Reject out-of-range versions |
| `0x00797540/50/710/5D0` | `UseTypedOffsetCountTable=true` | Validate `offset <= size` and `offset + count*stride <= size` |
| `0x00798DA0` | `NestedRecordStrides.SkinLikeA=0x70` | Enforce exact stride on table walk |
| `0x007985F0` | `NestedRecordStrides.SkinLikeB=0x2C` | Enforce exact stride on table walk |
| `0x00799340` | `NestedRecordStrides.EffectLikeA=0xD4` | Enforce exact stride on table walk |
| `0x0079A720` | `NestedRecordStrides.EffectLikeB=0x7C` | Enforce exact stride on table walk |
| `0x0079A8C0` split `<0x108` vs `>=0x108` | `VersionSplitThreshold=0x108` | Branch to version-specific decode path |

---

## Required diagnostics
For all model parsing under this contract, emit:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

And always include context:
- `buildId`
- `profileId`
- `filePath`
- `chunkFamily` (`M2|MDX`)

Additional strongly recommended model counters:
- `ModelHeaderValidationFailCount`
- `ModelSpanBoundsFailCount`
- `ModelVersionUnsupportedCount`
- `ModelProfileFamilyMismatchCount`

---

## Migration sequence (minimal-risk)
1. Add `M2Profile` type + registry entries (no runtime call-site changes yet).
2. Add a model-profile resolver call at model parse entry.
3. Guard current MDX path behind legacy profile family only.
4. Add M2 path guardrails (`MD20`, span checks, version gate).
5. Wire diagnostics counters and structured context fields.
6. Turn on `3.0.1.8303` mapping in registry.

---

## Acceptance criteria
Implementation is complete when all are true:
1. Build `3.0.1.8303` resolves to `M2Profile_301_8303` deterministically.
2. Non-`MD20` files on this profile fail fast with diagnostics.
3. Offset/count overrun cases increment bounds counters and abort section safely.
4. Legacy validated builds (`0.6`–`0.9`) still resolve and parse using existing profile behavior.
5. No silent fallback from M2 profile to legacy MDX parser.

---

## Out-of-scope for this patch set
- Full semantic renaming of every `0x0079A8C0` header slot.
- Rebuilding renderer material semantics beyond safety guards.
- Any ADT/WMO behavior changes not already listed in baseline diff.

---

## Implementation note
Keep the first code change set small:
- profile definitions + resolver + parse entry guards + diagnostics only.

Do not refactor rendering or animation internals until profile dispatch and validation behavior is stable in logs for `3.0.1.8303`.
