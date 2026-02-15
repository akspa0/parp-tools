# Deep Dive — 0.9.0 Transitional Model Routing (MDX ↔ M2)

## Purpose
Establish a proof-first plan for `0.9.0.3807` model loading in the transitional era where extension (`.mdx`) may not uniquely identify parser family.

This document is intentionally focused on **routing and crash triage**, not full parser rewrites.

---

## Baseline facts already proven

1. `0.9.0.3807` has a recovered legacy MDLX dispatcher chain (`FUN_0042a6a0`) with strict `TEXS` divisibility checks and `GEOS` section seeking.
2. `0.9.1.3810` contract requires container-first routing policy: extension is advisory, route by root magic (`MDLX` vs `MD20`).
3. `0.11.3925` and `0.12.0.3988` runtime model loaders are clearly `MD20`-family with strict version gating (not extension-first).

Implication: `0.9.x` is the likely transition window where a single extension may map to multiple container families.

---

## Transitional hypothesis matrix (0.9.0)

| Hypothesis | Description | Current Evidence | Risk | Status |
|---|---|---|---|---|
| H1 | `0.9.0.3807` runtime model loads are MDLX-only | Recovered MDLX dispatcher + section readers in known chain | Medium (may miss alternate loaders) | **Partially supported** |
| H2 | `0.9.0.3807` includes MD20+version path for some assets/loaders | Strongly suggested by `0.9+` architecture rule + later builds | High (extension-routed crash) | **Unproven** |
| H3 | Some 0.9-era files are MD20-like but with missing/ambiguous version field semantics | User crash behavior suggests transition inconsistency | High | **Unproven (priority proof target)** |
| H4 | Crashes are caused by entrypoint divergence (some callsites still extension-first) rather than core parser only | Confirmed historically in viewer before probe-first unification | High | **Supported** |

---

## Required proof targets (binary + runtime)

### P1 — Alternate 0.9.0 model loader discovery
- Identify all runtime model loader entrypoints in 0.9.0 binary, not only `FUN_0042a6a0` chain.
- Specifically search for `MD20` gate patterns and extension normalization logic in adjacent model load code.

### P2 — Version read semantics at MD20 gate
- If `MD20` path exists in 0.9.0, confirm whether it:
  1) reads `version` at `+0x04`,
  2) hard-requires exact value,
  3) or accepts lax/implicit transitional versions.

### P3 — Extension normalization/coercion timing
- Determine where extension rewriting (if any) happens relative to file open, cache lookup, and parser dispatch.
- Confirm whether `.mdx` and `.mdl` are normalized to model2 token prior to parser family selection.

### P4 — Runtime asset corpus fingerprinting
- Sample failing `0.9.0/0.9.1` assets and record:
  - file path
  - extension
  - root magic (first 4 bytes)
  - if `MD20`: header dword at `+0x04`
  - entrypoint (Disk/Catalog/DataSource/World)
  - build/profile context

---

## Minimal diagnostics contract (no parser rewrite)

For each model load attempt, emit one routing diagnostic record:

```text
ModelRouteProbe:
  build
  entrypoint
  file
  extension
  rootMagicHex
  rootMagicAscii
  md20VersionHex (if available)
  selectedFamily (MDX|M2|Unknown)
  mismatch (extension vs container)
```

Counters to increment where relevant:
- `UnsupportedProfileFallbackCount`
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`

---

## Near-term implementation deltas

1. Keep container-first dispatch mandatory for all viewer entrypoints.
2. Add routing probe logs with build + entrypoint context.
3. Keep `0.9.0/0.9.1` MDX profiles provisional until P1–P3 are closed.
4. Do **not** force legacy MDLX parser solely by extension in any `0.9+` path.

---

## Registry guidance pending proof

Until P1/P2 are proven for 0.9.0:
- `ResolveMdxProfile(0.9.0.3807)` may stay provisional (`MdxProfile_090_3807_Provisional`) for MDLX containers.
- `ResolveModelProfile(0.9.x)` should remain explicit/guarded for any MD20 evidence and avoid silent fallthrough.
- Unknown 0.9.x should keep warning-heavy fallback behavior.

---

## Open unknowns (must close before parser rewrite)

1. Exact adoption point in `0.9.x` where runtime accepts/requires MD20 in mainstream model paths.
2. Whether any `0.9.0` MD20 path tolerates absent/implicit version semantics.
3. Whether crash set is dominated by container mismatch vs deeper typed-table decode incompatibility.

---

## Outcome criteria for this deep dive

This deep dive is considered complete when all are true:
1. Every active `0.9.x` model entrypoint is mapped and tagged by parser family gate.
2. At least one representative failing asset from `0.9.0` and `0.9.1` is classified by magic/version/entrypoint.
3. Registry routing decisions are justified by proven binary gates, not extension assumptions.
