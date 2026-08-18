# Contract: Sky Source Resolution and Provenance

**Spec**: [../spec.md](../spec.md) | **Satisfies**: FR-001, FR-002, FR-003, FR-005, FR-019, FR-020, FR-021

This is a behavioural contract, not a REST/GraphQL surface — the feature has no network API. It
fixes the rules the resolver must obey and the assertions tests hold it to.

---

## C1: Gradient and model resolve independently

Two separate resolutions run per map load. Neither is a field of the other.

| Build era | Gradient source | Model source |
|---|---|---|
| LIT-era (e.g. 0.5.3 alpha) | Map-scoped `.lit` colour tracks | **No declaration** → discovery fallback |
| DBC-era | `Light*` chain via `LightIntBand` | `LightSkybox.Name` |
| Inside a WMO declaring a skybox | unchanged by the WMO | `MOSB` name overrides |

**Assertion**: a LIT gradient beside a discovered model is a **valid, non-degraded** result. Tests
must not treat a null model declaration on a LIT-era build as failure.

---

## C2: One source per result — never blended

Within a single `SkyGradientSource`, every band comes from one source. Within a single
`SkyModelReference`, the name comes from one declaration.

**Forbidden**: taking Sky Top from LIT and Sky Horizon from the `Light*` chain; taking a band set
from one source and a fog colour from another; filling a missing band from a different source.

**Permitted**: a gradient from LIT and a model from discovery — these are two results, not one
blended result.

**Assertion** (Phase 0, step 6): for every resolver input combination — LIT-only, DBC-only, both,
neither — assert every band in the returned set shares one `SourceKind` **and** one
`SourceIdentity`.

**Rationale**: inherited, not invented. Project research records that LIT tracks and `Light*` records
are separate sources that must not be mixed, and the existing fog code carries a comment that mixing
them "produced a profile that no client file actually authored".

---

## C3: Selection precedence

When more than one source resolves:

1. **Manual override**, if set — recorded with `IsManualOverride = true` (research R7).
2. **Map-scoped** source over **global** source. Map-scoped is the more specific authority.
3. **Authored** source over **fallback**.
4. On a remaining tie, a deterministic, documented order — never insertion or enumeration order.

**Assertion**: resolving the same build and map twice yields the identical selection, including
identical provenance.

---

## C4: Provenance is total

Every value reaching the renderer carries a `SkyProvenance`. There is no path that renders an
unprovenanced value.

**Assertion**: for any resolver result, walking to every leaf reaches a non-null `SkyProvenance`
with a non-empty `SourceKind`, `SourceIdentity`, and `BuildIdentity`.

**Assertion**: `IsAuthored == false` on every value originating from `DiscoveryFallback` or
`HardcodedFallback`, and `true` on every value from `MapScopedLit`, `LightDbcChain`, or
`WmoDeclaration`.

---

## C5: Fallback is reported, never silent

No profile resolving is a **valid** outcome that renders a documented fallback sky (FR-005). It is
never a black frame, an untextured sky, or an exception.

**Assertion**: with all sources unavailable, the resolver returns a usable gradient whose provenance
is `HardcodedFallback` with `IsAuthored = false`.

---

## C6: Classification is declaration-driven with a reported fallback

The declared-skybox name set is the union of `LightSkybox` names and `MOSB` names available for the
loaded build (FR-019).

- A placement matching a declared name **is** a skybox, regardless of its filename.
- A placement not matching any declared name is **not** a skybox, even if its filename contains
  `skybox`, `skybowl`, or `environments/stars/`.
- When no declaration source exists for the build, the filename heuristic applies as an
  **explicitly reported** fallback (FR-020).

**Note**: per C1, LIT-era builds have no outdoor model declaration, so the heuristic fallback is the
expected path there — FR-020 is load-bearing on this branch's target era, not a rare edge.

**Assertion**: a declared skybox whose filename lacks every keyword is classified as a skybox; a
non-sky asset whose filename contains a keyword is not; both report which rule decided (FR-021).

---

## C7: Failure never removes the sky

| Condition | Result |
|---|---|
| No sky profile resolves | Fallback gradient, reported |
| Model name resolves, asset missing | Gradient only; reported **once**, not per frame |
| Model still streaming | Gradient only; no render-thread block |
| WMO declared name unresolvable | Outdoor sky persists; reported |
| Sky rendering disabled | No evaluation, no submission, zero cost |

**Assertion** (SC-007): across this whole matrix, zero black or untextured sky frames.
