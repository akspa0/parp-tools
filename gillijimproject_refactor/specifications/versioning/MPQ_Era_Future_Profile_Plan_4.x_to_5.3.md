# Future Profile Plan — 4.x To 5.3 MPQ Era

## Goal

Define the future profile work needed for `4.x` and `5.x` client support without destabilizing the active `3.x` recovery line.

This is a planning document only. It does **not** change the current priority: smooth out `3.x` terrain behavior first, with real-data validation, before taking on broader format intake.

---

## Current Decision Boundary

- Immediate priority remains `3.0.1` / `3.3.5` terrain stability.
- The current `4.x` and `5.x` entries in `FormatProfileRegistry` are scaffolding only.
- Do not treat `AdtProfile_40x_Unknown` or `AdtProfile_50x_Unknown` as proof of real support.
- Do not reintroduce `*_tex0.adt` or `*_obj0.adt` into the active `3.x` smoothing path.
- Do not start CASC work from this plan. CASC is a separate later track.

---

## Scope

In scope for this plan:
- Build out future parser/profile families for MPQ-era `4.x` and `5.x` clients.
- Separate ADT, WMO, MDX, and M2 profile work by build family.
- Define evidence gates before any future profile is considered usable.
- Keep split terrain and placement sourcing (`*_tex0.adt`, `*_obj0.adt`) isolated to future profiles that explicitly opt in.

Out of scope for this plan:
- Additional `3.x` terrain experiments.
- Heuristic fixes that mix `3.x` and `4.x+` behavior.
- CASC client support.
- Blanket expansion-wide claims such as “Cataclysm supported” or “MoP supported”.

---

## Planning Principles

1. Build-level dispatch beats expansion-level assumptions.
2. `3.x`, `4.x`, and `5.x` must remain isolated unless binary evidence proves schema compatibility.
3. Split ADT routing is profile-controlled, never auto-enabled by filename presence alone.
4. Placeholder profiles are acceptable for scaffolding, but support claims require real-data validation.
5. Each format family needs its own contract: ADT, WMO, MDX, and M2 do not move in lockstep.
6. Validation must use real client data, not synthetic fixtures.

---

## Why Future Profiles Are Needed

The active registry already shows the right architectural direction:
- `3.x` terrain is now isolated from split ADT sourcing.
- `4.x` and `5.x` have provisional ADT placeholders that opt into `*_tex0.adt` and `*_obj0.adt`.

That is useful as a guardrail, but it is not enough for actual support. `4.x` and `5.x` need their own evidence-backed profile families because:
- split terrain and placement sourcing changes the file loading contract,
- parser assumptions that are safe for `3.3.5` may be wrong for later MPQ-era clients,
- `5.x` is not just “more `4.x`”; it needs its own research and promotion path.

---

## Planned Version Families

| Family | Role | Status | Notes |
|---|---|---|---|
| `3.0.1` / `3.3.5` | current stabilization line | active | finish smoothing here before future intake |
| `4.0.0` anchor | first `4.x` research/provisional family | planned | use as the first build-specific Cataclysm MPQ anchor |
| later `4.x` MPQ anchor | second `4.x` family if drift is proven | planned | do not assume `4.0.0` covers all `4.x` |
| early `5.x` MPQ anchor | first `5.x` research/provisional family | planned | must be isolated from `4.x` until proven compatible |
| `5.3.x` anchor | final MPQ-era target family | planned | end-state goal for MPQ-era client coverage |

Recommended interpretation:
- Start `4.x` with a build-specific anchor, not a generic `40x` family claim.
- Start `5.x` with a build-specific anchor, not a generic `50x` family claim.
- Keep `40x` and `50x` placeholders only as temporary holding profiles until exact-build families exist.

---

## Target Profile Set

The future registry should grow in matched sets instead of adding ADT-only support and calling the version “done”.

For each future anchor build, plan to add:
- `AdtProfile_<build>_Provisional`
- `WmoProfile_<build>_Provisional`
- `MdxProfile_<build>_Provisional`
- `M2Profile_<build>_Provisional`

Initial future targets:
- `4.0.0.11927` provisional set
- one later `4.x` provisional set after drift is proven
- one early `5.x` provisional set
- one `5.3.x` provisional set

Promotion rule:
- a provisional profile is promoted only after binary evidence is documented and real-data rendering/parsing is validated for the targeted format families.

---

## Format-Specific Work

## ADT Work

Future `4.x` / `5.x` ADT profiles must prove:
- root ADT vs `*_tex0.adt` responsibilities,
- root ADT vs `*_obj0.adt` placement responsibilities,
- MCNK subchunk ordering and size semantics,
- MCAL/alpha behavior and any post-`3.x` changes,
- liquid path expectations (`MCLQ`, `MH2O`, or both),
- placement chain correctness for `MMDX/MMID`, `MWMO/MWID`, `MDDF`, and `MODF`.

Implementation consequence:
- `StandardTerrainAdapter` should only route split terrain or split placement through profile-owned policy flags backed by evidence.

## WMO Work

Future `4.x` / `5.x` WMO profiles must prove:
- root required chunk order,
- group required chunk order,
- optional block gates,
- liquid behavior,
- placement interpretation if record or flag semantics drift.

Implementation targets:
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`

## MDX / M2 Work

Future `4.x` / `5.x` model profile work must prove:
- container identity and version gating,
- whether `.mdx` remains a reliable label or only a file extension alias,
- skin/effect record stride changes,
- any layout changes across `MD20`-family headers and typed offset/count tables,
- animation/material assumptions that differ from the current `3.x` M2 path.

Implementation targets:
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Formats/M2/*`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

---

## Required Evidence Before Coding A Future Family

Minimum proof pack per future anchor build:

1. ADT root parser anchor with chunk discovery rules.
2. `*_tex0.adt` and `*_obj0.adt` responsibility proof, if present.
3. MCNK field map deltas relative to the nearest known family.
4. Placement chain proof for doodads and WMOs.
5. Liquid parse/render expectations.
6. WMO root/group required chunk contracts.
7. Model container discriminator proof (`MDLX`, `MD20`, other version gates if any).
8. Real-data open/load/render validation on at least one representative map or asset set for that family.

Documentation requirement:
- each future family should get a build-specific contract or baseline-diff document under `specifications/<version>/Contracts/` or an equivalent versioned spec path before the implementation is promoted.

---

## Proposed Delivery Sequence

## Phase 0 — Hold The Line On 3.x

Exit criteria before future intake:
- `3.x` terrain alpha/chunk behavior is stable on the user’s real data,
- no active `3.x` work depends on `*_tex0.adt` or `*_obj0.adt`,
- the current `3.x` profile split is not regressing Alpha-era terrain.

## Phase 1 — Replace Major-Version Placeholders With Exact Anchors

Work items:
- keep `AdtProfile_40x_Unknown` and `AdtProfile_50x_Unknown` as temporary placeholders only,
- add one exact-build `4.x` profile set,
- add one exact-build `5.x` profile set,
- update resolver dispatch so known builds stop falling into generic major-version buckets.

## Phase 2 — Add Diagnostics Before Broad Intake

Add structured counters and log context for:
- split-source path selection,
- unsupported future-profile fallbacks,
- missing required split files,
- chunk contract violations.

This is required so later failures are visible instead of being mistaken for rendering bugs.

## Phase 3 — Promote ADT Support First

Rationale:
- terrain format drift is already known to be material,
- `4.x` / `5.x` split ADT support is the most explicit future requirement already identified.

Work items:
- make future ADT profiles exact-build or narrow-family,
- prove split file responsibilities per anchor build,
- validate terrain rendering on real data before expanding claims.

## Phase 4 — Promote WMO And Model Profiles Separately

Do not bundle these under a vague “version support” milestone.

Track separately:
- WMO root/group contracts,
- MDX container behavior,
- M2 version/layout behavior.

## Phase 5 — Reach Final MPQ-Era 5.3 Coverage

Definition:
- exact or explicitly bounded profiles exist for the targeted `5.3.x` family,
- known MPQ-era profile families are documented,
- unresolved cases remain visibly provisional instead of falling through silently.

---

## Registry And Code Changes To Expect Later

When this work starts, expect changes in:
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Formats/M2/*`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/*`

Preferred direction:
- keep resolver logic explicit,
- favor exact-build mappings over broad major-version fallbacks,
- keep split-source decisions on the profile contract,
- add diagnostics before widening compatibility claims.

---

## Non-Goals And Guardrails

- Do not let this roadmap distract from finishing `3.x` terrain smoothing.
- Do not use the existence of provisional `40x` / `50x` profiles to claim runtime support.
- Do not move `3.x` back onto split ADT sourcing to “reuse” future work.
- Do not collapse `4.x` and `5.x` into one shared family without evidence.
- Do not roll CASC support into this plan.

---

## Definition Of Done For This Plan

This plan is fulfilled only when:
- `3.x` stabilization is no longer being actively protected from future-profile churn,
- `4.x` and `5.x` both have exact or tightly bounded profile families,
- those families have documented evidence and real-data validation,
- unresolved future builds remain provisional and visible in diagnostics,
- the repo can describe MPQ-era coverage through `5.3.x` without hand-waving.