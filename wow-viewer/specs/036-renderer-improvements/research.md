# Research: Renderer Improvements Convergence

## Decision: Create a new convergence owner instead of mutating specs 030-032 into one of the old feature packs

**Rationale**:
- Specs 030, 031, and 032 each capture valid renderer slices, but they do not share one proof owner.
- Mutating one of them into the new owner would blur what is historical source material versus what is now the active roadmap.
- A new convergence feature keeps provenance intact and makes future routing explicit.

**Alternatives considered**:
- Update spec 032 to absorb 030 and 031: rejected because it would hide the original terrain and WMO slice boundaries.
- Keep all three plans independent: rejected because that preserves the current overlap and drift problem.

## Decision: Keep specs 030-032 as source-slice references and point them at spec 036

**Rationale**:
- Those feature packs contain detailed Ghidra-derived constraints and subsystem notes that are still useful.
- The repo needs a visible handoff note so future work opens the convergence plan first.

**Alternatives considered**:
- Delete or archive 030-032 immediately: rejected because the details are still implementation inputs.

## Decision: Keep M2 recovery work outside this convergence plan

**Rationale**:
- Spec 035 is already the active owner for current M2 route and parity recovery.
- Convergence of 030-032 is about terrain, WMO, lighting, sky/fog, liquid, and viewer host sequencing.
- Pulling M2 parity fully into this feature would break the “one phase at a time” guardrail and blur proof surfaces.

**Alternatives considered**:
- Fold spec 035 into 036 immediately: rejected because it would expand the scope too far and remove a currently useful regression-recovery lane.

## Decision: Sequence lighting and fog foundations before terrain/WMO pipeline convergence

**Rationale**:
- Terrain shading, WMO interior fog, water tint, sky clear color, and shadow behavior all depend on a stable lighting-state source.
- Building geometry or pass orchestration first would force later rewrites when the lighting source changes.

**Alternatives considered**:
- Start with terrain topology first: rejected because several downstream visual behaviors depend on shared lighting/fog contracts.
- Start with WMO pass architecture first: rejected because interior fog and exterior lighting ownership would still be unsettled.
