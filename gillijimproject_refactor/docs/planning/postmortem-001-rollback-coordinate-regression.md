# Postmortem 001 — WoWRollback Coordinate Regression (2025-10-01)

## 1. Summary
- **Symptom**: 0.5.x comparison CSVs regressed to `world_x/world_y/world_z = 0` and UniqueID coverage collapsed. Viewer overlays and analytics became unusable.
- **Trigger**: Attempt to split `MDDF` (doodads) and `MODF` (WMOs) into distinct plotted layers led to ad-hoc workflow changes, temporary directory usage, and partial automation rewrites.

## 2. Timeline
- **Earlier 2025-10-01** — Commit `bd759de0` verified working pipeline: assumed pre-converted LK ADTs at a stable path, produced valid coordinates.
- **Afternoon** — Began refactor to auto-convert Alpha ADTs inside `regenerate-with-coordinates.ps1`. Split MODF/MODF plotting was *scheduled* but never delivered.
- **Evening** — Temporary stash (`output_dirfart2/`) supplied converted ADTs. Scripts started assuming this directory existed, without regenerating it.
- **Later** — Temporary folder cleaned. Regenerate script ran, could not find converted ADTs, silently emitted zero coordinates. Additional edits tried to bolt on automation but relied on the same temp root, compounding failures.

## 3. Root Causes
- **[RC1] Missing source-to-build contract**: Scripts relied on manually pre-converted LK ADTs rather than owning the conversion step. No reproducible command sequence existed.
- **[RC2] Temporary assets baked into configuration**: `ConvertedAdtDir` default pointed at an ephemeral developer folder. Once removed, coordinates vanished.
- **[RC3] Feature creep without validation**: Work pivoted from "split plots" to "rewrite automation" without securing regression tests or verifying MODF/MDDM separation functionality.
- **[RC4] Lack of guardrails**: No smoke tests to fail fast when `MDDF`/`MODF` chunk counts read zero. The pipeline emitted success even when data was missing.

## 4. Impact
- Lost an afternoon to firefighting instead of shipping the desired MODF/MDDM split.
- Export pipeline now considered unstable; user confidence lost.
- Additional clean data sets (0.6.0) overwritten or invalidated while experimenting.

## 5. Immediate Fix
- Revert `WoWRollback` scripts and related changes back to commit `bd759de0`. Restore working coordinate outputs while we regroup.

## 6. Preventive Actions (Required)
1. **Document canonical workflow** (`0.5.x`, `0.6.0`): Write a runbook that lists exact commands to (a) convert Alpha ADTs, (b) regenerate outputs, (c) verify coordinate coverage.
2. **Stage converted assets under versioned cache**: Store regenerated LK ADTs inside `rollback_outputs/converted_adts/<version>/<map>/`. Treat external directories as read-only fallbacks.
3. **Smoke tests**: Add a CLI switch or script step that asserts `MDDF` + `MODF` counts > 0 for every `{version,map}`. Fail fast if not.
4. **Feature gates**: Keep MODF/MDDM plot splitting behind a feature flag until validated with fixtures.
5. **Docs first**: Before touching pipelines, update planning doc with scope, exit criteria, rollback plan.

## 7. Lessons Learned
- Automation that depends on temp assets is a time bomb. Own the build inputs or fail loudly.
- Each regression cost >1 hr to diagnose. A 5-minute smoke check would have caught missing ADTs immediately.
- Scope creep during crisis creates more risk than value. Finish the original deliverable or formally pause it.

## 8. Next Steps
- Freeze new changes until the revert lands and baseline outputs are verified.
- Schedule a dedicated task to rebuild conversion automation from scratch with tests.
- Resume MODF/MDDM plotting only after coordinates are trustworthy again.
