<!-- SPECKIT START -->
Read current spec pack before non-trivial work. If no pack fits, create one.
<!-- SPECKIT END -->

# wow-viewer AGENTS

Short file. Current truth only. Root `../AGENTS.md` still wins on workspace policy.

## Mission

- `wow-viewer/` is active repo.
- `gillijimproject_refactor/` is read-only reference. DO NOT ALTER THE CODE IN THIS FOLDER! IT IS FOR REFERENCE ONLY.
- Goal: keep `wow-viewer` extractable, spec-driven, and proof-backed.

## Current active lanes

- Spec 102 `102-v25-terrain-convergence` — active model lane. Current path is the
  simple M0 (RGB minimap → object mask) on the existing precise-mask store; the
  strict fragment-trace target is committed but parked. See memory-bank.
- Spec 080 `080-wow-ui-consolidation` — active viewer-shell doc and compatibility lane.

## Background / historical lanes

- Spec 089 `089-dav2-height-predictor` — HISTORICAL/PAUSED. DA-family (Depth
  Anything) models are blacklisted for terrain work (non-deterministic). Do not
  reopen without an explicit instruction.
- Spec 088 `088-v22-enrichment-from-v18` — historical dataset contract; superseded by 102.
- Spec 047 `047-v18-distill-corpus-open-source-loop` — focused V18 operator path.
- Spec 079 `079-runpod-integration-guide` — shared remote bundle/runtime pattern.
- Spec 076 and Spec 077 — paused/background; reuse only when explicitly reopened.

## Hard rules

- The user runs training, GPU jobs, harvests, and any long/heavy/billed run. Prepare the script and hand over the exact `uv run ...` command — do NOT launch it yourself. Communicate directly and respectfully regardless of tone. See root `../AGENTS.md` RULE 0.
- New code stays in `wow-viewer/`.
- Staged clients only: `output/tmp/wowarchive-clients/`.
- No new parser clones when shared `Core` or `Core.IO` surface already exists.
- Implement as much as possible in each session, with memory bank and speckit documentation update at every step. Do not do wasteful partial work that leaves the repo in a state that cannot be built or tested.
- Doc sync same pass: spec, architecture note, memory-bank.
- Consolidate implemented speckit plans when little work or testing remains. Do not leave half-implemented plans in the repo.
- Implement small wins first, then larger wins. Avoid large refactors that break the repo for days.
- C# first, python second for ML tasks. C# is our main tooling for a reason - scalability and maintainability. Python is for ML tasks only. Do not implement new C# features in Python unless the feature is ML-specific and cannot be implemented in C#.

## Spec flow

- Check existing spec first: `wow-viewer/specs/<NNN>-<name>/`.
- If behavior changes, update `spec.md`, `plan.md`, or `tasks.md` in same pass.
- If no spec fits, create spec -> plan -> tasks before broad implementation.

## Canonical docs

- [README.md](/I:/parp/parp-tools/wow-viewer/README.md)
- [docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
- [docs/architecture/wow-engine-modernization-plan-2026-05-14.md](/I:/parp/parp-tools/wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md)
- [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
- [memory-bank/progress.md](/I:/parp/parp-tools/wow-viewer/memory-bank/progress.md)

## Validation

- C#: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- C# tests: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Python: run from `wow-viewer/data-harvester/` with `uv run ...`
- Do not run Python entrypoints from repo root when package imports depend on `data-harvester/src/`.

## Historical docs

- `specs/archived/` = closed or superseded.
- `specs/086-*` and `specs/087-*` stay on disk only as superseded evidence.
- `plans/` = old planning notes unless current spec links them.
- `docs/WoWViewer/` = viewer-facing guide layer; keep current if edited.
- `docs/MdxViewer-legacy-documentation.tar.gz` = archive only.
