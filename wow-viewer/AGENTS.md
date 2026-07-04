<!-- SPECKIT START -->
Read current spec pack before non-trivial work. If no pack fits, create one.
<!-- SPECKIT END -->

# wow-viewer AGENTS

Short file. Current truth only. Root `../AGENTS.md` still wins on workspace policy.

## Mission

- `wow-viewer/` is active repo.
- `gillijimproject_refactor/` is read-only reference.
- Goal: keep `wow-viewer` extractable, spec-driven, and proof-backed.

## Current active lanes

- Spec 089 `089-dav2-height-predictor` — active model lane.
- Spec 088 `088-v22-enrichment-from-v18` — active dataset contract feeding 089.
- Spec 080 `080-wow-ui-consolidation` — active viewer-shell doc and compatibility lane.

## Background lanes still relevant

- Spec 047 `047-v18-distill-corpus-open-source-loop` — focused V18 operator path.
- Spec 079 `079-runpod-integration-guide` — shared remote bundle/runtime pattern.
- Spec 076 and Spec 077 — paused/background; reuse only when explicitly reopened.

## Hard rules

- New code stays in `wow-viewer/`.
- Staged clients only: `output/tmp/wowarchive-clients/`.
- Any `H:\CLIENTS` reference is bug.
- No new parser clones when shared `Core` or `Core.IO` surface already exists.
- One phase at a time. Real-data proof ends phase.
- Doc sync same pass: spec, architecture note, memory-bank.

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
