# Planning Receipt: v65 Reference and Converter Rebase

**Historical 54-task snapshot**: The later per-layer opacity/writer amendment expands Spec 285
to 62 pending tasks. See [latest receipt](opacity-writer-planning-receipt.md); prior results below
remain evidence only for their original document snapshot.

**2026-10-21 — documentation only.** No implementation task is complete. No build, harvest,
texture extraction, GPU/training or viewer/client run was performed.

## Operator scope represented

- Current data-harvester is the interchange tooling; v65 is the decoded convergence/reference.
- Placement metadata stays in v65; object geometry/masks are excluded (explicit operator answer).
- CASC is a required read-only source backend with product/build/FDID/companion evidence.
- Unknown terrain metadata/chunks remain retained and indexed for later decoding/viewing/models.
- Native cell/layer appearance, full terrain dependencies and all referenced source/RGBA PNG
  textures persist as a common reference; individual target writers report their limitations.
- Old Converter is unverified algorithm reference, not a runtime dependency or parity prerequisite.

## Files authored or amended

Spec 285: `spec.md`, `plan.md`, `tasks.md`, `research.md`, `data-model.md`, `quickstart.md`,
`contracts/terrain-interchange.md`, `contracts/appearance-retention.md`,
`checklists/requirements.md`, and this receipt, all under `specs/285-v65-terrain-interchange/`.

Spec 284: `spec.md`, `plan.md`, `tasks.md`, `research.md`, `data-model.md`, `quickstart.md`,
`contracts/conversion-tools.md`, historical `evidence/planning-receipt.md` supersession note,
and [rebase receipt](../../284-map-object-converter-revitalization/evidence/rebase-v65-receipt.md).

Continuity: `specs/STATUS.md`, `specs/254-epic-datasets-and-terrain-ml/plan.md`,
`memory-bank/activeContext.md`, `memory-bank/progress.md`. Unrelated Spec 283/UI WIP is preserved.
No executable source/project/client files or persisted `.specify/feature.json` were changed.

## Commands and document evidence

From `I:/parp/parp-tools/wow-viewer`:

```powershell
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/285-v65-terrain-interchange'
$env:SPECIFY_FEATURE = '285-v65-terrain-interchange'
& ./.specify/scripts/powershell/setup-plan.ps1 -Json
```

Exit **0**; returned Spec 285 feature/spec/plan paths. Template replaced by authored design.
This is process-local logical feature routing, not a Git branch switch.

Prerequisite validation uses the following command for each of Spec 284 and Spec 285:

```powershell
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/285-v65-terrain-interchange'
$env:SPECIFY_FEATURE = '285-v65-terrain-interchange'
& ./.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks
```

Task-format/count/sequence/phase validation returned exit **0**:

| Package | Sequential tasks | Checked | Malformed | Phase counts |
|---|---|---|---|---|
| 284 | 74 | 0 | 0 | 3,6,7,7,6,8,7,7,8,8,7 |
| 285 | 54 | 0 | 0 | 3,4,7,8,5,6,8,8,5 |

Final prerequisite/link/fence/PowerShell-snippet/containment validation results are recorded in
the linked rebase receipt. Checks concern documentation structure, not implemented features.

## Criterion → planning artifact, with actual proof status

| Criterion | Planning coverage | Actual implementation evidence |
|---|---|---|
| SC-001 native fidelity | US1/US2, native field schema and stream contracts | Pending |
| SC-002 chunk coverage | Per-occurrence/per-stage ledger and missing-handler tasks | Pending |
| SC-003 placements/exclusions | Typed integer metadata and geometry/mask exclusion | Pending |
| SC-004 target output | Reverse-stream readiness, then joint Spec 284 target adapter acceptance | Pending |
| SC-005 migration | Read-only audited copying with missing/re-harvest state | Pending |
| SC-006 CASC source | Actual install/product/build/FDID companion and control pilot | Pending |
| SC-007 retention | Source/native/unknown raw records and query indexes | Pending |
| SC-008 PNG resources | Source bytes, native RGBA/mips, complete texture inventory | Pending |
| SC-009 cell appearance | Cell/layer/native-mask/material and inverse texture query | Pending |
| SC-010 future reuse | Versioned model/view interpretations and immutable committed hashes | Pending |

Read-only review corrected two dependency issues: early terrain proof no longer claims placements
before their implementation, and native/store/transport readiness enables the target adapter before
joint output acceptance. Source inspection also found RGB/opaque-alpha/resized PNG exporters; these
are not treated as preservation exports. Full native RGBA resource support remains a task.
