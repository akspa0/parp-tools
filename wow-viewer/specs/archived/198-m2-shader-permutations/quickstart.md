# Quickstart: M2 and WMO Shader Permutation System

**Date**: 2026-09-01

> Commands are PowerShell 7. Nothing here launches a heavy or billed run; the capture and
> real-client comparison steps are operator-owned.

## Build and test

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj `
    -c Debug --filter "FullyQualifiedName~Permutation"
```

Known baseline: `WowViewer.Core.Tests` carries **9 pre-existing failures** unrelated to this
feature (`WorldFramePassCoordinatorTests` x3, `WtfLineClassifierTests` x2,
`EnrichmentStreamFormatTests`, `ModelFootprintReaderTests`, `V18StorePlacementsReaderTests`,
`AdtV23SummaryReaderTests`). Compare against that number, not against zero.

## Reading the selection report

The report is per scene load and is surfaced in the viewer's diagnostics panel. It answers
the three questions this feature exists to make answerable:

- **What did the corpus ask for?** `Requested` — permutation to batch count.
- **What is still missing?** `NotImplementedPermutations` — this is SC-006, and it is the
  work queue for Phase 4, ordered by request count.
- **What could not be resolved at all?** `Unresolved` plus `UnresolvedModels`. A non-zero
  count here is a decode gap, not a rendering gap — it means a batch's selector was not
  understood, and per FR-002 it must never be silently assigned a plausible permutation.

## Verifying the off-switch (SC-005)

The permutation path must be reducible to exactly today's rendering:

1. Load the reference scene, capture with the permutation path **enabled**.
2. Toggle it off in the same session, capture again.
3. With no permutation implemented yet (end of Phase 1), the two captures must be
   **pixel-identical**. After Phase 4 they must differ only where a difference has been
   verified against the real client.

## Verifying era isolation (FR-006)

0.5.3 is this project's primary lane and must not move:

1. Load a 0.5.3 scene with the permutation path enabled.
2. The report must show every batch as non-applicable, not as fallback — those are different
   states and the distinction is the whole point of the gate.
3. The capture must be pixel-identical to the same scene with the path disabled.

## Operator-owned steps

Per the execution boundary in `AGENTS.md`, these are yours, not the agent's:

- Real-client side-by-side comparison for SC-002, one scene per implemented permutation.
- The before/after CPU stage capture for SC-004, using spec 136's measurement path.

The agent prepares the exact commands and stops.
