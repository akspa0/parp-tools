# Quickstart: MCAL Alpha Map Decode Correctness

**Date**: 2026-09-01

> PowerShell 7. The corpus sweep and the captures are operator-run; the agent prepares
> commands and stops.

## Build and test

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj `
    -c Debug --filter "FullyQualifiedName~AlphaDecode"
```

Baseline: `WowViewer.Core.Tests` has **9 pre-existing failures** unrelated to this feature.
Compare against 9.

## The sweep — the measurement this spec is built on

```powershell
dotnet I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll `
    alpha-decode-sweep `
    --clients-root H:/CLIENTS `
    --out I:/parp/parp-tools/wow-viewer/output/alpha-decode-report
```

Run it **before** any decode rule changes (T005) and again after consolidation (T206). The
comparison is the gate — Phase 3 may not make the fully-accounted rate worse in any era.

## Reading the report

Three numbers matter, per era:

- **Fully-accounted chunks** — decoded layers consume the MCAL payload exactly. This is the
  real measure of "do we know how to read this". Rising is the goal.
- **Failure counts by reason** — `OffsetOutOfRange` and `TruncatedPayload` point at the rule;
  `CompressedStreamOverrun` points at the RLE decoder. `NoAlphaFlag` is **not** a failure and
  is expected to dominate on 0.5.3.
- **`UnexplainedFiles`** — the paths the rules cannot account for. This list *is* the
  remaining unknown (SC-006). An empty list means the era is solved; a short list is a work
  queue; a long list means the rule is wrong, not that the files are.

A rate that looks good with a large `UnexplainedFiles` list is the failure mode to watch for:
it means something is absorbing the remainder instead of reporting it, which contract C5
forbids.

## Expect terrain to look worse before better

After Phase 4 removes the fabrication, layers that fail to decode will be **absent** rather
than rendered as opaque blocks. That is the intended state: the blocks were invented data,
and the sweep report — not the viewport — is what says how much is genuinely missing.

## Operator-owned steps

- **T005 / T206** — the sweep across the client library.
- **T304** — Mogu'shan Palace capture (SC-004).
- **T305** — 0.5.3 reference capture, must be pixel-identical (SC-007).
- **T404** — harvest-vs-render byte comparison (SC-005).
