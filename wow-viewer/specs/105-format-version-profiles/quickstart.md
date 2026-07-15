# Quickstart: Validating Spec 105

**Feature**: 105-format-version-profiles | **Date**: 2026-07-15

Commands per phase gate. Per **AGENTS Rule 0**, the agent prepares these; the **user runs** anything
touching a real client or the viewer. §3 is the only gate that proves the feature works, and it
cannot be self-certified.

All paths are relative to `i:\parp\parp-tools`. Validation uses staged clients only (Constitution III).

---

## §0. Pin the 3.x/4.x baseline — RUN THIS FIRST

**Ordering is load-bearing.** Once a shared type changes, a baseline measures the new code against
itself and proves nothing (research R7). Do not skip because "the tests pass" — the memory bank
records that `M2Era1121ModelReaderTests` passed 9/9 while never touching the era-100 path.

```powershell
dotnet build wow-viewer\WowViewer.slnx -c Debug
dotnet test wow-viewer\tests\WowViewer.Core.Tests --filter "FullyQualifiedName~M2ThreeXBaselineRegression"
```

**Expected**: green, and `M2ThreeXBaseline.json` written + committed.
**Gate**: baseline committed **before** any shared type is touched. Any later diff is a real regression.

---

## §1. Profile contract lands, inert M2 half deleted

```powershell
dotnet build wow-viewer\WowViewer.slnx -c Debug
dotnet test wow-viewer\tests\WowViewer.Core.Tests --filter "FullyQualifiedName~M2"
```

**Expected**: 0 errors; all M2 tests green; §0 baseline unchanged (this phase is behaviour-neutral).

**SC-004 shrink check** — the bar the feature is measured against:

```powershell
git diff --stat HEAD -- wow-viewer/src/core/WowViewer.Core/M2 `
                        wow-viewer/src/core/WowViewer.Core.IO/M2Chunked `
                        wow-viewer/src/viewer/WoWViewer/Terrain/FormatProfileRegistry.cs
```

**Expected**: net **negative**. Research R6 estimates ≈ −50. **A positive net means a third scheme was
built instead of two being reconciled — stop and reconsider, do not proceed to §2.**

---

## §2. Era-aware track addressing + time base

```powershell
dotnet test wow-viewer\tests\WowViewer.Core.Tests --filter "FullyQualifiedName~M2TrackSamplerEra"
dotnet test wow-viewer\tests\WowViewer.Core.Tests --filter "FullyQualifiedName~M2ThreeXBaselineRegression"
```

**Expected**:
- New era tests green: inclusive bounds, empty ranges, global-sequence override, degenerate range (`last ≤ first`), cross-range clamp (FR-004).
- **Baseline bit-identical** (FR-015). Should hold *structurally*: Wrath sets `Start = 0`, so `Start + (elapsed mod Duration)` reduces to today's `elapsed mod Duration`. **A diff here means the normalization is wrong, not that the baseline needs updating. Do not re-baseline to make it pass.**

**Era-100 sanity — the "3333" pathology**: inspect a 1.0.0 model's sequence durations.

```powershell
dotnet run --project wow-viewer\src\tools\... -- m2 inspect --path "<staged 1.0.0 client>\Creature\...\TrollMale.m2"
```

**Expected**: durations are plausible animation lengths. Before the fix, `Duration` was populated from
`start` (research R1) — the user's reported "frame 2030/3333" was a global-timeline offset displayed
as a duration. **If durations still look like timeline offsets, the offset fix did not land.**

---

## §3. 1.0.0 models animate — SC-001 — **USER RUNS THIS**

The only gate that proves the feature. **Cannot be self-certified** (AGENTS Rule 0). A green build is
not signoff.

```powershell
dotnet run --project wow-viewer\src\viewer\WoWViewer\WoWViewer.csproj
```

Then in the viewer:
1. Open a staged **1.0.0** client.
2. Load **TrollMale** and **DoomGuard** (the models from the original report).
3. Play an idle/walk sequence.

**Expected**:
- Visible **skeletal motion**. The "frame 2030/3333 but static" symptom is gone.
- Status bar reports `Renderer: M2Renderer` (the native path, not the MDX fallback).
- Sequence frame counts look like animation lengths.

**Watch for** (each has a specific cause):
- *Bones move but poses are wrong* → interpolation ranges misindexed, or the era-100 sequence offset fix (§2) did not land.
- *Model animates but jitters at loop boundaries* → the total-count clamp (FR-004) or the `Start` offset.
- *Still static* → bones not populated (Phase 3 step 3), or the model resolved to the wrong era.

**Then check no-regression by eye**: load a **3.3.5** and a **4.0.0** model and confirm they animate as before.

---

## §4. Deterministic era resolution

```powershell
dotnet test wow-viewer\tests\WowViewer.Core.Tests --filter "FullyQualifiedName~M2EraResolution"
```

**Expected**: 1.0.0 and 1.12.1 models each route correctly **with the trial-parse removed**; an
unrecognized model raises an error **naming the version and the ambiguity** (FR-013) rather than
falling through.

**Also confirm** (SC-005):

```powershell
git grep -n "ValidateLayout" -- wow-viewer/src/core/WowViewer.Core.IO/
```

**Expected**: no remaining trial-parse call in `DetectEra`.

---

## Open items needing a user decision

1. **FR-017 reading** (research R6): deleting the inert `M2Profile` records leaves a dangling
   validation call in `WarcraftNetM2Adapter`. Removing it technically touches a file FR-017 says is
   untouched — but it validates against records proven inert (identical strides everywhere; it can
   only ever pass), so removal cannot change behaviour. **Confirm**: remove it, or keep a local copy
   of the records in the adapter? Both satisfy SC-004.
2. **1.12.1 may be broken today** (research R1): era-1121 uses the same Wrath sequence offsets under a
   non-Wrath stride. If 1.12.1 models animate wrongly and it has gone unreported, that is evidence the
   bug class is broader. Tracing a 1.12.1 client in Ghidra would settle it and de-provisionalize the
   `Era1121` row.
