# Implementation Plan: WTF Command Inspection

**Branch**: `v0.5.3-dev` (this repo keeps all specs on one branch; no per-feature branch is created)
**Date**: 2026-08-16
**Spec**: [spec.md](./spec.md)

> **This plan is partly retroactive, and says so deliberately.** Phases 1-3 were implemented and
> committed (`f0dffdaa`) before this plan was written — the work ran ahead of the process, which is not
> how this project is supposed to operate. Rather than back-date a plan that pretends to have guided
> work it did not, each phase below records its real status, and what shipped is described as shipped.
> Phase 4 is the only genuinely forward-looking phase.

## Summary

Inspect WTF files across the staged client library and classify every line, to discover real command
syntax rather than assume it. Four capabilities: line classification (US1), archive-packed *and* loose
file discovery (US2), full-library sweep (US3), and candidate-name probing for uncatalogued files (US4).

The load-bearing insight, learned the hard way during this feature's own drafting: **WTF content is
shipped inside the game's data archives, not only loose on disk.** Three separate searches found nothing
until an archive-based one was actually run. That is the difference between this feature working and this
feature reporting a confident, wrong "nothing here."

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.IO.Files` (`ArchiveCatalogSession`, `MpqArchiveCatalog`) for archive access; no new external dependencies

**Storage**: None — this is a read-only inspection tool; output is console/report text, no persisted artifact

**Testing**: xUnit (`WowViewer.Core.Tests`), plus real-data runs against staged clients

**Target Platform**: `tools/inspect` CLI (cross-platform; no viewer/GPU dependency)

**Project Type**: CLI inspection tool over a core library

**Performance Goals**: None stated — the WTF corpus is tiny (single-digit files per build); sweep time is dominated by archive-catalog bootstrap, which is already an existing, shared cost

**Constraints**: Read-only (FR-007); must never modify, move, or delete source files; must distinguish "could not read" from "read and found nothing" (FR-006)

**Scale/Scope**: Small — a handful of files per build across ~10 staged builds

## Constitution Check

| Principle | Check | Status |
|---|---|---|
| I. Repo Independence | All code under `wow-viewer/src/` and `wow-viewer/tools/` | PASS |
| II. Library-First | Classification and sweeping live in `WowViewer.Core.IO/Wtf/`; `tools/inspect` is a thin CLI wrapper (`WtfCommandSupport.cs` parses args and prints, nothing else) | PASS |
| III. Real-Data Validation | Validated against real 0.5.3.3368 and 2.0.0.5610 staged clients, with committed measured output | PASS |
| Format Reader/Writer Ownership | WTF is a new format surface with no prior owner; `WtfLineClassifier` is now its single canonical owner. Archive access reuses `ArchiveCatalogSession` rather than reimplementing MPQ reading | PASS |
| One Phase at a Time | Four phases, each validated against real data before the next | PASS |
| Bite-Sized Plans | Each phase ≤10 steps | PASS |
| No Client Path Assumptions | Roots and listfile paths are CLI arguments, never hardcoded | PASS |
| Data Policy | No client data enters the repo; the tool reads and reports, and its output is not committed | PASS |

## Project Structure

```text
src/core/WowViewer.Core.IO/Wtf/
├── WtfModel.cs             # WtfLine, WtfLineKind, WtfFileSurvey, WtfBuildSurvey, probe result
├── WtfLineClassifier.cs    # SET / bind / PortCommandCandidate / Unrecognized
└── WtfSweeper.cs           # corpus enumeration (archive + loose), sweep, candidate probe

tools/inspect/WowViewer.Tool.Inspect/
├── WtfCommandSupport.cs             # `wtf sweep`, `wtf probe`
└── ArchiveReadTextCommandSupport.cs # `archive read-text` — dump any file's text, loose or packed

tests/WowViewer.Core.Tests/
├── WtfLineClassifierTests.cs
└── WtfBuildSurveyTests.cs
```

## Phases

### Phase 1 (US1) — Line classification — **SHIPPED** (`f0dffdaa`)

1. `WtfLineKind`: `Set`, `Bind`, `PortCommandCandidate`, `Unrecognized`.
2. `WtfLineClassifier`: recognize `SET name "value"`, `bind KEY ACTION`, and a keyword followed by 3-4
   numeric args (port-command-shaped).
3. Retain every line's exact original text — an unrecognized line's real syntax is the entire point.
4. Blank-line filtering; a blank line is not a statement of any kind.
5. Unit tests for each shape, including the real measured lines from `DefaultBindings.wtf`.

**Real-data outcome**: two classifier gaps were found by *running it*, not by reasoning about it —
(a) `realmlist.wtf` uses `set name value` (lowercase, unquoted), which the initial quoted-only pattern
mis-flagged as unrecognized; the pattern was widened to accept an unquoted value. This is exactly the
kind of thing SC-003's "show unrecognized lines verbatim" requirement exists to surface.

**Status**: done, validated, committed.

---

### Phase 2 (US2) — Archive-packed and loose discovery — **SHIPPED** (`f0dffdaa`)

1. Enumerate `.wtf` from each archive's internal listfile (`ExtractInternalListfiles`).
2. Also from the catalogue's broader known-file set (`GetAllKnownFiles`).
3. **Also walk the loose filesystem** — nothing scans for `.wtf` the way `ScanWmoMpqArchives` scans for
   `.wmo.mpq`, so an archive-only corpus misses `Config.wtf` entirely.
4. Record each file's real source (`Loose` vs `Archive`) so attribution is visible in output.
5. Read through `ArchiveCatalogSession`, which already handles both cases.

**Real-data outcome**: step 3 was added *because the first run exposed its absence* — the initial sweep
of 0.5.3.3368 found `DefaultBindings.wtf` (archive) but silently missed `Config.wtf` (loose). Both
sources are required; neither alone is sufficient.

**Status**: done, validated, committed.

---

### Phase 3 (US3) — Full-library sweep — **PARTIALLY SHIPPED**

1. Sweep every `.wtf` file found for a build, classify every line. — done
2. Aggregate per-build counts; deduplicate distinct unrecognized shapes without losing uniques. — done
3. Distinguish unreadable from read-and-clean (FR-006). — done
4. Run against 0.5.3.3368 and 2.0.0.5610. — done; both 100% recognized, zero unrecognized
5. **Run against the remaining ~8 staged builds.** — NOT DONE
6. **Wire `--listfile` so the external community listfile can supplement thin internal listfiles.** —
   done in code, but never validated end-to-end: the last run passed a *relative* path, and
   `MpqArchiveCatalog.LoadListfile` silently returns when the file does not exist, so a mistyped or
   unresolved path produces a normal-looking result with no warning. **Must be re-run with an absolute
   path and the entry count confirmed before any "nothing found" conclusion is trusted.**

**Status**: steps 1-4 done and committed; steps 5-6 outstanding. **The feature is not finished**, and
its current results should be read as "clean for two builds, with a known unvalidated listfile path,"
not as a library-wide negative result.

---

### Phase 4 (US4) — Candidate-name probing — **SHIPPED** (`f0dffdaa`), UNEXERCISED

1. `ProbeCandidate` resolves a guessed name directly through `ArchiveCatalogSession`'s hash-table lookup,
   bypassing every listfile.
2. A resolved candidate is classified identically to a swept file (FR-010).
3. `wtf probe --name <n> [--name <n2> ...]` for batch testing.
4. Unit-testable against a known-present name (`WTF\DefaultBindings.wtf`) and a known-absent one.

**Status**: implemented and committed, but **never run against a real unknown candidate**, because no
candidate name exists yet. This is the live mechanism for finding uncatalogued files — it needs a name
to try, which is a data problem, not a code problem.

## Phase 4 results — 2,217 real candidate names probed across 8 builds (2026-08-16)

The community listfile (`libs/wowdev/wow-listfile/listfile.txt`) turned out to be exactly the candidate
source Phase 4 was waiting for: **2,220 `.wtf` entries**, 2,217 of them under `wtf\`, and they are
precisely the arbitrarily-named, zone-named pattern described — `wtf\1000needles.wtf`,
`wtf\ahnqiraj.wtf`, `wtf\alcazisland.wtf`, `wtf\agmondsend.wtf`, and so on. A `--names-file` option was
added to `wtf probe` to test a list that size (repeated `--name` flags cannot carry 2,217 entries).

Every one of those 2,217 names was probed directly against the archive hash table — bypassing listfiles
entirely — in eight staged builds:

| Build | Resolved | Files found |
|---|---|---|
| 0.12.0.3988 | 1 / 2217 | `defaultbindings.wtf` |
| 0.5.3.3368 | 2 / 2217 | `config.wtf` (loose), `defaultbindings.wtf` |
| 2.0.0.5610 | 1 / 2217 | `defaultbindings.wtf` |
| 2.0.0.5665 | 2 / 2217 | `defaultbindings.wtf`, `runonce.wtf` |
| 2.4.3.8606 | 2 / 2217 | `defaultbindings.wtf`, `runonce.wtf` |
| 3.3.0.10958 | 2 / 2217 | `defaultbindings.wtf`, `runonce.wtf` |
| 4.0.0.12635 | 2 / 2217 | `config.wtf` (loose), `defaultbindings.wtf` |
| Cata beta 11927 | 2 / 2217 | `config.wtf` (loose), `defaultbindings.wtf` |

One genuinely new file was discovered — `wtf\runonce.wtf`, absent from 2.0.0.5610 but present from
2.0.0.5665 onward — read directly, and it is EULA/TOS acknowledgement flags
(`SET readTOS "-1"`, `SET readEULA "-1"`, `SET checkAddonVersion "1"`), not demo content.

**Zero zone-named WTF files resolved in any staged build.** The names are real and catalogued — they came
from somewhere — but that somewhere is not any client currently staged here. Since the community listfile
skews heavily toward modern CASC-era WoW, the most likely explanation is that these zone files live in a
build outside this library, not that they don't exist. This is a *negative result for the staged corpus*,
not a claim the files aren't real.

## Outstanding Work

1. **Obtain a build that actually contains the zone-named WTF files.** Every mechanism to read and
   classify them now exists and is proven; what is missing is a client that has them. This is a data
   acquisition problem, not a code problem.
2. Sweep the remaining staged builds not yet covered by the eight above (the 1.x line, 3.0.1.x, 3.3.5.x)
   — lower priority, since the eight probed span 0.12.0 through 4.0.0 and none carried zone files.
3. If a zone-named file is obtained, its content will immediately exercise the `PortCommandCandidate`
   classifier (keyword + 3-4 numeric args, with coordinate plausibility), which is implemented and
   unit-tested but has never seen a real port command.

## Complexity Tracking

*No Constitution Check violations.*

One process deviation is recorded rather than hidden: **implementation preceded this plan.** The code
shipped first and this plan was written afterward, which inverts this project's "Spec Docs Are Source of
Truth" workflow. The plan is therefore written as an honest record of what exists and what does not,
rather than as a forward-looking document pretending to have directed work already done. The outstanding
items above are genuinely outstanding.
