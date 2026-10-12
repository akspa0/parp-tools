# Contract: CLI Commands

**Phases**: 2–9 | **Satisfies**: FR-002, FR-011, and the operator surface for every gate

All commands are **thin wrappers** (Constitution II) over `WowViewer.Core.PM4` analyzers, added to
the flat `switch` in `RunPm4` in `tools/inspect/WowViewer.Tool.Inspect/Program.cs`. Every one follows
the shape of the existing `RunPm4Unknowns`:

```csharp
string? input  = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
string? output = GetOption(args, "--output", "-o");
// ... analyzer call ...
// --output present  -> write indented JSON, print "Wrote <path>"
// --output absent   -> print a human-readable report to stdout
// missing input     -> stderr message, Environment.ExitCode = 1
```

**Note on `pm4` subcommand nesting**: the dispatch in `RunPm4` is a **flat** switch despite its
inconsistent indentation — `unknowns`, `cross-tile`, `linkage` and the rest are all direct
subcommands. New commands are added flat, at the same level.

## New commands

### `pm4 evidence` — Phase 2

Read, seed, or merge the decode evidence register.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory (for `corpusSignature`) |
| `--register`, `-r` | no | register path; default `output/pm4-decode/evidence-register.json` |
| `--seed` | no | seed from the nine `Pm4ResearchUnknownsAnalyzer` findings; refuses to overwrite an existing register without `--force` |
| `--output`, `-o` | no | write JSON; otherwise print to stdout |

### `pm4 grouping-rules` — Phases 3–4

Evaluate every candidate grouping rule corpus-wide in one corpus read.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--rules` | no | comma-separated rule ids; default all (`G0,G1,G2,G3,G4,G5`) |
| `--output`, `-o` | no | write `grouping-comparison.json` |
| `--register`, `-r` | no | also record each rule's outcome, including eliminations, in the register |

### `pm4 object-identity` — Phase 5

Apply one rule and emit the per-surface assignment table.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--rule` | no | rule id; default is the Phase 4 winner recorded in the register |
| `--output`, `-o` | **yes in practice** | `object-identity.json` — this is the artifact 129 and the viewer read |

### `pm4 connective-geometry` — Phase 7

Discriminate what the connective geometry is — MSPV/MSPI windows **and** MSCN, as co-equal candidates.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--source` | no | `mspi`, `mscn`, or `both` (default) |
| `--verify-detector` | no | run the constructed-case detector-power check and report only that |
| `--output`, `-o` | no | write `geometry-stream.json` |
| `--register`, `-r` | no | record the outcome, including elimination, in the register |

`--verify-detector` exists so the Phase 7 detector-power gate is runnable on its own, before any
corpus claim.

### `pm4 reconstruct-object` — Phase 8

Rebuild one object with and without the connective stream and measure it against a real asset.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--identity` | yes | `object-identity.json` from Phase 5 |
| `--object-id` | yes | which object to reconstruct |
| `--asset-root` | yes | **configured** client/asset root — Constitution VI, never hardcoded |
| `--output`, `-o` | no | write the comparison JSON |

If the corresponding asset cannot be located, the command reports which objects were attempted and
why each was rejected, and exits non-zero. It does not substitute a weaker comparison.

### `pm4 mprr` — Phase 9

Sweep MPRR against every domain and test the sentinel-delimited run hypothesis.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--runs` | no | include per-run structure analysis (slower; tile 0_0 alone holds 81,936 entries) |
| `--output`, `-o` | no | write JSON |
| `--register`, `-r` | no | record eliminated domains |

## Extended commands

### `pm4 bounds-audit --by-region` — Phase 0.5

Groups MSVT by `MSHD.Field04` and reports, per region, the frame the placement pipeline actually
resolves and how far that moves geometry off the ADT-verified canonical transform.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--by-region` | — | selects this mode; without it the command behaves exactly as before |
| `--placements` | no | directory holding the companion `_obj0.adt` files; defaults to `--input` |
| `--output`, `-o` | no | write JSON; otherwise print to stdout |

**Why it does not simply group raw bounds.** The obvious reading of the plan — group MSVT bounds by
region, check each against the tile band its filename implies — is a test every input passes
(309/309), so it reports "uniform" regardless of the truth. It is still emitted as the continuity
baseline, but the discriminating measurements are the **resolved frame per CK24 object** (which
varies) and **agreement with real ADT placements** (which is external ground truth).

`--placements` is read in the tool rather than the analyzer because `WowViewer.Core.IO` already
references `WowViewer.Core.PM4`; reading ADTs from inside the analyzer would be a reference cycle.
The analyzer instead accepts reference points as an argument, which also makes it unit-testable.

### `pm4 yaw-evidence` — Phase 0.5

Decides whether the placement fitter's per-object yaw correction helps or hurts, by scoring each
object's footprint against the world bounding box of the WMO placement it stands in.

| flag | required | meaning |
|---|---|---|
| `--input`, `-i` | yes | PM4 corpus directory |
| `--placements` | no | directory holding the companion `_obj0.adt` files; defaults to `--input` |
| `--output`, `-o` | no | write JSON; otherwise print to stdout |

Exits non-zero when no MODF world-model placements can be found, rather than reporting a verdict
from nothing. MDDF doodad placements are points and cannot score a rotation, so only MODF is usable.

**Detector power is built in.** Every object is also scored under a deliberate 45° control rotation.
An object whose containment the control does not reduce is one whose box cannot see a rotation at
all; it is excluded from the headline and counted separately, so "the yaw makes no difference" can
never be confused with "this test cannot tell". Matching is by centroid containment, which a
rotation about the centroid cannot change — matching on best fit would have selected for the
unrotated reading and then concluded in its favour.

### `pm4 unknowns`

**Extended, not changed.** Phase 9 adds the full-domain MPRR sweep to its findings. Every existing
number it emits — the six zero-miss relationships, the six partial edges, all field distributions,
`Pm4MspiInterpretationSummary`, `Pm4LinkIdPatternSummary`, the nine `Pm4UnknownFinding` records —
must come back **bit-identical** on the same corpus.

Those numbers are the baseline the spec, the epic, and the memory bank all cite. A regression test
pins them.

## Conventions inherited from the existing surface

- `--input` accepts a workspace-relative path; `Pm4CoordinateService.ResolveMapDirectory` walks
  parent directories to resolve it, so commands work from anywhere in the tree.
- `--output` creates its parent directory.
- JSON is `System.Text.Json` with `WriteIndented = true`.
- Missing required input → message on stderr, `Environment.ExitCode = 1`.
- No `--output` → human-readable report on stdout.

## Documentation accuracy

`quickstart.md` must be re-verified against the real argument parsing **after** implementation, not
written once from this contract and trusted. A documented command that does not run is a defect —
this repo has been bitten by exactly that before. The Phase 2 task list includes the verification
step, and every phase that adds a command re-runs it.

Two confirmed defects in the existing surface, found while measuring for research.md R7/R8:
**`pm4 inspect` and `pm4 audit` accept `--output` and silently ignore it** — `RunPm4Inspect` and
`RunPm4Audit` only call their `Print…` method. `pm4 unknowns`, `pm4 mshd`, `pm4 cross-tile` and
`pm4 export-json` do honour it. Fix opportunistically when those functions are touched.
