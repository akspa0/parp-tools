# GV-26 Plan Pack Audit Matrix

## Intent

Track whether each micro-plan is truly implementation-ready or still too vague.

## Scope

- plan id
- size rating
- dependency clarity
- touched-surface clarity
- proof clarity
- split-needed flag

## Outputs

- audit matrix format
- red/yellow/green readiness rules
- first-pass audit snapshot for the current pack

## Dependencies

- GV-25
- all existing micro-plans

## Proof

- the pack can be re-audited later without rereading every plan from scratch

## Non-Goals

- no implementation scheduling

## Readiness Rules

- `green`: small-model safe, bounded, clear touched surfaces and proof
- `yellow`: mostly bounded but still needs one more clarifying split or typed surface note
- `red`: too vague or too broad for safe execution by a smaller model

## First-Pass Audit Snapshot — 2026-05-14

| Group | Plans | Readiness | Notes |
|---|---|---|---|
| Universal foundation | `GV-00`, `GV-00A`, `GV-01` | green | clear engine-neutral story and proof boundary |
| Base extraction and topology | `GV-00B`, `GV-00C` | green | clean future-repo story; enough to route ownership before extraction |
| Dotnet orchestration boundary | `GV-00D` | green | clear language/runtime ownership story for future slices |
| WoW/WC3/Museums profile seeds | `GV-02` to `GV-06A` | green | small and profile-scoped; profile/personality contract now explicit |
| Roots and artifact/data seams | `GV-07` to `GV-10A` | green | good split between roots, sources, catalogs, raw capture, and sidecar schema |
| Generic import/export | `GV-11`, `GV-12`, `GV-13` | yellow | bounded, but should gain explicit touched-surface lists later |
| Museums-specific import/export | `GV-11A`, `GV-12A` | green | concrete and profile-specific |
| Render/runtime layer seams | `GV-14` to `GV-18` | yellow | still abstract enough that future implementation slices should cite exact namespaces/files |
| Audio subsystem seams | `GV-14A`, `GV-14B`, `GV-14C`, `GV-14D`, `GV-17A`, `GV-17B` | green | explicit BASE vs profile split, plus decoded-audio vs MIDI-bank split |
| Web delivery seam | `GV-17C` | green | keeps browser/output ambition explicit without weakening native backend ownership |
| Metadata/editor seams | `GV-19`, `GV-19A`, `GV-20`, `GV-21`, `GV-22` | green | editor story is now well split |
| Generated/distilled content | `GV-23`, `GV-23A` | yellow | intentionally exploratory; keep proof claims narrow |
| Governance | `GV-24`, `GV-25`, `GV-26` | green | enough to police future plan quality |

## Immediate Audit Corrections Still Worth Doing

1. Add explicit `touched surfaces` sections to the yellow plans before implementation starts.
2. When a yellow plan is first implemented, split it again if more than 2 to 3 files/modules would need simultaneous ownership.
3. Keep Museums storage/index questions separated from Museums object-package questions until the backing-store choice becomes real.
