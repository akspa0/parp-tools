# Phase 0 Research: Asset Reference Inventory

**Status**: Sections 1–3 established. Sections 4–7 are open and gate Phase 1.

## 1. Corpus access is solved; one surface is a trap

**Decision**: The corpus comes from the archive access layer, never from archive internal listfiles.

**Rationale**: The archive catalogue already scans the loose tree for per-asset containers, the data
source maps a container back to the logical asset path it holds, and the native archive service
identifies these as listfile-less single-file archives and de-duplicates their double registration so
enumeration does not emit each world object twice. The V14 converter documents that it handles
per-asset containers automatically. The viewer reads this data today.

**Alternatives considered**: Building the corpus from the listfile index cache. **Measured and
rejected** — it names one world object for the earliest staged build against an actual 492, because
per-asset containers carry no internal listfile by design. That surface answers "what does this
archive's listfile declare", which is a different question, and it is retained for exactly that role as
the catalogued set.

## 2. Measured corpus sizes

| Build | Catalogued | World objects | Models | Model route |
|---|---|---|---|---|
| 0.5.3.3368 | 42,765 | 492 (per-asset containers, loose tree) | 5,545 | Alpha `MDLX` — reads |
| 3.0.1.8303 | 131,106 | 9,711 (packaged entries) | 17,296 | `MD20 0x107` — **blocked** |

## 3. Model readability is build-dependent and partly blocked

**Decision**: Model texture sweeping is scoped to routes that read today; blocked builds are recorded
as blocked, never swept and reported as zero.

**Rationale**: Spec 154 measured that model reading fails for builds declaring `MD20 0x100` through
`0x107` and succeeds for the Alpha route and `MD20 0x108`. A sweep over a blocked build would find no
texture references and report no missing textures — indistinguishable in the output from a healthy
build unless blockage is recorded as its own state.

**Alternatives considered**: Waiting for Spec 154 before sweeping models at all. Rejected — the Alpha
route reads today and carries 5,545 models, including the likely home of the positive control, so
useful work is available now. Deferring would also leave the control unexercised, which is the one
thing this plan refuses to do.

**This is the plan's most dangerous failure mode.** "Could not check" rendering as "nothing missing" is
the same error shape as a documented tool that did not exist and a corpus reported as 1 of 492. The
contracts make it structurally unrepresentable rather than relying on discipline.

## 4. Scale of the missing-asset population — MEASURED (2026-08-16)

First full sweep, `0.5.3.3368`, via `assets sweep`:

| | |
|---|---|
| World objects examined | 492 |
| Models examined | 5,545 |
| References collected | 22,025 |
| Resolved via extension substitution | 1,433 |
| Unresolved references | 4 |
| **Distinct missing assets** | **3** |
| Referencing assets unreadable | 1 |
| Report complete | **false** |

### The three missing assets

| Asset | Referenced by |
|---|---|
| `DUNGEONS\TEXTURES\LAVA\BURNINGSTEPPSLAVA02.BLP` | `World\wmo\Blackrock.wmo` |
| `ITEM\GROUNDOBJECTS\GOAXE01.MDX` | `World\wmo\OrcBarracks.wmo` |
| `WORLD\...\TROLLRUINSBASINWALL02\TROLLRUINSBASINWALL02.MDL` | world object doodad reference |

The first is a **missing lava texture** in Blackrock — the same class as the Mt. Hyjal effect objects,
found without being looked for.

### Extension substitution: the false positive that would have buried them

**Decision**: Resolution attempts the shipped compiled extension when an authored source extension does
not resolve, and records the substitution rather than hiding it.

**Rationale**: The first sweep reported **366** distinct missing assets. 364 were `.MDL` references —
and 363 of those have a `.MDX` present in the same build. Authored references name the source format;
the client loads the compiled one. Reporting them as missing would have produced a 366-entry list in
which the 3 real findings were 0.8% of the noise. After accounting for substitution the count is 3, and
1,433 references are reported as substitution-resolved, which is itself a fact about how the data was
authored.

**Alternatives considered**: Treating every unresolved reference as missing. Rejected by the data
above. Silently resolving substitutions without recording them was also rejected — the substitution
count is a finding, not an implementation detail.

### One unreadable asset

`World\wmo\OilPlatform.wmo` fails with a chunk overrun at offset 10592. Its references are therefore
**unknown, not absent**, and the report is marked incomplete because of it. This is a reader defect
worth its own investigation; it is recorded rather than swallowed.

### Corpus count note — RESOLVED (2026-08-16)

The sweep examined **492** world objects. Earlier drafts of this document cited "532" as the loose
tree's true per-asset container count and treated 492 as an unexplained shortfall against it. That 532
figure was never produced by any tool in this project — it predates the `assets`/`archive` CLI surface
existing at all and most likely originated in the manual `H:\CLIENTS` filesystem poking this spec's own
`checklists/requirements.md` already records as a corrected error. It should not have been carried into
this table without being re-derived, and it does not survive verification.

`archive scan-wmo-containers --archive-root "H:/CLIENTS/Vanilla/0.x/0_5_3_3368/World of Warcraft"` was
added specifically to settle this: it calls `MpqArchiveCatalog.ScanWmoMpqArchives` (the exact production
scan the sweep already uses, unmodified) and separately walks the entire game root for
`*.wmo.mpq`/`*.WMO.MPQ` with no directory scoping at all, then diffs the two sets. Result:

| | |
|---|---|
| Scoped production scan (7 candidate directories) | 492 |
| Unscoped whole-root walk | 492 raw files, 492 distinct virtual paths |
| Containers the scoped scan missed | **0** |

The two independent methods agree exactly. **492 is the verified, complete count of per-asset WMO
containers in 0.5.3.3368.** SC-002 is met for this build. The "532" figure is retracted; it does not
appear elsewhere in this document as of this correction.

## 4a. Blocked-route detection, and the three 3.0.1 builds — MEASURED (2026-08-16)

Two implementation gaps flagged as open in the prior entry are now closed:

**`RouteBlocked` is now real, not just a type.** `ModelRouteClassifier` classifies a model's era from its
header (reusing `M2ModelReaderDispatcher.DetectEra`, no new decoding) *before* attempting a full parse.
An era Spec 154 measured broken (MD20 0x102-0x107 via the dispatcher's own refusal; MD20 0x100 Era100
layout; MD20 0x109+) is recorded as `RouteBlocked` and never attempted; an era Spec 154 never measured
either way stays `Unknown` and is attempted normally rather than guessed into either bucket. Blocked
models are aggregated once per route into `SweepReport.BlockedRoutes`, not reported as thousands of
identical per-asset failures.

**WMO group files were being swept as if they were roots.** First sweep of 3.0.1.8303 reported "World
objects: 9711, Unreadable: 8331" — a 86% failure rate that looked like a corpus-wide reader defect. It
wasn't: 8,107 of those were WMO *group* files (`Name_000.wmo`, `Name_001.wmo`, ... — the established
convention this project's own `WmoV14ToV17Converter`/`WmoV17ToV14Converter` already write group paths
with), which carry geometry, not root-level MODD/MOTX references, and were correctly detected as
`WmoGroup` by `WowFileDetector` and then wrongly reported as unreadable roots. `Sweep()` now excludes
them from the root corpus via `AssetReferenceSweeper.IsWmoGroupFile` before attempting anything, and
`SweepReport.WorldObjectGroupFilesExcluded` records the exclusion so the corpus accounting stays fully
explained rather than just presenting a smaller number. This does not apply to 0.5.3.3368 (Alpha's v14
format has no separate group files — geometry is embedded in the single per-asset container) and its
sweep is unaffected (492, 0 excluded).

| | 3.0.1.8303 | 3.0.1.8334 | 3.0.1.8391 |
|---|---|---|---|
| World objects (roots) examined | 1,604 | 1,616 | 1,633 |
| WMO group files excluded | 8,107 | 8,147 | 8,206 |
| Models examined | 17,296 | 17,358 | 17,513 |
| References collected | 42,519 | 42,787 | 43,091 |
| Resolved via extension substitution | 22,502 | 22,655 | 22,782 |
| Unresolved references | 1,323 | 1,317 | 1,302 |
| Distinct missing assets | 286 | 286 | 286 |
| Unreadable | 224 | 362 | 563 |
| Blocked routes | 1 (MD20 0x102-0x107, all 17,296/17,358/17,513 models) | 1 | 1 |
| Report complete | false | false | false |

**Blocked route confirms Spec 154 exactly.** Every model in all three builds classifies as MD20
0x102-0x107 and is blocked — these builds have zero model-texture references available by design, not
by defect. `ModelsExamined` counts them; `BlockedRoutes` says why none were actually read.

**Unreadable count rises 8303 → 8334 → 8391 despite being the "same" patch.** Not yet explained — this
is exactly the class of adjacent-patch difference this spec's own constraints warn about (three separate
3.0.1 builds, never assumed identical). Flagged, not investigated further here.

**What "unreadable" actually is, on inspection**: mostly *models*, not world objects — `unreadableAssets`
mixes both kinds. The dominant pattern in 8303 is real M2 files failing with "has a non-finite
bones[N].pivot.y value" (NaN/Infinity bone pivot data), concentrated in Wrath-era creature/doodad content
(`Item\ObjectComponents\HEAD\...`, `World\Expansion02\Doodads\...`). This is an M2-reader-robustness
question distinct from Spec 154's era-routing findings and is out of this spec's scope — recorded, not
chased. One genuine WMO-side failure was found:
`World\wmo\Kalimdor\CollidableDoodads\Hyjal\WorldTreeRoots\WorldTreeRoots01.wmo` — "MOMT payload size 0
is not compatible with inferred entry size 0," a malformed materials chunk, unrelated to the group-file
issue above.

## 5. Why a full sweep, not a hunt for known instances

Nobody knows how many references in any build resolve to nothing. The Mt. Hyjal effect objects are one
instance that became famous because a person happened to walk past it; that is not a sampling method,
and there is no reason to treat it as rare.

The first sweep answers this. It is not a research question to settle before building — it is the
output of Phase 1, and the count of unresolved references is the headline number.

Known instances are a sanity read on a finished report, not a target. Nothing in this plan is scoped
around locating a particular object, and no phase is gated on one.

## 5. Presence probe semantics — OPEN

A probe must distinguish three outcomes, not two: the asset is present and readable; the asset is
absent; the asset is nominally present but cannot be read. Collapsing the third into the second would
manufacture missing assets. How the archive layer surfaces that distinction must be established before
resolution is implemented.

## 6. Catalogued-set coverage — OPEN

The listfile index is the catalogued set. Its coverage relative to what the archive layer enumerates
must be recorded per build, because the gap between them is itself one of the four reported categories.
For the earliest build the index reports far fewer world objects than exist, and that discrepancy is a
finding to be reported, not a bug to be silently corrected.

## 7. Sweep coverage limits — OPEN

Orphan detection is bounded by what is swept. This feature reads references from world objects and
models only. Assets referenced from anywhere else would appear as orphans. The set of reference sources
*not* swept must be enumerated so the orphan report can state its own limits rather than implying that
an unswept reference does not exist.

## Open Research Boundaries

- Intent is out of reach. A missing asset may be deliberate — the control objects are themselves the
  reason anyone knows that. The inventory reports the fact; nothing in this feature decides whether an
  absence was intended, and repair must never assume it was not.
- Sound and other asset classes are out of scope.
- `uniqueId` is out of scope as a chronology source; it dates placements, not assets.
- Renames cannot in general be distinguished from a removal plus an introduction. Where the system
  cannot tell, it must say so rather than manufacture an introduction.
