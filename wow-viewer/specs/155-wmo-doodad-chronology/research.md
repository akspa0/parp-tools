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
rejected** — it names one world object for the earliest staged build against an actual 532, because
per-asset containers carry no internal listfile by design. That surface answers "what does this
archive's listfile declare", which is a different question, and it is retained for exactly that role as
the catalogued set.

## 2. Measured corpus sizes

| Build | Catalogued | World objects | Models | Model route |
|---|---|---|---|---|
| 0.5.3.3368 | 42,765 | 532 (per-asset containers, loose tree) | 5,545 | Alpha `MDLX` — reads |
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
the same error shape as a documented tool that did not exist and a corpus reported as 1 of 532. The
contracts make it structurally unrepresentable rather than relying on discipline.

## 4. Where is the positive control? — OPEN

The Mt. Hyjal effect objects render as green smoke because their water-spray texture is missing; the
engine paints untextured geometry neon green in at least every pre-alpha and beta Vanilla build. They
were confirmed in-world by explorers after Classic launched in 2018, and this project's viewer showed
the missing texture reference directly.

Still to establish:

- Which staged build or builds contain them.
- Their asset paths, and whether they are world objects or models.
- The exact reference that fails to resolve.
- **Whether that build's model route reads today.** If it does not, the control cannot fire, and the
  plan is gated here rather than proceeding to build a sweep whose success criterion cannot be
  evaluated.

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
