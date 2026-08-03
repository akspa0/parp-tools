# Weak-Signal Tile Archaeology — capability note

Recorded: 2026-08-03. Status: tooling built, validated, and parked. Not a training dependency.

## What this is

A pipeline that finds and visualizes terrain the corpus throws away: tiles whose relief is
compressed to near-nothing, or absent entirely. Built on 0.5.3, then run against 4.0.0.11927
(Azeroth, Kalimdor, Deephome, Gilneas2, LostIsles). Everything below is measured, not estimated.

## The tools (all `data-harvester`, all era-agnostic)

| script | what it does |
|---|---|
| `v50_tile_inventory.py` | per-tile record: classification, height range, `surviving_height_levels`, MCNR tilt, 4 neighbour keys + their real range, suggested amplification factor. CSV + JSON. Filters nothing. |
| `v50_synthesize_weak_tiles.py` | 4-panel sheet per degenerate tile (autostretched height, raking hillshade, XY-amplified normals, minimap) + a grid mosaic |
| `v50_tile_composite.py` | whole-map renders in 4 modes from ONE read pass: `absolute` / `autostretch` / `restored` / `liquid` |
| `v50_tile_version_diff.py` | pairs two builds' inventories by tile coord; transition matrix + side-by-side pair renders |

Run order: inventory -> synthesize -> composite. Version diff needs two inventories.

**Always pass `--near-zero-band inf` on any non-alpha client.** See the gates section.

## The measurement that matters: `surviving_height_levels`

Count of distinct height values in a tile. Amplitude alone cannot tell a squeezed landscape from a
squeezed nothing — two 0.5.3 tiles both measure ~0.512 range, one holds 2 distinct values and one
holds 27,132. Only the level count separates them.

Bands: `<=1` bit_exact_flat, `<=8` trace, `<=64` coarse_terrain, `65+` rich_terrain.

## Findings

- **0.5.3**: 1756 tiles, 361 degenerate. 205 carry non-zero relief; only 156 are bit-exact flat.
  33 are rich_terrain — shape intact, amplitude crushed. Sub-millimetre relief (5.19e-4 at world
  Z -501) is real: it **continues across tile boundaries**, which per-tile noise cannot do.
- **4.0.0 Azeroth**: 174 degenerate tiles, **zero** bit-exact flat. The weak-signal tiles are almost
  all ocean — seafloor is low-relief by design. The `liquid` vs `absolute` pair is what shows this:
  flat because water hides it, not flat because nothing is there.
- **The alpha squeezed tiles are NOT previews of the shipped game.** Tested on 62 Azeroth tiles that
  were trace/coarse in 0.5.3 and gained detail by 4.0.0: same-tile |r| median 0.119 vs a
  mismatched-pair control of 0.083 — indistinguishable. Power control (rich in both builds) fires at
  **|r| = 0.959** vs 0.248 shuffled, so the test works and the null is real.
  Conclusion: alpha terrain that existed survived nearly intact into Cataclysm; the degenerate tiles
  were built from scratch later. The squeezed data is genuine authored content of something that was
  **abandoned** — lost work, not an early draft.

## Three gates that each silently excluded the data we were hunting

Same failure mode three times. Worth checking for a fourth before trusting any null result here.

1. `RANGE_FLOOR = 1.0` (`height_relative_model.py`) — 149 tiles with real relief below it get a
   training target occupying a median 9.2e-05 of [0,1].
2. `nearZeroBand = 50.0` (`WeakSignalDetector.cs`, ported) — assumes alpha terrain near sea level.
   4.0.0 Kalimdor sits at |Z| p50 = 441, so **all 71** of its compressed tiles were rejected and the
   first run reported zero. Now `is_compressed_range` (band-free) is recorded alongside
   `is_weak_signal` so this can never hide again.
3. `usable = Height257 != null && MinimapRgb256 != null` (`Program.cs:3378`) — `discover-maps`
   excludes maps for lacking a **minimap**, not terrain. Drops Uldum (25 tiles), MaelstromZone (25),
   DeepholmeDungeon (16). This analysis needs only height + normals. The gate is advisory; `build`
   takes `--map` directly, untested whether harvest-stream enforces it.

## Known limitation if this is ever resumed

Post-alpha clients decode height as `float32(rawDelta + baseHeight)`; alpha stores absolute with no
addition. That sum destroys relief below the float32 ULP at altitude (6.1e-05 at |Z|=515, measured
on 4.0.0 Kalimdor). **`surviving_height_levels` therefore undercounts on every non-alpha client** —
false negatives possible, false positives not. To measure a later client faithfully, count levels on
the raw MCVT deltas before the base is added. NOT IMPLEMENTED.

## Why a full-corpus campaign was dropped

Measured 1.4 MB/tile. One late client = 69 harvestable maps, 6,178 tiles, ~8.4 GB. 28 clients
~237 GB against 148.5 GB free on I:. Does not fit, and the ULP issue above would bias almost every
client in it. Revisit only with the delta-based level counting done first.

## Related

- Spec 127 (`127-weak-tile-explorer`) — viewer explorer + wiring
  `WeakSignalDetector.EstimateFactorFromRanges`, which exists with **zero callers**; the viewer's
  auto factor uses a WDL guide and its `epsilon = 0.001` refuses exactly these tiles. Spec drafted,
  unimplemented.
- The one training-relevant output is in `progress.md`: gate curation on `surviving_height_levels`.

## Holes: a third category, and the biggest one

User domain knowledge (2026-08-03), verified against the data:

Burgundy/dark-red in a minimap marks terrain cells flagged as **holes**. Blizzard used the hole flag
for artistic effect and to hide in-development work from players. Outland (`Expansion01`) is mostly
holes — the floating-island look was made by driving the wanted chunks above 0, flagging everything
else as holes, and adding an artificial death floor partway down so anyone who fell was
teleported/bounced rather than seeing the void.

**The geometry behind a hole is NOT removed.** `AlphaWdtReader.TryParseMcnk` reads MCVT under
`if (mcvtRel >= 0 ...)`, which never consults `holeMask`. The hole flag only tells the client not to
draw those quads. So holed terrain is already sitting in our harvested `height_257`.

### The hole colour, MEASURED (not guessed)

**0.5.3 void/hole colour is exactly `#310000` = RGB(49, 0, 0).** Sampled from 9 uniformly-void
tiles: 589,824 pixels, ONE distinct colour, zero variation. It is an exact-match detector, not a
threshold.

**The colour is era-specific** (user, confirmed by scan): matching `#310000` against 4.0.0 finds
almost nothing, so each build needs its own sample before its holes can be counted.

### A retracted claim

An earlier pass used an ad-hoc heuristic (`r > g+25 and r > b+25 and r < 140`) and reported
"51 burgundy Kalimdor tiles, 36 of them rich_terrain = hidden geometry". **That was wrong.** The
heuristic caught reddish TERRAIN — Durotar/Barrens orange, canyon rock — not holes. Scanning for the
exact `#310000` instead finds 19 Kalimdor tiles with any hole pixels, 15 above 50%, and **none of
them rich_terrain**: they are the bit-exact-flat x19-21, y12-16 block. There is no hidden geometry
behind holes in 0.5.3 Kalimdor.

### Per-era void colours, sampled from uniformly single-colour tiles

| build | colour | what it actually is |
|---|---|---|
| 0_5_3_3368 | `#310000` RGB(49,0,0) | **hole/void** (15 Kalimdor tiles) |
| 0_5_3_3368 | `#68C1E1` RGB(104,193,225) | cyan ocean (106 tiles) |
| 4_0_0_11927 | `#001D29` / `#001D28` | **deep ocean, NOT void** |
| 4_0_0_11927 | `#4F8EFF` | bright ocean (LostIsles) |

`#001D28` differs from `#001D29` by 1 in blue — DXT1/RGB565 quantisation of one colour.

### A second retraction, same root cause

Matching `#001D29` on 4.0.0 found 233 Azeroth / 281 Kalimdor tiles above 50%, of which 185 and 119
were rich_terrain — which looked like a large hidden-geometry discovery. It is not. Cross-checking
against `liquid_mask`: **all 514 are under liquid, zero dry.** That colour is deep ocean, and the
"finding" was the already-known ocean-floor result relabelled.

**The 4.0.0 hole colour remains unidentified. No case of rich geometry behind a hole has been
demonstrated in either era.**

### Why colour inference cannot settle this

A minimap pixel records what the client DREW. "Drew nothing (hole)", "drew dark (deep water)" and
"drew dark (unlit)" are not separable from the pixel alone — each attempt needs another discriminator
bolted on, and the 0.5.3 burgundy heuristic vs 4.0.0 ocean shows the constant does not even
generalise across eras.

`hole_mask_16` is the real answer: the MCNK header `0x40` bitmask, already produced by
`AdtTensorPackBuilder` (`HoleMask16`) and present in NEITHER store. It gives an exact per-CHUNK mask
instead of a per-tile colour guess. The mechanism is verified — `TryParseMcnk` reads MCVT under
`if (mcvtRel >= 0 ...)` with no reference to `holeMask`, so geometry can survive a hole flag — but
verifying the mechanism is not the same as finding an instance, and none has been found.

Lesson, twice in one session: an eyeballed colour threshold is not a detector. Sample the constant,
then find a discriminator that rules out the alternative explanation.

This is a third category, distinct from the other two:

| category | geometry | why it is invisible |
|---|---|---|
| white_plate | absent | nothing was authored |
| weak_signal | present, amplitude crushed | squeezed by the editor |
| **holed** | **present, full fidelity** | **flagged not to render** |

### The gap that blocks using it

`holeMask` is read from the MCNK header at **offset 0x40** — a DIFFERENT field from
`mcnk_flags_16` (offset 0x00), so it cannot be derived from anything in the v50 stores. The C#
`AdtTensorPackBuilder` already produces a `hole_mask_16` signal (`HoleMask16`), but it is in
NEITHER the 0.5.3 nor the 4.0.0 store.

So the hidden terrain is already harvested and we cannot currently say which tiles it is. Carrying
`hole_mask_16` through the harvest stream into the v50 store is the whole fix — no re-harvest of
geometry needed, and it turns "36 tiles that look burgundy" into an exact per-chunk mask.

Same pattern as `HasSparseChunks`/`ActiveChunkCount`: computed by the reader, dropped before the
store. That is now four dropped-or-single-gate signals found in one session.

**Expansion01 (Outland, 826 tiles) is the highest-value target** for this once hole_mask_16 lands.

## Engine constraints (user domain knowledge, 2026-08-03)

Context for why the nesting exists and why it never changed:

- The engine still addresses **16x16 cells per chunk, per ADT, in 2026**. Blizzard scaled up objects,
  players and cameras to get more visual density rather than raising the **64x64 = 4096 tiles per
  map** ceiling. That ceiling looks hardcoded, inherited from the "worlds" -> "maps" rename around
  2000-2001, and unescapable without redesigning the map system wholesale.
- WoW began as a Warcraft 3 script-modded map; WC3 engine limits carried through every rewrite. The
  0.5.3 MDX format is a slightly newer, experimental WC3 MDX.
- WMO is Blizzard's answer to multi-nested/connected Quake 3-style BSPs, scaled to 384 groups by end
  of 2003. **One WMO group is roughly one Quake 3 map.** The dungeon generator used BSPs the same
  way, likely taking Q3 engine limits as design constraints.
- **The earliest WMOs were authored in Radiant**, the Quake 3 level editor.

So the hierarchy the hole masks decode (tile -> chunk 16x16 -> quad 4x4) is not incidental; it is a
20-year-old fixed addressing scheme. Signals are worth probing at every level of it, not just the
tile.

## Confirmed: a developer asset staging area at the Azeroth map corner

`nsabbey.wmo` (the Radiant-authored Northshire Abbey) is placed TWICE in 0.5.3 Azeroth: at
(32,48) in Elwynn where it belongs, and at **(0,2) in the map corner**.

That corner is a staging area, found by `v50_tile_mismatch.py` with no prior hint:

| tile | WMOs | doodads |
|---|---|---|
| (0,2) | `nsabbey.wmo`, `dsnightelfmoonwell.wmo` | 8 |
| (1,0) | `sunkentemple.wmo` | 35 |
| (1,2) | `orcbarracks.wmo` | 392 |

Cross-continent assets (Kalimdor buildings in the Azeroth map) and a dungeon interior, parked in the
corner. `Azeroth_02_02` scores highest for signal mismatch in the whole map: it has height, normals,
liquid and object masks but NO `alpha_256`, `mcly_layer_mask`, `mcly_texture_ids`, minimap or
`shadow_mask` — untextured terrain carrying placed objects, which is what staging looks like.

## The mismatch detector (`v50_tile_mismatch.py`)

Inverts the usual method. Instead of starting from a hypothesis, it measures how every pair of
signals co-occurs across the corpus, derives near-universal `A -> B` rules (support >= 30,
confidence >= 0.95, violations <= 10%), and ranks tiles by how many rules they break, weighted by
confidence. Picks up every per-tile array automatically — no hand-picked signal list.

Validation: it independently rediscovered `Azeroth_42_39`/`42_40` (the "minimap but no terrain"
tiles found by hand earlier) purely because they break a rule 640 other tiles obey.

Known limit: presence is "any non-zero", so a populated-but-degenerate signal reads as present. It
catches missing-vs-present, not weak-vs-strong. Pairing it with `surviving_height_levels` would
catch the second kind.

## Origin-era context (user domain knowledge, 2026-08-03)

Recorded because it is otherwise undocumented anywhere in the repo and it shapes what the data IS.

- The game began as **ONE world named Azeroth**, fitting in a container smaller than 64x64 tiles.
  When it outgrew that it was split into separate maps in separate map folders.
- The only surviving artifact of that 1999-era single-world build is the original **`world.def`**,
  containing nothing but day/night light-cycle data for the world sun — plus a hand-drawn blockout
  map showing **both continents in one frame** with names that never shipped (Amberhorn Caverns,
  Jaedenar as a dungeon, Kobold Lair, Tauren-Newbie, Stonetalon Peak) and difficulty tiers already
  assigned, and a handful of other images from the same period.
- **PM4 is NOT distilled**, unlike terrain. Shipped ADT resolution is plausibly 1/4 or 1/16 of the
  internal authoring resolution, and the undistilled PM4s are the basis of that estimate. PM4 is
  therefore the highest-fidelity surviving record of object placement and should be treated as the
  reference, not a derived by-product.
- **PM4 holds the NEGATIVE BSP** for every placed object on a tile: the surfaces players and NPCs
  walk on and that pathfinding uses. It carries no WMO geometry yet records more per object than the
  model file does. This reframe is Spec 128.

Open hypotheses the user holds, not yet tested:

- Terrain brushes may be procedural with **audio or other signals** as input — a 2000-era studio had
  audio CDs on hand, and The WoW Diary dwells on what the team listened to. Two testable
  discriminators exist and neither has been run: (1) an 8-bit grayscale scan source would pile tile
  level-counts at or below 256, and `surviving_height_levels` is already in `tiles.csv` for all 1817
  tiles — the histogram is unplotted; (2) a brush derived from a 1D waveform must be swept/revolved
  into 2D and would leave **anisotropy**, whereas a scanned drawing or 2D fractal is isotropic, so
  row-wise vs column-wise autocorrelation on reused brush patches separates them.
- The competing account is that hand-drawn maps were scanned and the pencil sketch guided a
  grayscale depth map per tile.

## The jigsaw hypothesis: weak tiles are displaced older terrain (user, 2026-08-03)

The user's account, recorded verbatim in substance because it reframes the whole weak-tile corpus
from "damaged leftovers" into "a recoverable earlier map."

**The claim.** Weak tiles are not noise. Many of them — especially in 0.5.3 — are **leftover terrain
from earlier versions of the game**, stranded when the map was enlarged. In November 2001 the world
grew substantially; zones were shifted around, and terrain that had been in-bounds got pushed out
into what now reads as weak signal. It still borders the later terrain it was separated from.

**Why it should be reassemblable.** Across the weak tiles there are *a lot of similar patterns,
randomly plastered at different relative positions on different tiles*. That repetition is the
handle: if the same pattern appears at different offsets on different tiles, the offsets are
evidence about how the pieces were displaced. The corpus should behave like a **jigsaw puzzle** —
find adjacent agreements between piece edges and fit them back together.

**Weak tiles may join non-weak tiles.** The reassembly is not weak-to-weak only. A displaced older
piece is expected to abut current terrain, so edge-matching must consider the full corpus, not just
the weak subset. **There is shadow proof in at least one instance** — a case where shadow evidence
ties a weak tile to terrain it must once have adjoined.

**What the user wants.** Automation, and soon. This is sitting unexploited. The goal is to restore
as much of the earlier terrain as the surviving data allows, using **what is already on hand
first** — the harvested corpus, before any new capture.

### Why this is credible rather than speculative

It predicts things that are checkable with what is already built:

- If pieces were displaced by a map-resize, displacement offsets should **cluster**, not spread
  uniformly. A histogram of best-match offsets is the first test and needs no new harvest.
- Edge agreement is measurable directly: heights along a tile edge either continue into a neighbour
  candidate or they do not, and `tiles.csv` already carries the per-tile data to try it.
- The single shadow-proof instance is a **known-answer case** — the detector must recover that pair
  before any of its other proposals are trusted. That is the detector-power gate this project
  already requires, and here it comes for free.

### Hazards to design against, learned from this same corpus

- **Verify detector power before reporting a null.** A matcher that cannot rediscover the known
  shadow-proof pair has not shown that other pairs are absent.
- **Repeated brush patterns are a confound, not only a signal.** This corpus is already known to
  contain fractal brush copy-paste reuse, so two tiles can share a pattern without ever having been
  adjacent. Edge continuity must outrank pattern similarity, and a match supported only by a reused
  brush must be reported as weak.
- **Partition, never filter.** Proposed fits, rejected fits and unmatched tiles all stay queryable.
- **`surviving_height_levels` gates trust.** Tiles holding very few distinct heights cannot support
  edge matching; four Azeroth tiles hold exactly 2 across a 516-unit range. Exclude them from
  matching rather than letting them match everything.

Status: hypothesis recorded, nothing built. Next concrete step is the offset-cluster histogram and
the shadow-proof known-answer case, both runnable against the existing corpus.

### Direction of displacement — user's visual reading (2026-08-03, second pass)

The user has eyes on this data and reports specific, directional structure. These are observations
to test, not yet measurements, but they are far more constrained than "things moved":

- **Weak-signal terrain predates the tiles two rows below it, and was shifted UP.** So displacement
  is not random scatter: at least one family of pieces moved by a **whole number of rows in +row**,
  and the pre-image is still present two rows away.
- **Some border tiles belong to the TOP of the map, not the sides.** A tile sitting on a side border
  whose content reads as top-of-map content means pieces were relocated **across edges**, not merely
  nudged.
- **It happened many times, not once.** Tiles were added to the quilt repeatedly, so there is no
  single global offset to solve for — expect a **small set of discrete displacement events**, each
  with its own offset, layered over each other.
- **Earlier overhead shots show implied terrain that is missing later** — and it *sorta matches*
  weak-signal tiles found on the **opposite border edge** in the data we hold. That is the
  strongest single lead: an external, dated reference image constrains both the content and the
  direction of travel.

### What this changes about how to search

The naive formulation — "score every tile pair for edge agreement" — is O(n^2) over ~1817 tiles and
mostly wasted. The observations above turn it into a much smaller search:

1. **Solve for offsets, not pairs.** For a candidate displacement (drow, dcol), score the whole
   corpus at once by testing whether weak tiles at (r, c) agree with terrain at (r+drow, c+dcol).
   A real resize event shows up as a **spike** at its offset. This is a vote over a small integer
   grid, not a pairwise search.
2. **Expect several spikes, not one** — one per resize event. Rank them and peel them off in order.
3. **Row shifts are the first thing to try**, because the user has already seen a two-row
   relationship. Pure-row offsets are a tiny search space and can be run immediately.
4. **Allow cross-edge relocation.** A side-border tile matching top-of-map content means candidate
   offsets must include large jumps and wraps, not just small neighbourhoods.
5. **The overhead shots are ground truth for direction.** Terrain visible in an early shot and
   absent later, matched to a weak tile on the opposite edge, gives a known-answer pair with an
   external date attached — a second detector-power gate alongside the shadow-proof instance.

Restated as the standing hazard: a matcher that finds a spike must still be shown to recover the
shadow-proof pair and the overhead-shot pair. Brush reuse will manufacture agreement between tiles
that were never adjacent, so a spike supported only by pattern similarity and not by edge continuity
is not evidence of displacement.
