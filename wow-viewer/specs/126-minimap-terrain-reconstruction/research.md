# Phase 0 Research: Experiments That Gate the Build

**Feature**: 126-minimap-terrain-reconstruction | **Date**: 2026-08-02

Four experiments. None trains a model. All four can be completed in hours, and any of them can
reshape the architecture. **No model is trained until all four report.**

Each experiment below fixes its decision thresholds *before* the run. That is the point: a threshold
chosen after seeing the number is not a threshold, it is a rationalisation.

---

## 0. Calibration: what "good enough" means here

This is a hobby restoration project, not an attempt to reproduce anyone's IP exactly, and not
metrology. **The project cutoff is 0.75** (user, 2026-08-02). Past that the returns do not justify
the training time.

| Band | Meaning |
|------|---------|
| **>= 0.75** | **Project target met. Ship it and move on.** |
| **0.60 - 0.75** | Partial. Useful as a prior for a human or a downstream model, not as a final answer. |
| **< 0.60** | Not yet working. Diagnose before scaling. |

Read as Pearson correlation of recovered relief against real MCVT on held-out tiles, unless a
specific experiment states otherwise.

**100% is explicitly not the goal and no gate requires it.** Several outputs cannot reach it in
principle — the codec floor, shading saturation on back-facing slopes, and genuinely flat terrain all
bound it from above. More importantly, chasing the last few points costs training time that this
project does not want to spend. A run that reaches 0.75 is finished, not a starting point for
tuning.

### Measured: synthetic vs authored appearance (2026-08-02, 83 random Azeroth tiles)

Recorded so it is not re-litigated. Empty authored *and* empty synthetic tiles excluded; split by
`liquid_mask`.

| Region | Synthetic RGB | Authored RGB | Gap | Contrast (std) |
|--------|--------------|--------------|-----|----------------|
| Land | 111.1 / 98.1 / 62.0 | 100.1 / 90.1 / 55.3 | -11.0 / -8.0 / -6.7 | 48.3 vs 55.4 |
| Water | 113.2 / 102.8 / 61.6 | 51.6 / 132.7 / 146.6 | -61.6 / +29.9 / **+85.1** | 32.0 vs 16.5 |

**On land the difference is a mild, near-uniform brightness offset** — our renders are roughly 8-11
lighter across all three channels, with about 15% less contrast. It is not a colour-balance problem
and it does not warrant a lighting recalibration. DXT1 accounts for only -0.56 of it (measured
directly: the round-trip is very slightly *darkening*, never brightening), so the offset is real
rather than codec — but it is small, uniform, and **removed for free by per-tile input
normalization**. That is the response, not a re-render.

**Water is a separate and much larger discrepancy**, and it is a palette problem rather than a
lighting one: authored 0.5.3 water is cyan-teal (52/133/147) while our synthetic water is nearly
identical to our synthetic land — we are barely differentiating it. This restates the already-recorded
era-scoped water finding. Water carries no terrain relief and is excluded from relief scoring, so it
does not block this feature; it is logged here so the number is not rediscovered as a surprise.

**Consequence for E1's elevation reading.** The best-fit sun came back at 48 degrees against a traced
client band of 20-37. Given that the land-side appearance gap is a uniform ~8-11 offset rather than
the directional redistribution a badly wrong sun would produce, this is **not** treated as a blocking
discrepancy. Normalize the input, mask the water, move on.

### The two hard problems, named

Everything else in this feature is scaffolding around two inversions:

1. **Minimap RGB -> residual (shading field).** Strip the albedo and the objects, recover the
   terrain-shadow field underneath. A model for this already exists and is now runnable
   (`residual_extractor_v125`, 1.56M params) but has **never been trained**.
2. **Residual (or minimap) -> heightmap.** Turn the shading field into relief.

E1 and E2 exist to tell us how hard each of those actually is before we commit to an architecture for
them. If E1 says the residual is a clean function of the terrain normals, problem 2 is a
well-conditioned inversion of a known law. If it does not, problem 2 becomes blind regression and the
whole justification for this spec over prior attempts weakens.

---

## E1 — Is the residual actually Lambert shading over the real terrain normals?

**Clears**: R2. **Build cost**: zero. The script exists and has never been run on real data.

### Hypothesis

The textureless residual is the compositor's shading output, so it should be well explained by
`clamp(dot(N, L), 0, 1)` over the client's own MCNR normals, under a single per-corpus sun direction,
with an affine gain/ambient term.

### Detector

`scripts/v50_measure_residual_shading_law.py`. It correlates the residual against Lambert shading of
`normal_xyz` (257x257x3 MCNR normals streamed from the client) and **sweeps azimuth and elevation
rather than assuming a sun**, so it recovers the compositor's sun as a by-product.

### Proof of detector power (already obtained)

`--self-test` plants a known sun in a synthetic residual built from known normals and requires the
sweep to recover it. It currently recovers **azimuth 225.0 / elevation 30.0 exactly**, r = 0.99999,
fitted gain 0.699 against a planted 0.700. A null result from this detector is therefore meaningful.

This matters because the previous hillshade check could **not** have confirmed the hypothesis even if
it were true: it derived normals with pixel-unit `np.gradient` (a tile is 533.333 world units across
256 pixels, so gradients were ~2.08x too steep, and the normal is nonlinear in the gradient so the
factor does not cancel), and it pinned the sun at ~45 degrees when the traced client sun is low.

### Decision thresholds (fixed in advance)

| Result | Reading | Action |
|--------|---------|--------|
| r >= 0.85 **and** best-fit elevation in 20-37 deg | Premise confirmed. The residual is the shading field, and our sun model agrees with the compositor. | Proceed as specced. Problem 2 is inversion of a known law. |
| r >= 0.85, elevation **outside** 20-37 deg | Shading law holds but our sun model disagrees with the compositor. | Reconcile the sun before training anything that assumes it. Not a stop. |
| 0.60 <= r < 0.85 | Shading present, something else material is unmodelled. | Inspect the `r` vs `r_lit` gap: a large gap means cast-shadow occlusion dominates the residue, which is informative rather than fatal. Proceed with cast shadows modelled explicitly. |
| r < 0.60 | The residual is not the shading field we believe it is. | **STOP and reassess.** This removes the central advantage of this spec over prior attempts. Do not proceed to Phase 4 on the assumption that a bigger model compensates. |

### What changes if it goes the other way

A low `r` does not kill the feature, but it converts problem 2 from "invert a known law" into "learn
an unknown transform" — which is what every previous failed attempt was doing. In that case the
correct response is to find what the residual *is* a function of (test against slope magnitude, against
cast-shadow visibility alone, against ambient occlusion) before committing to an architecture.

### Command

```powershell
uv run python scripts/v50_measure_residual_shading_law.py --self-test
uv run python scripts/v50_measure_residual_shading_law.py `
    --residual-dir <residual-output>\tiles `
    --store <v50-store> `
    --map Azeroth `
    --output out\spec126\e1_shading_law.json
```

---

## E2 — How much of a minimap is shading, and how much is texture?

**Clears**: R1. **Build cost**: one C# render pass (FR-001) plus one measurement script.

### Hypothesis

`minimap ~= albedo (*) shading`. The shading half carries all the terrain shape. If shading accounts
for a meaningful share of minimap variance, a model can read relief from the raw minimap; if albedo
dominates, the shading is a small modulation riding on a large signal and must be separated first.

### Detector

Add the unlit-albedo pass — composited terrain texture, lighting flat, no objects — symmetric to the
existing `--textureless-residuals`. Then decompose per tile and report the variance share.

### Proof of detector power (required before any conclusion)

Two checks, both mandatory:

1. **Recombination.** `albedo (*) shading` must reproduce the full synthetic minimap within tolerance.
   If it does not, the decomposition is wrong and the split means nothing.
2. **The albedo pass must be genuinely unlit.** A pass that accidentally retains lighting still passes
   the recombination check while making the split meaningless. Verify by rendering the same tile with
   two different sun directions: the albedo output must be **identical**, the shading output must not.

The second check is the one that is easy to skip and fatal to omit.

### Decision thresholds (fixed in advance)

| Shading share of minimap variance | Reading | Action |
|---|---|---|
| >= 30% | Shading is directly readable from the raw minimap. | Albedo removal is a refinement, not a gate. Height model may take raw minimap. Iterative refinement stays P3. |
| 10 - 30% | Albedo is dominant but shading is recoverable. | Albedo removal moves onto the critical path. Train the extractor first, feed shading to the height model. |
| < 10% | Shading is a small modulation on a large albedo signal. | **Single-pass height from raw minimap is unlikely to work.** Promote iterative refinement (US7) from P3 to mandatory, and make the residual extractor the primary model rather than a preprocessing step. |

### What changes if it goes the other way

A very low shading share does not mean the information is absent — it means it is low-amplitude, and
low-amplitude signal under an 8-bit codec is exactly where DXT1 quantisation bites hardest. In that
regime E3's codec fidelity stops being a hygiene check and becomes a first-order constraint on the
achievable ceiling.

---

## E3 — Does our DXT1 degradation match the authored one?

**Clears**: R4. **Build cost**: wire an existing, never-called function to the CLI.

### Hypothesis

Our encoder reproduces the authored degradation *class*, which is all that matters for domain
matching. It does not need to be Blizzard's encoder.

### Detector

`Dxt1TileCodec.RoundTripAgreement` — decode an authored tile, re-encode it, and measure the fraction
of 4x4 blocks whose bytes match the authored blocks, exploiting DXT1's near-idempotency on
already-decoded data. It exists, is unit-tested on the negative case, is **not wired to any CLI**, and
has never been run against authored bytes. Plus the distribution checks: unique-colour count and
block-edge ratio on **real** synthetic corpus output.

### Proof of detector power

Near-idempotency is the assumption the metric rests on and it is measured, not assumed: the Python
codec's second round-trip moves pixels less than half as far as the first. The negative control is
already covered — a non-BLP input returns 0 agreement.

### Decision thresholds (fixed in advance)

| Result | Action |
|--------|--------|
| Block agreement >= 0.95 | Encoder matches. Done. |
| 0.70 - 0.95 | Same degradation class, different endpoint heuristic. Acceptable for domain matching; record the number rather than claiming a match. |
| < 0.70 | Our bounding-box endpoint fit differs materially. Switch to PCA-axis endpoint selection and re-measure. |
| Unique-colour median outside the authored 1196-5269 band on real synthetic output | Degradation is mis-calibrated regardless of block agreement. Fix before training. |

### Note

Current parity is bit-exact between our C# and Python codecs, and validated on synthetic fixtures
only. That proves the two implementations agree with each other — it says nothing about whether
either agrees with Blizzard. This experiment is what turns an assumption into a number.

---

## E4 — Do layer masks derive from terrain shape?

**Clears**: R3. **Build cost**: two measurement scripts, reusing the archived relational-layers
spec's design.

### Hypothesis

Terrain texture painting followed shape — rock on steep slopes, snow at altitude, sand at shoreline.
If so, albedo is partly a *function of* height, the two signals are mutually informative, and texture
decode inherits a strong geometric prior.

### Detector

Fit a mapping from surface properties (slope magnitude from `normal_xyz`, altitude from `height_257`,
curvature) to per-layer coverage from `alpha_256` / `mcly_layer_mask`. Report explained variance per
layer. Second measurement: whether texture identities are consistent per *slot* or per *family*
across tiles, which decides the decode vocabulary.

### Proof of detector power (MANDATORY — this is a repeat offence)

The archived spec explicitly recorded that its earlier linear test was **underpowered** and its result
inconclusive. Repeating an underpowered test and recording a null is the specific failure mode this
project has hit more than once.

Before any negative finding is recorded, the detector must recover a **planted** coupling: synthesise
layer coverage as a known function of slope and altitude, add realistic noise, and confirm the fit
recovers it. A detector that cannot find a planted coupling cannot report its absence. If the linear
fit is underpowered, escalate to a non-linear fit before concluding, and **report "underpowered", never
"no coupling"**.

### Decision thresholds (fixed in advance)

| Explained variance, dominant layer | Reading | Action |
|---|---|---|
| >= 0.50 | Strong coupling. | Albedo and height are mutually informative. Justifies joint modelling and promotes the iterative loop. Texture decode gets a geometric prior; tiers 3-4 become substantially more reachable. |
| 0.20 - 0.50 | Partial coupling. | Use shape as an auxiliary input to decode, not as its backbone. |
| < 0.20 **with proven power** | Texture authored independently of shape. | Decode relies on colour alone. Tiers 3-4 get harder; feeding texture to height is less likely to help, and Phase 5's contribution measurement (FR-021) may legitimately come back negative. |
| < 0.20 with **unproven** power | Nothing learned. | Not a result. Fix the detector and re-run. |

---

## E5 — Is the object capture library still lit correctly?

**Clears**: R9. **Build cost**: a sample re-render plus a diff. Cheap to measure, expensive to fix —
which is exactly why it belongs in Phase 0 rather than being discovered at Phase 3.

### Why this exists

Objects are **lit by the same model the terrain is**. The lighting work that produced correct terrain
shadows — the solar-elevation correction (the horizontal magnitude was pinned at 0.5, so elevation
was computed as `atan(z/0.5)` and came out roughly twice too high), the minimap lighting calibration,
and the terrain cast-shadow path — changed the illumination that objects receive too.

The per-object capture library (spec 077 lineage: `capture_rgb`, `capture_mask`, `capture_alpha`,
`assets.parquet`) was rendered **before** those corrections. Any synthetic minimap that composites
objects from stale captures therefore carries object appearance that authored tiles do not.

That is the same class of problem as the DXT1 codec gap, and it has the same fix: correct it in the
dataset. It matters because objects are in the **model input**, not just the supervision.

### What is NOT affected

Phase 0's core experiments are safe. The textureless residual pass and the unlit-albedo pass both
render **without objects**, so E1 and E2 are unaffected by object lighting. Phase 1 is also specced
object-free. R9 first bites at Phase 3, when objects enter.

### Hypothesis

Object appearance under corrected lighting differs from the stale captures by more than the codec
noise floor — in which case the captures must be regenerated before objects are used as input.

### Detector

Re-render a sample of library objects under the corrected lighting model, and measure three things:

1. Stale capture versus corrected re-render — how much did the correction move object appearance?
2. Each of those versus the corresponding object regions in **authored** minimap tiles — which one is
   closer to the truth we are trying to match?
3. Both differences expressed relative to the DXT1 quantisation floor, so "different" is judged
   against the noise we already accept rather than against zero.

### Proof of detector power

The comparison must be able to detect a lighting change of known size: re-render one object under two
deliberately different sun elevations and confirm the metric separates them by more than it separates
two renders under identical settings. Without that, a small measured difference cannot be
distinguished from render nondeterminism.

### Decision thresholds (fixed in advance)

| Result | Action |
|--------|--------|
| Stale-vs-corrected difference is **below** the DXT1 noise floor | Stale captures are usable. Record the number and proceed; no re-render needed. |
| Difference above the codec floor, and corrected renders are **closer to authored** | **Re-render required before Phase 3.** Schedule the harvest early — it is a long user-run job and it must not become a Phase 3 blocker discovered late. |
| Difference above the floor, but stale captures are **closer to authored** | The lighting correction does not transfer to objects as expected. Investigate before regenerating anything — this would mean the object lighting path diverges from the terrain path. |

### Why this is in Phase 0

The measurement is cheap. The remedy is a long harvest run the user has to execute. Knowing at the
start of the project rather than at Phase 3 is the difference between scheduling a run and being
blocked by one.

---

## Cross-cutting risks resolved inside build phases

### R5 — Height has never beaten the tile-mean baseline

Recorded in the corpus notes as still unbeaten. This plan does not assume a larger model fixes it;
Phase 1 exists to test exactly that, on the easiest possible configuration, before anything expensive
is built. Gate: relief correlation >= 0.85 on held-out (the "acceptable" band above), reported per
tile with the failing fraction stated explicitly.

### R6 — Seams versus per-tile normalization

Per-tile min-max normalization makes each tile's height independent **by construction**, so seam
discontinuity is the expected default, not a bug to be discovered.

Candidate mechanisms, to be decided in Phase 6:

1. **Predict a gradient field and integrate globally** over the submitted region. Continuity becomes
   structural rather than post-hoc, and it matches the physics — shading encodes the gradient, not the
   height. **Leading candidate.**
2. Overlap-tile inference with blended seams. Simple, hides rather than solves.
3. Region-scale normalization instead of per-tile. Breaks the altitude-invariance property the
   relative-height contract exists to guarantee.
4. Per-tile offset regression against the WDL prior. Reuses existing work; limited by WDL resolution.

### R7 — Occlusion masks

**Revised down from the original framing.** The occlusion-correct signals already exist in the v50
manifest: `object_geometry_visible_mask_257`, `object_geometry_visible_instance_257`,
`object_geometry_visible_source_257`. This is verification work, not construction.

What must be verified in Phase 3: that they are populated on the real corpus, and that they mark
terrain *actually hidden in the rendered view* rather than full ground footprints.

#### Why masking was previously unusable, and why it is not any more

The earlier attempt at object masking **ate too much data to train on**. That was a property of the
mask that was available at the time, not of masking as an idea: `object_precise_mask` is each
object's full ground footprint, which removes on the order of 80-90% of the terrain under an object
even though only the genuinely occluded part carries no evidence. Masking with it meant discarding
most of a tile, so the choice looked like "hallucinate under objects" versus "have no data".

The `object_geometry_visible_*` signals dissolve that trade-off. They describe what the render
actually hid, so the excluded region is the region that genuinely contains no ground evidence, and
everything else — including most of the terrain inside an object's footprint — stays supervised.
**The same tiles become usable at far higher coverage**, from data that already exists.

This is why Phase 3 measures coverage as a first-class number rather than assuming it:

| Measurement | Why it matters |
|-------------|----------------|
| Retained-terrain fraction under `object_geometry_visible_mask_257` | The quantity that made the old approach unusable. Must be reported per tile. |
| Same fraction under `object_precise_mask` | The old behaviour, measured on the same tiles, so the improvement is a number rather than a claim. |
| Tiles pushed past the occlusion threshold by each | How many tiles each mask costs us outright. |

**Do not substitute `object_precise_mask` for the visible mask.** It is retained in the store for
this comparison and for full-footprint reasoning, not as a supervision mask.

### R8 — The per-object library is outside the v50 dataset contract

The per-object capture library exists and is substantial — `capture_rgb`, `capture_mask`,
`capture_alpha`, `assets.parquet` (one row per library entry), `index.parquet` (one row per capture
variant), with provenance and explicit `capture_status=not_attempted` rows for jobs whose artifacts
are missing. It carries object identity and appearance that this feature wants for occlusion
attribution and for object-aware reconstruction.

It is **not part of the v50 dataset spec**. It was built under a different lineage with its own
contract, so nothing in the v50 store references it and no v50 tooling can join against it.

**Decision: bind it as a v50 sidecar rather than merging it into the base store.** Rationale:

- The base v50 store is one row per *tile*; the library is one row per *object capture variant*.
  They have different grains, and forcing one into the other would either duplicate captures per tile
  or lose variants.
- The library changes on a different cadence than the tile corpus. A sidecar can be regenerated
  (see R9) without rebuilding the tile store.
- The v50 provenance contract already models per-signal availability and migration policy, so a
  sidecar can declare itself without weakening the base store's guarantees.

What the sidecar must provide: a join from `object_geometry_visible_instance_257` values to library
asset identity, so a masked pixel is attributable to a *known object*, not just to "something".
That join is what turns occlusion masking from a blunt exclusion into usable per-object information.

**Open**: whether the instance IDs in the v50 masks and the library's asset keys share a vocabulary,
or whether a mapping table has to be constructed. Verify before Phase 3.

### R9 — Object captures predate the lighting corrections

Covered as experiment E5 above. Summary: objects are lit by the same model as terrain; that model was
corrected after the captures were made; objects appear in the model **input**, so stale captures are
an input-domain gap of the same class as the codec gap. Measured in Phase 0 because the remedy is a
long user-run harvest that must be scheduled early rather than discovered late.

---

## Summary of what is genuinely unknown

| Risk | Status | Cost to clear | Can it reshape the architecture? |
|------|--------|--------------|----------------------------------|
| R2 shading law | **Unrun. Script ready.** | Zero | **Yes — it is the premise** |
| R1 albedo/shading split | Pass not built | Low | **Yes — decides topology** |
| R4 codec fidelity | Unrun, unwired | Low | Bounds the ceiling |
| R3 layer/shape coupling | Unrun, previously underpowered | Low | **Yes — decides decode approach** |
| R5 baseline | Never beaten | Phase 1 | Kills or confirms the feature |
| R6 seams | Undesigned | Phase 6 | Local to multi-tile |
| R7 masks | Signals exist, unverified | Phase 3 | Determines whether Phase 1's numbers are honest |
| R8 object library sidecar | Exists, unbound to v50 | Low | Enables per-object attribution |
| R9 object lighting stale | **Unmeasured** | Low to measure, **long to fix** | Input-domain gap; schedule the re-render early |

The honest position: **we have proof the signals and signal paths are sound, and we do not yet have
proof that the inversion works.** Phase 0 converts five assumptions into five numbers before anything
expensive is built.

The two inversions named in section 0 are where the difficulty actually lives. Everything else in
this feature is plumbing around them — and the plumbing is, unusually, already built. The corpus is
clean-room, the curation layer partitions rather than filters, the codec gap is closed, the occlusion
signals exist, the WDL prior supplies the absolute datum, and the forward model can generate exact
supervision for every intermediate. What remains genuinely unknown is whether shading can be pulled
back out of a minimap, and whether relief can be pulled back out of shading.

Phase 0 does not answer those two questions. It establishes how hard they are before we spend
anything finding out.
