# Quickstart: Phase 0 De-Risking

**Feature**: 126-minimap-terrain-reconstruction | **Date**: 2026-08-02

All commands are user-executed. Nothing here is launched automatically. Run from
`wow-viewer\data-harvester` unless stated otherwise.

**Do not skip to Phase 1.** Phase 0 is five measurements, four of which can change the architecture.
Full designs, thresholds, and what-if branches are in [research.md](./research.md).

---

## E1 — Shading law (run this first)

Zero build cost. The script exists and has never been run on real data. It is the premise the whole
feature rests on.

**Step 1 — prove the detector still has power:**

```powershell
uv run python scripts/v50_measure_residual_shading_law.py --self-test
```

Expect `SELF-TEST PASS`, recovered azimuth 225 / elevation 30, r ≈ 0.99999. If this fails, stop — a
null result from a broken detector is worthless.

**Step 2 — run it on real corpus data:**

```powershell
uv run python scripts/v50_measure_residual_shading_law.py `
    --residual-dir <residual-output>\tiles `
    --store <v50-store> `
    --map Azeroth `
    --limit 64 `
    --output out\spec126\e1_shading_law.json
```

Read `mcnr_normals.r` — that is the answer. Compare `r` against `r_lit`: a large gap means
cast-shadow occlusion is the dominant unexplained term.

| `r` | Meaning |
|-----|---------|
| ≥ 0.85 | Premise confirmed. Proceed. |
| 0.60–0.85 | Something material is unmodelled. Inspect the `r`/`r_lit` gap first. |
| < 0.60 | **STOP.** The residual is not the shading field we assume. Reassess before building. |

Also check the recovered elevation is in 20–37°. Outside that band means our sun model and the
compositor disagree — reconcile before training anything that assumes it.

---

## E2 — Albedo/shading split

Requires the unlit-albedo render pass (FR-001), which does not exist yet.

**Once `--albedo-only` is implemented:**

```powershell
# from wow-viewer\
dotnet run --project tools\harvest\WowViewer.Tool.Harvest -- synthetic-minimap `
    --client-root <client-root> `
    --map Azeroth `
    --albedo-only `
    --textureless-residuals `
    --output <albedo-output>
```

**Verify the pass is genuinely unlit** — this check is easy to skip and fatal to omit. Render the
same tile under two different sun directions; the albedo output must be *identical*, the residual
must not:

```powershell
uv run python scripts/v50_measure_albedo_shading_split.py --verify-unlit `
    --albedo-dir <albedo-output>\tiles `
    --output out\spec126\e2_unlit_check.json
```

**Then measure the split:**

```powershell
uv run python scripts/v50_measure_albedo_shading_split.py `
    --albedo-dir <albedo-output>\tiles `
    --residual-dir <residual-output>\tiles `
    --minimap-store <v50-store> `
    --map Azeroth `
    --output out\spec126\e2_variance_split.json
```

| Shading share | Consequence |
|---------------|-------------|
| ≥ 30% | Height model may read the raw minimap. Refinement stays optional. |
| 10–30% | Albedo removal moves onto the critical path. |
| < 10% | Iterative refinement becomes **mandatory**; the residual extractor becomes the primary model. |

---

## E3 — Codec fidelity against authored bytes

`RoundTripAgreement` exists but is wired to nothing. Surface it on `inspect-minimap-blp`, then:

```powershell
# from wow-viewer\
dotnet run --project tools\harvest\WowViewer.Tool.Harvest -- inspect-minimap-blp `
    --client-root <client-root> `
    --map Azeroth `
    --round-trip-agreement `
    --output <survey-output>\e3_codec_fidelity.json
```

Then confirm real synthetic output lands in the authored colour band:

```powershell
uv run python scripts/v50_measure_codec_fidelity.py `
    --synthetic-dir <minimap-output>\tiles `
    --authored-survey <survey-output>\e3_codec_fidelity.json `
    --output out\spec126\e3_codec.json
```

Block agreement ≥ 0.95 means the encoder matches. Below 0.70, switch bounding-box endpoints to a
PCA-axis fit and re-measure. A unique-colour median outside 1196–5269 means the degradation is
mis-calibrated regardless of agreement.

---

## E4 — Do layer masks derive from terrain shape?

**Prove detector power first.** The archived relational-layers spec already recorded that its earlier
linear test was underpowered; repeating it and recording a null would repeat a known failure:

```powershell
uv run python scripts/v50_measure_layer_shape_coupling.py --self-test
```

This plants a known slope/altitude → coverage coupling and requires the fit to recover it.

```powershell
uv run python scripts/v50_measure_layer_shape_coupling.py `
    --store <v50-store> `
    --map Azeroth `
    --output out\spec126\e4_layer_shape.json
```

Explained variance ≥ 0.50 on the dominant layer means albedo and height are mutually informative and
texture decode inherits a geometric prior. Below 0.20 **with proven power** means texture was
authored independently of shape. Below 0.20 **without** proven power is not a result.

---

## E5 — Object lighting currency

Cheap to measure, long to fix — which is why it runs now rather than at Phase 3.

```powershell
uv run python scripts/v50_measure_object_lighting_drift.py `
    --library <object-library>.zarr `
    --authored-minimaps <client-root> `
    --map Azeroth `
    --sample 64 `
    --output out\spec126\e5_object_lighting.json
```

If the stale-vs-corrected difference exceeds the DXT1 noise floor and the corrected renders are closer
to authored, **schedule the object re-render now** — it is a long harvest and it blocks Phase 3.

---

## Phase 0 exit

All five reported, each with its decision recorded against the threshold fixed in research.md:

- [ ] E1 shading law — the premise
- [ ] E2 albedo/shading split — decides topology
- [ ] E3 codec fidelity — bounds the ceiling
- [ ] E4 layer/shape coupling — decides decode approach
- [ ] E5 object lighting — schedules or skips a long re-render

Only then does Phase 1 start, and Phase 1 runs on the easiest possible configuration: synthetic,
object-free, DXT1-degraded input, one tile, relief only. If that cannot beat a tile-mean baseline,
scaling the model will not save it.
