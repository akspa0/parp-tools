# Feature Specification: V24 Full Reconstruction Lattice Model (Spec 098)

**Feature Branch**: `098-v24-lattice-reconstruction`
**Created**: 2026-07-10
**Status**: Vision document (not yet implemented)
**Owner**: wow-viewer
**Parent**: Spec 094 (V24 WDL prior + lattice detailer), Spec 096 (deployment), Spec 097 (per-map export)

**Input**: User description — "we also need to think about the next step - training stages A and B fully, and then the v24 WDL priors as lattice - model for reconstructing the full detailed mesh, including detection for fractals and hand-painted data, so details are properly discerned from any minimap. We need to be thinking about that, too."

---

## Problem Statement

Spec 094 / 096 / 097 build the **infrastructure** for the V24 WDL prior pipeline:
- Stage A: minimap + alpha + normal + mcnr → WDL prior (17×17 outer + 16×16 inner)
- Stage B: WDL prior + minimap + alpha + normal → 257×257 heightmap residual
- Per-map stitched OBJ with edge alignment
- Standalone PNG → WDL prior NPZ deployment

What Spec 098 is about is the **next-level model**: take the WDL prior as a *lattice anchor* and reconstruct the full 257×257 detailed mesh from the minimap, with the prior's coarse structure as a soft constraint. The model must be able to **discern fractal brush detail and hand-painted layer detail from the minimap** — the hard cases where Stage B currently overfits or underfits.

This is the actual full v24 vision. The prior specs are prerequisites; this spec is the work that uses them.

---

## What This Spec Is (Vision, Not Yet Implementation)

This is a planning document. It is the next spec the user wants to attack. It is broken into **sub-specs (099, 100, 101, …)** that each deliver a bounded piece:

1. **Spec 099 — full Stage A training on every V18 build.** The current Stage A is trained only on `3_3_5_12340`. We need to retrain on every build in the V18 store so the model is build-agnostic. Spec 099 is the data prep + training + validation.

2. **Spec 100 — full Stage B training with proper cross-tile consistency loss.** The current Stage B is trained per-tile; the per-map stitched OBJ has seams because adjacent tiles' Stage B outputs don't agree at the border. Spec 100 adds a **border-consistency loss** that enforces the Stage B residual on the 16-pixel border of one tile matches the next tile's border.

3. **Spec 101 — fractal / hand-painted detail detector.** A small classifier (U-Net, ≤ 1M params) that takes a 256×256 minimap + a per-tile V18 substrate and outputs a per-pixel "is this fractal brush detail or hand-painted terrain detail" probability map. The probability map is then a *channel input* to the Stage B trainer — telling the model where to attend to fine detail. Spec 101 is a separate model per [RULE 7](AGENTS.md) (small, modular, residual-predicting).

4. **Spec 102 — V24 reconstruction lattice model.** The full-detail model. Input: cleaned minimap (256×256) + V18 alpha + normal + mcnr + object_mask + the WDL prior as a 5-channel tensor (outer 17×17 + inner 16×16 + the prior confidence/source maps) + the fractal detail map (Spec 101) + the per-tile normal channel. Output: a 257×257 heightmap that matches the WDL prior at the lattice points (within ±1 world unit) and reconstructs fine detail in the cells between lattice points. The model is **constrained by the lattice**: at the (16r, 16c) and (16r+8, 16c+8) sample points, the output must match the WDL prior exactly. This is a hard constraint baked into the model architecture (e.g. the lattice points are passed through unchanged from the prior, the model only fills the in-between cells).

5. **Spec 103 — full-map round-trip via the lattice model.** A pipeline that:
   - Loads a per-map V18 Zarr
   - Runs Stage A → WDL prior per tile
   - Runs Spec 101's fractal detector per tile
   - Runs Spec 102's reconstruction model per tile
   - Stitches with edge alignment (Spec 097 Slice 1's algorithm + Spec 100's border-consistency-aware version)
   - Writes a single OBJ + atlas per map
   - Writes real `.wdl` and `_tex0.adt` per tile (Spec 097 Slices 2/3, retrofitted)
   - Round-trip smoke: the C# WdlRead + AdtRead parse the output without errors

---

## Why This Is Bounded and Achievable

The user has been pointing at this vision for a while. The good news is **most of the pieces already exist**:
- V18 store with all the substrate (alpha, normal, mcnr, object_mask, liquid_mask, height_257, minimap_rgb)
- V24 prior store (Spec 094) with the (17,17)+(16,16) prior grids per tile
- Trained Stage A (cheat + minimap-only) and Stage B checkpoints
- C# WdlRead / AdtRead shims
- Per-map export with edge alignment (Spec 097 Slice 1)
- 40/40 v24 test suite

The new work is:
- The fractal detail detector (Spec 101)
- The lattice-constrained reconstruction model (Spec 102)
- The border-consistency loss (Spec 100)
- A real `.wdl` / `.adt` writer (Spec 097 Slices 2/3)

Each is a real but bounded chunk of work. Spec 099 is data prep + retraining. Spec 100 is a loss-term change + retrain. Spec 101 is a new model. Spec 102 is the main model. Spec 103 is the integration.

---

## Hard Preconditions

Before Spec 098 starts:
- [ ] Spec 097 Slices 2/3 (WDL/ADT writers) are in. Otherwise the round-trip smoke at the end of Spec 098 has nothing to read back. The honest current state is: Spec 097 Slices 2/3 are next-session work; Spec 098 is the work after that.
- [ ] Spec 095 (learned minimap cleaner) — the user's previous observation is that the minimap-only regime is 190 world units L1 vs 1.31 baseline. The cleaner is the difference between the prior being a deployment tool vs a research artefact. With 095, the minimap-only prior L1 should drop into the 1–5 range, and the rest of Spec 098's pipeline becomes tractable.

If those are not in, Spec 098 is still doable but the minimap-only L1 will dominate every error budget and the lattice-constrained model will not converge to a useful local minimum.

---

## Open Questions (For User Review Before Plan)

1. **Lattice constraint strictness.** Should the reconstruction model be **hard-constrained** (the lattice points are passed through unchanged) or **soft-constrained** (the model is penalised for deviating from the prior at the lattice points but is free to do so)? The hard-constraint path is the safer one (the WDL file the viewer reads will be consistent with the prior the model used); the soft-constraint path is more flexible if the prior is wrong. Recommended: hard-constrained.

2. **Fractal / hand-painted detection.** What does "fractal" mean here — terrain fractal brush detail (the alpha brush library) or something else? And "hand-painted" — the alpha layer composite or the manual terrain sculpt? Recommended: start with the alpha brush detector (Spec 074 has the brush library; we have ground truth for what is and isn't a brush). Hand-painted detail is harder and can be a Spec 102+ slice.

3. **Per-tile vs per-region model.** Spec 102 is per-tile (one model forward pass per 256×256 tile). A per-region model (e.g. 512×512 or 1024×1024) might give more context for the reconstruction. The cost: a model with 4× the parameters and 4× the compute. Recommended: per-tile for the first version; per-region is a follow-up if the per-tile is too local.

4. **Spec 098 vs Spec 095 priority.** Which comes first? Spec 095 is small (a single U-Net, ≤ 1M params) and is a strict improvement for the V24 prior. Spec 098 is large (multiple sub-specs). Recommended: Spec 095 first, then Spec 098 in sub-spec chunks.

---

## What This Spec Does NOT Do (Explicit Out of Scope)

- Real-time inference in the viewer. The lattice model is a per-tile forward pass; it is not real-time. The viewer can show a low-res WDL prior live and the high-res reconstruction on demand.
- Re-training Stage A on data outside the V18 store. We have what we have.
- Replacing Stage B with a single end-to-end model. The two-stage design is the spec; the reconstruction model is a third stage that consumes Stage A's prior as a hard constraint.
- MAHO-aware priors. The C# WDL reader does not expose MAHO (Spec 094 amendment A1). The lattice model does not use MAHO either.
- Full-fidelity ADT writing. Spec 098 assumes the Spec 097 Slice 3 minimal ADT is enough for the viewer round-trip. A full-fidelity ADT is Spec 105 or later.
- New C# tooling. The WDL/ADT writers in Spec 097/098 are Python; if a C# shim extension is needed, that is its own bounded slice.

---

## Success Criteria

- [ ] SC-098-001: All 5 sub-specs (099-103) have a concrete spec/plan/tasks written and accepted by the user before any implementation starts.
- [ ] SC-098-002: Spec 099 retrains Stage A on every V18 build with a documented per-build per-tile split, and reports the per-build L1 numbers honestly. No "average across builds" fudging.
- [ ] SC-098-003: Spec 100's border-consistency loss is implemented and validated on a known seam tile pair. The seam L1 between adjacent tiles' Stage B outputs is < 0.5 world units after the loss.
- [ ] SC-098-004: Spec 101's fractal detector achieves ≥ 90% per-pixel accuracy on a held-out alpha brush test set.
- [ ] SC-098-005: Spec 102's reconstruction model produces 257×257 heightmaps whose L1 vs the V18 height_257 ground truth is **< 5 world units** on the held-out V18 prior validation. The model's output at the lattice points (16r, 16c) and (16r+8, 16c+8) matches the WDL prior within ±1 world unit (hard constraint).
- [ ] SC-098-006: Spec 103's pipeline runs end-to-end on a 64×64 map in < 30 minutes on a 12 GB GPU, produces the per-map OBJ + atlas + WDL + ADT, and the round-trip smoke is green.
- [ ] SC-098-007: Memory bank + progress.md + architecture doc updated at every sub-spec completion. No "I forgot to update the docs" sessions.

---

## Architecture Sketch

```
V18 Zarr (per map)
  │
  │  Spec 099 retrained Stage A
  ▼
Per-tile WDL prior (17,17) outer + (16,16) inner
  │
  ├───► Spec 101: fractal / hand-painted detail detector
  │       (separate U-Net, output is a probability map)
  │
  ├───► Spec 100: Stage B with border-consistency loss
  │       (existing Stage B + new loss term)
  │
  └───► Spec 102: V24 reconstruction lattice model
          Input: minimap + alpha + normal + mcnr + object_mask
                 + WDL prior (hard constraint at lattice points)
                 + fractal detail map
          Output: 257×257 heightmap

Spec 103 stitches all the above with edge alignment + writes WDL + ADT + round-trips.
```

The user sees the same V18 Zarr they were looking at in Spec 097, but the output is now a **detailed reconstruction**, not a coarse prior. The OBJ opens in the viewer, the WDL file the viewer reads is consistent with the model's prior, and the ADT files carry the full detail.

---

## Open Questions (Continued)

5. **Where the lattice constraint lives.** Three options:
   - (a) Pass-through: the lattice points are concatenated from the prior; the model only learns the in-between cells.
   - (b) Soft penalty: a loss term penalises deviation at the lattice points; the model can deviate.
   - (c) Hybrid: pass-through for the first ~30 epochs, then a soft penalty for fine-tuning. Recommended: (c).
6. **Spec 102's input channel count.** The minimum channels are: cleaned minimap (3) + alpha (4) + normal (3) + mcnr (1) + object_mask (1) + prior (2 = outer + inner, plus confidence and source for 4 more) + fractal (1) = **15 channels**. With the lattice constraint, this is tractable on a 12 GB GPU. Recommended: 15-channel input.

---

## End of Spec

This is the user's vision. It is broken into 5 sub-specs, each bounded, each with measurable success criteria, each with a clear pre-condition dependency. Spec 095 (learned minimap cleaner) should land first. Spec 097 Slices 2/3 (WDL/ADT writers) should land before Spec 103 (round-trip smoke). The order is:

1. Spec 095 (minimap cleaner)
2. Spec 097 Slices 2/3 (WDL/ADT writers, multi-session)
3. Spec 099 (Stage A retrain on every V18 build)
4. Spec 100 (Stage B border-consistency loss)
5. Spec 101 (fractal detail detector)
6. Spec 102 (V24 reconstruction lattice model — the main work)
7. Spec 103 (full-map round-trip integration)

Realistically this is **3-6 months of focused work** for one engineer. Each sub-spec is a real chunk. The user can pick the order based on what they need first.