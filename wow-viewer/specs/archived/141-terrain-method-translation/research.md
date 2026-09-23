# Research: Terrain Method Translation and Evidence Gates

## Decision 1: Separate DSM/DTM filtering from RGB-only completion

**Decision**: Treat DSM-to-DTM and LiDAR ground filtering as method references and offline diagnostic branches. Treat the current WoW minimap problem as RGB-only terrain completion unless a future runtime contract explicitly supplies a height prior.

**Rationale**: The [DSM2DTM paper](https://elib.dlr.de/198246/1/ISPRS2023_Bittner_final.pdf) describes a DSM as containing terrain plus above-ground objects and frames the task as detecting/removing non-ground objects before recovering bare-earth height. The [DSM2DTM repository](https://github.com/KseniaBittner/DSM2DTM) supplies the closest neural architecture and training-data precedent, but it expects DSM/elevation input. An RGB minimap does not observe the hidden terrain surface in the same way.

**Alternatives considered**:

- Feed minimap RGB into DSM2DTM unchanged: rejected because the input semantics and spatial signal are wrong.
- Pretend MCSH or target height is an equivalent DSM: rejected because those are not minimap-observable inference signals.

## Decision 2: Keep classical ground filters as diagnostic baselines

**Decision**: Record [PDAL SMRF](https://pdal.org/en/stable/stages/filters.smrf.html) and [CSF](https://github.com/jianboqi/CSF) as classical baselines for any future project-owned LiDAR/DSM sample, not as RGB inference components.

**Rationale**: SMRF classifies ground/non-ground returns from point-cloud data, while CSF uses cloth simulation over airborne LiDAR. Both answer the surface-observed problem and provide useful failure vocabulary, but neither can recover terrain from albedo-only pixels.

**Alternatives considered**:

- Implement a point-cloud pipeline before a point-cloud source exists: deferred; it would create plumbing without a valid WoW input contract.
- Use a classical filter on `height_257` targets as if it were deployment evidence: rejected; targets are evaluation-only.

## Decision 3: Use ResDepth as an architecture reference for a future height-prior branch

**Decision**: Study [ResDepth](https://github.com/prs-eth/ResDepth) for residual correction, image/DSM fusion, and building-mask-aware metrics, but keep it outside the first RGB-only benchmark.

**Rationale**: ResDepth assumes an imperfect DSM plus registered imagery and refines the elevation residual. That is a strong pattern if a WoW pipeline later exposes an observable height prior. It is not a valid answer to the current RGB-only contract.

**Alternatives considered**:

- Add a synthetic height prior immediately: rejected because it would conflate architecture research with deployment evidence and repeat the old target-derived-input failure.

## Decision 4: Use aerial segmentation models only as predicted-mask auxiliaries

**Decision**: Treat the [Restor tree-cover SegFormer](https://huggingface.co/restor/tcd-segformer-mit-b3) and [Kastela building segmentation model](https://huggingface.co/dnovak232/kastela-dof5-building-segmentation) as candidate mask priors, not terrain predictors.

**Rationale**: These models demonstrate useful RGB-to-object-mask tasks, but their training domains, resolutions, classes, and labels do not match WoW minimaps. A predicted mask can guide an ablation; it cannot be treated as ground-truth object identity or terrain height.

**Alternatives considered**:

- Download external weights for the first benchmark: rejected; the first benchmark must remain project-owned and license/domain evidence must be explicit.
- Use a source-side object mask at inference: rejected; it is target-side supervision and unavailable in a real minimap.

## Decision 5: Defer a general geospatial foundation encoder

**Decision**: Keep [Prithvi EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) and [TerraTorch](https://github.com/torchgeo/terratorch) as future research references, not first-run dependencies.

**Rationale**: Prithvi is a broad Earth-observation encoder and TerraTorch is a geospatial model framework. Neither is a drop-in bare-earth predictor for WoW minimap RGB, and adopting them now would add domain and dependency uncertainty before the signal contract is proven.

## Decision 6: Make method translation a staged evidence lane

**Decision**: Implement the work in this order: external method ledger, modality/provenance audit, RGB-only no-mask/predicted-mask comparison, optional DSM/point-cloud diagnostic when a valid source exists, then a translation decision.

**Rationale**: This preserves the project’s research advantage: unusual observations become durable, falsifiable leads; promising methods are tested against the actual available signal instead of being imported on analogy alone.

## Open questions deliberately retained

- Whether a useful object mask can be predicted from WoW minimap RGB after domain adaptation.
- Whether mask-aware terrain completion improves cross-tile continuity rather than only local MAE.
- Whether any future client-backed artifact exposes a legitimate runtime height prior distinct from the target arrays already rejected.
- Whether external aerial segmentation models transfer at all; a failure is useful evidence and must be recorded as such.
