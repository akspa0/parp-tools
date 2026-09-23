# Phase 0 Research — Modern-to-Legacy Map Conversion

Date: 2026-09-18

## R1 — Where the modern layer stack comes from

**Decision**: Read the modern map through the existing modern readers (Specs 238/239/240) and take
the per-chunk layer stack (texture references + `AMAP` blend weights) as the source of truth, exactly
as the viewer renders it.

**Rationale**: The operator's premise is "we can literally render it perfectly in the viewer, so we
know how to read just enough of the data to preserve it." The viewer already decodes the modern
terrain; the converter must consume the same decoded model, not re-parse the raw chunks.

**Alternatives considered**: Re-parsing modern ADT chunks inside the converter (rejected — duplicates
the reader, violates Constitution II "one canonical owner per format surface").

## R2 — Merge policy for N modern layers → target capacity

**Decision**: A deterministic **coverage-ranked merge**:
1. Rank the chunk's layers by total `AMAP` weight summed across the chunk (base layer always kept).
2. Keep the base layer plus the top `capacity - 1` overlays, where `capacity` is the target's layer
   limit (LK v18 and Alpha 0.5.3 both cap at 4).
3. Fold each dropped layer's weighted contribution into the nearest kept layer's alpha mask (nearest
   by texture family / index distance), so appearance degrades gracefully rather than losing a band.
4. Tie-break by original layer index (ascending) for determinism.

**Rationale**: Preserves the dominant appearance while staying within the target's layer model, and
is fully deterministic (SC-004). The per-tile report (FR-003) records what was merged/dropped.

**Alternatives considered**: Nearest-texture-id collapse (rejected — loses appearance); hard drop of
excess layers (rejected — visible banding).

## R3 — Alpha-mask re-expression

**Decision**: Downsample the modern per-vertex `AMAP` weights to the target's alpha resolution
(Alpha 0.5.3 MCAL 64×64; LK v18 MCAL 64×64) by area-average, then combine dropped layers' weights
into the kept layer's alpha as in R2.

**Rationale**: The target alpha grids are coarser than modern per-vertex weights; area-average is the
faithful downsample and matches how the existing `LkToAlphaConverter`/`AlphaToLkConverter` already
move alpha between eras.

**Alternatives considered**: Nearest-vertex sampling (rejected — aliases shorelines).

## R4 — Texture reference resolution

**Decision**: Resolve each modern texture FileDataID to a path through the Spec 238/239 resolution
chain; write the resolved path into the target's MTEX list and reference it by index in MCLY. An
unresolved reference is reported (FR-003) and the tile still converts (US1 scenario 2).

**Rationale**: Legacy MCLY stores an index into MTEX (a path list), so a path is required; the modern
id-addressed world must be translated, and misses are expected in CDN-less installs.

**Alternatives considered**: Fail the tile on a missing texture (rejected — violates US1 scenario 2).

## R5 — Writer entry points

**Decision**: Reuse the existing writers unchanged:
- LK v18: `LkAdtWriter.Write(path, LkAdtData, MapConversionTargetFormat.LkAdtV18)` + `LkWdtWriter`.
- Alpha 0.5.3: `AlphaWdtWriter` (monolithic WDT).

**Rationale**: Constitution II (one canonical owner per format surface); the writers already produce
loadable output and are exercised by the existing Alpha↔LK routes.

**Alternatives considered**: A new modern-specific writer (rejected — duplicates the format surface).

## R6 — Entry point: CLI vs Editor

**Decision**: **One owned service** (`ModernToLegacyMapConversionService`) surfaced in **both** the
CLI (`convert-map` batch driver) and the Editor Map Converter dialog. The CLI is the batch driver;
the dialog is the low-touch single/batch surface.

**Rationale**: Resolves the spec's open question; satisfies FR-005 (low-touch UI) and FR-004 (batch)
without duplicating logic, and keeps the god-class freeze (AGENTS.md §10) by putting state in the
service, not `ViewerApp`.

**Alternatives considered**: CLI-only (rejected — the operator asked for a low-touch UI); Editor-only
(rejected — batch belongs in a CLI).

## R7 — Determinism

**Decision**: Sort all iteration (maps, tiles, layers, assets) by a stable key; never depend on
dictionary enumeration order; write provenance with a content hash. Reruns must be byte-identical
(SC-004).

**Rationale**: SC-004 is an explicit success criterion; the existing writers are deterministic given
ordered input.

## R8 — Route validation before writing

**Decision**: Extend `MapConversionFormats` with a `ModernFileDataId` source format and validate the
(source, target) pair before any write; unsupported/lossy combinations are surfaced (FR-008) rather
than half-written.

**Rationale**: The existing `MapConversionFormats.Validate` already gates the Alpha↔LK routes; the
modern route joins the same gate.

## Open items carried into Phase 1

- Exact modern layer-capacity constant per target (confirm 4 for both LK v18 and Alpha 0.5.3).
- Whether asset inclusion copies minimaps too (spec says "textures, models, minimaps").
