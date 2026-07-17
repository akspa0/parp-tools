# Research: Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction

## Decision: shading-match inference lives in C#, reusing the production compositor directly

- **Decision**: Implement the shading-match scorer as a new C# component next to
  `MinimapLightingProvenance` in `WowViewer.Core`/`WowViewer.Core.IO`, rendering candidates through
  the exact same `TerrainMinimapCompositor` + `TerrainSolarDirection` path the live viewer and the
  Harvest `synthetic-minimap` exporter already use. Do not port the lighting model into a second
  Python implementation to run the candidate sweep.
- **Rationale**: Auditing the existing synthetic-lighting-variant generator
  (`data-harvester/src/harvester/spec103/terrain_lighting.py`) found it already carries its own
  independent reimplementation of the solar-direction model
  (`PROFILE_REVISION = "wow-1.0.0-authored-day-night-v1"`, a hardcoded
  `AUTHORED_MCSH_BAKE_DIRECTION` constant, and a separate `GRID_TO_RENDERER_NORMAL_TRANSFORM`) that
  has already drifted from the corrected C# `AuthoredTerrainDayNightProfile`
  (`ProfileRevision = "...-v3"`) fixed today. A second candidate-sweep implementation in Python would
  either repeat that drift immediately or require every future lighting fix to be ported twice. Using
  the C# production path as the single source of truth eliminates the class of bug this feature exists
  to correct for.
- **Alternatives considered**: Reimplement the sweep in Python against the harvester's existing
  in-process terrain arrays (rejected: duplicates the exact reimplementation risk just found in
  `terrain_lighting.py`, and violates the constitution's Library-First / Format Reader-Writer
  Ownership principles — "check if one already exists before writing a new one"); call into C# via a
  new RPC/IPC boundary (rejected as unnecessary complexity: the existing C# harvester -> Python Zarr
  streaming protocol, constitution principle V, is already the sanctioned boundary and is reused as-is
  in Phase 1 below).

## Decision: retire the drifted Python `terrain_lighting.py` sweep in favor of streamed C# output

- **Decision**: Phase 2 replaces `terrain_lighting.py`'s independent lighting-direction reimplementation
  with consumption of lighting parameters computed in C# (via the same streaming protocol used for
  Phase 1's bucketing pass) rather than updating the Python constants to match v3 by hand. Where the
  Python module's non-direction responsibilities (color/fog interpolation, MCSH bake authoring) are
  still needed and have no C# equivalent yet, they are left in place but clearly re-labeled so the
  *direction* component specifically is never computed independently again.
- **Rationale**: Hand-porting v1 to v3 constants would fix today's drift but leaves the same
  duplication in place to drift again at v4. Given this project's explicit "time-to-signal over rigor"
  philosophy and the fact the C# side is the one with ground-truth validation (§2.1 of the world-lighting
  doc, plus the real-client minimap comparison from today's fix), collapsing to one source of truth is
  both less code and permanently closes the drift risk, not just this instance of it.
- **Alternatives considered**: Update the Python constants to match v3 now (rejected: cheaper today,
  but recreates exactly the problem being fixed, and the next lighting correction would need to
  remember to touch two files in two languages); leave `terrain_lighting.py` untouched and scope this
  feature to bucketing only (rejected: Spec 103's synthetic variants would keep training the model
  against a demonstrably wrong lighting model even after this feature "fixes" the real-data bucketing).

## Decision: shading-match score isolates directional structure, independent of the existing tint signal

- **Decision**: Score each time-of-day candidate against the authored minimap using a metric over
  luma *gradient direction* (e.g. normalized cross-correlation of Sobel-style gradient orientation
  fields, or an equivalent directional-structure comparison), computed after removing each image's own
  mean/tint so the score cannot simply reward color matching. Reuse `MinimapLightingProvenance`'s
  existing MCSH-shadow mask correlation to exclude/down-weight regions already flagged as
  likely-baked-static-shadow before scoring.
- **Rationale**: The existing tint-ratio inference (`MinimapLightingProvenance.Infer`) already answers
  "what color was this lit with"; it cannot answer "which direction do the shadows fall," which is
  exactly what today's TerrainSolarDirection bug was about and exactly the new information this feature
  needs. A gradient/structure-based metric is the standard way to compare hillshade-style relief images
  without being fooled by material-color differences between the authored minimap and the synthesized
  terrain-only baseline.
- **Alternatives considered**: Reuse the tint-ratio metric directly (rejected: proven insufficient --
  it was the mechanism already in place and did not catch either of today's two direction bugs, which
  were only caught by a human visual side-by-side); full per-pixel MSE between authored and synthesized
  RGB (rejected: dominated by material/texture differences unrelated to lighting direction, since the
  synthesized baseline uses BLP material averages rather than reproducing exact authored art).

## Decision: bucket results and distribution reports are additive Zarr/Parquet fields, never NPZ

- **Decision**: Store shading-match results as new fields on the existing per-build Zarr store
  (alongside the existing tint-based `minimap_lighting` metadata) and derive the distribution report
  from a Parquet index pass over those fields, matching the existing index-file convention.
- **Rationale**: Constitution principle V is explicit: "No intermediate NPZ files on disk. The Zarr
  store is the only on-disk artifact." The existing tint-based `MinimapLightingProvenance` already
  follows this (Spec 110/109 research: "Full and V22 raw streams carry this sidecar"); the new
  shading-match fields extend the same store and streaming contract rather than introducing a second
  artifact format.
- **Alternatives considered**: A standalone NPZ/JSON report file per build (rejected: forbidden by
  constitution V; also creates a second source of truth that can drift from the Zarr store itself).

## Decision: held-out evaluation set for Phase 3 reuses the existing Spec 108 group-holdout contract

- **Decision**: The retrain-and-evaluate comparison (User Story 3) uses the same source-group-held-out
  split discipline already established for the Spec 108 WDL-prior mixed store (real + synthetic groups
  held out together, never split across train/eval), rather than defining a new split policy.
- **Rationale**: Spec 108's plan already solved the group-leak problem for this exact
  real+synthetic-lighting-variant training corpus. Reusing it keeps this feature's evaluation
  comparable to the existing checkpoint's own validation history instead of introducing a second,
  differently-biased evaluation methodology that would make "improved vs. regressed" ambiguous.
- **Alternatives considered**: Define a fresh held-out set specific to lighting buckets (rejected:
  would conflate "did rebalancing help" with "is this a harder/easier split than before," making the
  go/no-go comparison unreliable).

## Decision: Phase 3 execution is a distinct, separately gated step, never a byproduct of Phase 1/2 completion

- **Decision**: Phases 1 and 2 (bucketing, rebalancing) are implementation work with their own
  focused-test validation and can be completed and committed independently. Phase 3 (actual GPU
  training run and checkpoint comparison) is written as a plan step that explicitly stops and requires
  a separate, in-session user go-ahead immediately before the training command is executed -- matching
  established project practice of never launching a resource-intensive local GPU run or any cloud pod
  without that explicit confirmation.
- **Rationale**: This is existing, repeatedly-confirmed user guidance (never auto-launch billed cloud
  compute; confirm before long local GPU runs), not new policy invented for this feature. Spec 108's
  own plan already marks its training/inference steps "User-run" for the same reason.
- **Alternatives considered**: Bundle Phase 3 into the same automatic pass as Phases 1/2 (rejected:
  directly contradicts established guidance and this feature's own FR-013/SC-006).
