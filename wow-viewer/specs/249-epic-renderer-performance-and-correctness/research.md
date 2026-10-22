# Renderer fidelity research — 2026-10-22

Read-only bounded inspection. Facts below are source behavior, not runtime confirmation. Covers
the operator's plan refresh only; no archive audit or claim that all epic backlog is current.

## Ground effects (owner Spec 259)
- **Observed:** [scene service](../../src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs:467)
  uses 3D distance rejection and last-25-unit opacity fade. [Terrain density](../../src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs:112)
  is placement-time, not distance thinning. [Draw loop](../../src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs:539)
  submits individual instances; the dictionary is not itself GPU instancing.
- **Observed:** [compatibility material](../../src/viewer/WoWViewer/Rendering/ModelRenderer.cs:1446)
  already switches 0.50 cutout to blended fading at 0.05. [Caller](../../src/viewer/WoWViewer/ViewerApp.cs:1118)
  passes fixed lighting/fog defaults, not the evaluated world lighting.
- **Decision:** identify native/compatibility route, fade and counts on real examples first;
  preserve Spec 276's delivered fade work and Spec 259's reverted tint boundary.
- **Hypotheses:** overlap, absent distance thinning, material-route differences and lighting
  mismatch can contribute to distant density. None is established as the complaint's cause.
- **Rejected:** unconditional opacity reduction, radius reduction or instancing overhaul as a fix.

## WMO / MH2O liquid resolution
- **Observed:** [WMO service](../../src/viewer/WoWViewer/Rendering/WmoLiquidRenderer.cs:292)
  classifies first visible nibble (water/magma/slime), applies ocean group hint, logs group/material
  IDs; appearance is a fixed palette and simple procedural shader, not DBC texture selection.
- **Observed:** [live MH2O mapping](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:2840)
  uses exact-build class column, name heuristics if absent, water for missing rows. Its
  [merge](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:2655) reduces instances to one
  chunk surface/type with magma/slime/water/ocean precedence. Terrain appearance is also
  [procedural](../../src/viewer/WoWViewer/Terrain/LiquidRenderer.cs:403).
- **Observed:** [vertex-format chain](../../src/core/WowViewer.Core.IO/Dbc/LiquidVertexFormatChain.cs:35)
  resolves direct values or LiquidObject → LiquidType → LiquidMaterial layout; this is not the
  visual material. [Harvest reader](../../src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs:305)
  has a separate family fallback path; do not confuse it with the live adapter.
- **Decision:** trace identity → exact row → family/layout → submitted material; locate mismatch
  before deciding a policy/service change. Leave readers frozen absent a verified, approved bug.
- **Existing tests:** [DBC family](../../tests/WowViewer.Core.Tests/Dbc/DbcLiquidTypeTableTests.cs:38),
  [ADT resolution](../../tests/WowViewer.Core.Tests/AdtLiquidReaderTests.cs:96),
  [two-decoder layout agreement](../../tests/WowViewer.Core.Tests/Dbc/LiquidVertexFormatChainTests.cs:120),
  [WMO summary](../../tests/WowViewer.Core.Tests/WmoGroupLiquidSummaryReaderTests.cs:11),
  [WMO layout](../../tests/WowViewer.Core.Tests/WmoLiquidLayoutResolverTests.cs:9). None proves
  rendered material parity.
- **Rejected:** assuming material table lookup already gives native visual fidelity, or correcting
  only the converter decoder while the live adapter keeps different behavior.

## Sky / lighting
- **Observed:** [dome](../../src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs:49),
  [lighting chain](../../src/viewer/WoWViewer/Terrain/LightService.cs:63) and
  [atmosphere backdrop](../../src/viewer/WoWViewer/Terrain/Scene/Atmosphere/SceneAtmosphere.cs:357)
  already exist; the active client sky model is night-visibility-gated, placed backdrops separate.
- **Decision:** audit exact-build row/model/visibility/pass and day/night behavior against verified
  references; preserve blank sky and expose proven choices through existing controls.
- **Rejected:** treating sky support as absent and replacing the working base from scratch.

## Observed reference inventory and limits

| Reference | Observed availability | Use / limit |
|---|---|---|
| [LiquidType schema](../../libs/wowdev/WoWDBDefs/definitions/LiquidType.dbd:105), [material schema](../../libs/wowdev/WoWDBDefs/definitions/LiquidMaterial.dbd:9), [LightSkybox](../../libs/wowdev/WoWDBDefs/definitions/LightSkybox.dbd:16), [Light](../../libs/wowdev/WoWDBDefs/definitions/Light.dbd:22) | Repo-local, covers 3.3.5 and 4.3.4 | Layouts, not actual client rows or decomp algorithms |
| [LiquidObject schema](../../libs/wowdev/WoWDBDefs/definitions/LiquidObject.dbd:11) | Repo-local 4.3.4 definitions | Separate from direct 3.3.5 vertex formats |
| [3.3.5 decomp pointer](../273-client-ground-truth-renderer-performance/plan.md:5) | Repo documentation points to an external checkout | Exact root is in cited document; checkout/revision not inspected or assumed installed |
| [4.3.4 DetailDoodad pointer](../276-trinitycore-population-pipeline-and-ground-effect-fidelity/tasks.md:11) | Repo documentation cites external client source | Verify source/build/symbol before parity claims; not a new task to reapply shipped threshold |

No repo-local decomp checkout was observed in bounded directory inspection; this is not an
exhaustive filesystem absence claim. No external executable or analysis-tool installation was
verified. Gate E requires configured reference paths/revisions, not a guessed tool dependency.

Design unknowns resolve to evidence gates and conditional narrow corrections; empirical causes
and client parity remain unproven until operator/reference receipts. No new dependency is chosen.
