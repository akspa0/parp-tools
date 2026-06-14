# Spec Analysis: 055-unreal-engine-bridge

**Date**: 2026-06-09
**Source Spec**: `wow-viewer/specs/055-unreal-engine-bridge/spec.md`

## Completeness

### Strengths
- Clear architectural thesis (Native AOT bridge, C# stays canonical, UE owns conversion).
- 6 user stories prioritized P1-P3, each independently testable.
- Key entities well-defined (ClientRoot, MapManifest, TerrainTileData, WmoPayload, M2Payload, BlpTexture, LiquidPayload, DbcTableSchema).
- Non-goals section is honest about what's out of scope.
- Impact on existing plans is acknowledged.

### Gaps
- **UE version not pinned**: spec says "5.x (exact version TBD by installed engine)". Plan must pin a specific version early to lock down API surface.
- **Material translation detail missing**: WoW M2/WMO materials have flags (blend, unlit, unlit specular, two-sided, emissive, etc.) that need explicit translation. Spec mentions "correct material assignments" but no material family table.
- **Animation curve fidelity**: spec says "baked poses" but M2 has Bezier-interpolated tracks. Baking at 30fps loses sharp keyframes. Plan should address this.
- **Particle/ribbon emitters not mentioned**: M2 has particle systems (MDID, MODL, etc.) and ribbon emitters. Excluded or planned?
- **WMO portals not in scope**: WMO portals affect visibility and culling. UE has its own culling; this might not need explicit handling, but should be acknowledged.
- **Memory ownership contract undefined**: who frees buffers returned by the C API? Must be specified.
- **Build pipeline unclear**: where does the AOT DLL get built? How is it shipped with the UE plugin? Versioned together or separately?
- **LOD policy not specified**: WoW has 1-4 LOD levels per M2. UE's LOD system needs to consume this.
- **Sky rendering under-detailed**: spec mentions WMO skyboxes but no UE conversion strategy.
- **Test map/scenario contract missing**: which staged client maps serve as the validation surface? Need a defined test matrix.

## Dependencies

### Exists already
- Format readers in `WowViewer.Core.IO/` (WMO, M2/MDX, BLP, ADT, DBC/DB2, terrain, liquid).
- Runtime contracts in `WowViewer.Core.Runtime/` (world composition, M2 animation, terrain).
- Native MPQ service for archive access.
- Existing diagnostic tools and validation capture.

### Does not exist (must be built)
- C API export surface on top of C# libraries.
- .NET 10 Native AOT build configuration for the bridge.
- UE 5.x plugin source tree.
- C++ UE module with bridge loader, type converters, actor spawners.
- UE project for development testing.
- Test maps defined as the bridge validation surface.

### External dependencies
- .NET 10 SDK with Native AOT support (Windows x64).
- Unreal Engine 5.x installed (exact version TBD; candidate 5.4 for stability).
- Visual Studio with UE C++ toolchain.

## Risks

### Architectural risks
- **Risk: Native AOT compatibility surface unknown.** Some C# features (reflection, dynamic loading) don't work in AOT. The C# libraries may need code changes to be AOT-compatible. **Mitigation**: Phase 1 must include an AOT compatibility audit of existing libraries.
- **Risk: BLP texture decode happens on the C# side.** Decoding 4096x4096 BLP to RGBA8 in C# is slow. UE has its own BLP-compatibility story. **Mitigation**: Start with BLP-RGBA8 decode on C# side; revisit native-side decode if profiling shows it as bottleneck.
- **Risk: Animation baking loses fidelity.** Baked transforms can't represent sharp keyframes well. **Mitigation**: Phase must include a fidelity comparison test between baked C# output and raw interpolated values.

### Process risks
- **Risk: UE C++ is a new surface for the project.** No existing C++ code or build expertise. **Mitigation**: Phase 0 must be a focused learning spike on UE plugin development.
- **Risk: Scope creep into UE feature adoption.** Tempting to add physics, post-processing, etc. **Mitigation**: Spec's non-goals section is binding.
- **Risk: AOT build pipeline blocks progress.** Native AOT for .NET 10 is relatively new. Build errors might require significant refactoring. **Mitigation**: Phase 1 includes AOT build validation as a first-class deliverable.

### Compatibility risks
- **Risk: Existing C# libraries use features that are not AOT-friendly** (e.g., `System.Reflection.Emit`, dynamic types, runtime code generation). **Mitigation**: Audit before committing to architecture; alternative is C++/CLI bridge.
- **Risk: C# standard library interop with AOT is limited.** Some dependencies may not work. **Mitigation**: Audit all `WowViewer.Core` package references for AOT compatibility.

## Constitution Compliance

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | COMPLIANT | Spec explicitly keeps all C# code in `wow-viewer/`. UE plugin must not reference `gillijimproject_refactor/`. |
| II. Library-First | COMPLIANT | Spec adds bridge as a thin export layer. No format parsing moves to UE/C++. |
| III. Real-Data Validation | COMPLIANT | Every story has validation against staged client roots. |
| IV. Residual Model Chain | N/A | No model training changes. |
| V. Streaming-First Dataset Pipeline | N/A | No dataset changes. |
| VI. No Game Client Path Assumptions | COMPLIANT | Uses `output/tmp/wowarchive-clients/` paths. |
| Read-Only Reference Codebase | COMPLIANT | Spec is silent on `gillijimproject_refactor/` — must not touch it. |
| Format Reader/Writer Ownership | COMPLIANT | Spec is explicit: bridge is an export surface, no parser changes. |

## Gap Analysis vs Existing Plans

### Relation to `wow-engine-modernization-plan-2026-05-14.md`
- That plan calls Vulkan as primary backend, OpenGL as fallback. This spec replaces that with UE. **Resolution**: the spec acknowledges this in "Impact on Existing Plans" — that plan must be amended to recognize UE as primary.
- That plan has 10 phases (E0-E9). This spec's bridge work overlaps with E1 (Runtime Contracts), E2 (Vulkan Backend), and E3 (OpenGL Fallback). **Resolution**: the bridge spec supersedes E2 and most of E3; the OpenGL path in `WowViewer.App` becomes a headless-only diagnostic surface.

### Conflicts with `game-viewer-host-plan-2026-05-13.md`
- That plan targets `WowViewer.App` as the "game-viewer" host. This spec moves that to UE. **Resolution**: `WowViewer.App` becomes a CLI/diagnostic host only; the "game-viewer" product identity is now UE-based.

### Conflicts with `wow-viewer-library-completeness-plan-2026-05-06.md`
- That plan tracks renderers as missing (MdxRenderer, TerrainRenderer, etc.). This spec deprecates that need. **Resolution**: that plan's renderer gap column becomes lower-priority.

### Gaps vs `game-viewer-plan-pack-2026-05-14/`
- That plan pack has 49 micro-plans (GV-00 through GV-26) targeting a `WowViewer.App` host. This spec's UE bridge covers GV-17 (Backend Bridge), GV-14 (Render Layer Contracts), GV-09 (Archive Adapter). The other GV- plans remain valid for the C# side and the diagnostic app host.

## Recommendation

**APPROVE WITH REVISIONS.**

The spec is solid at the architectural level. The plan and tasks must:

1. Pin a specific UE version (recommend **UE 5.4** for stability).
2. Add a Phase 0: AOT compatibility audit + UE plugin spike to de-risk before committing to the full plan.
3. Add explicit memory ownership rules in the C API contract.
4. Define the test map matrix (which staged clients/maps serve as validation surface).
5. Add a material translation table (WoW material flags → UE material properties).
6. Define the build pipeline for shipping the AOT DLL with the UE plugin.
7. Decide particle/ribbon emitter strategy (expose in this phase, or defer).
8. Add an animation fidelity test (baked vs. raw interpolation comparison).

Proceed to `speckit-plan` with these revisions folded into the plan structure.
