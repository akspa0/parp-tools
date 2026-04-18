# wow-viewer MdxViewer Feature-Parity Cut-Away Plan

## Status

- status: active
- intent: reach functional feature parity with the old `gillijimproject_refactor/src/MdxViewer` viewer stack, but with implementation ownership moved into `wow-viewer`
- program shape: a numbered cut-away plan set, where each plan closes one ownership lane in `wow-viewer` and leaves `MdxViewer` behind as a bounded compatibility consumer only

## Why This Plan Exists

- the user direction is no longer “keep the old viewer alive forever and port isolated tricks over when convenient”
- the user direction is now explicit:
	- `wow-viewer` must become the real implementation owner
	- the remaining `MdxViewer` feature surface must be cut away deliberately
	- parity has to mean viewer behavior, not just parser summaries or narrow proof commands
- the repo already has strong partial plans for:
	- M2 runtime ownership
	- world-runtime extraction
	- viewer-app cutover
	- editor transition
	- shared-I/O ownership
- what was missing was one top-level numbered program that says how those plans fit together into full old-viewer replacement

## Non-Negotiable Goal

The migration is only done when:

1. `wow-viewer` owns the active parsing, runtime, rendering, tool, and app surfaces needed for the old viewer's supported workflows
2. `WowViewer.App` is the default home for viewer behavior and bounded editor-facing continuation work
3. `MdxViewer` is reduced to one of:
	- a temporary compatibility consumer of shared `wow-viewer` seams
	- an archaeology/reference host for legacy-only features not being carried forward
	- a removable shell once parity and cutover are complete

## Cut-Away Rules

1. Do not port old `ViewerApp` panels into `wow-viewer` before the shared runtime or I/O seams they depend on are owned there.
2. Do not leave permanent format truth or render-state truth in `MdxViewer` once a `wow-viewer` seam exists.
3. Do not call a slice complete because a CLI proof or test passes if the real viewer behavior is still missing.
4. Do not deepen `MdxViewer` as the design owner for shell, runtime, renderer, or editor architecture.
5. Every numbered plan must end with an explicit “what ownership moved” statement.

## Current Baseline

As of Apr 18, 2026:

- `wow-viewer` already has:
	- a real desktop shell in `WowViewer.App`
	- bounded M2 and world-session consumers
	- partial world-runtime ownership
	- strong PM4 ownership momentum
	- growing shared-I/O ownership
	- meaningful classic `MDX` standalone runtime/renderer foundations
	- deterministic and now visual regression proof seams for several bounded consumers
- `MdxViewer` still remains the practical owner of:
	- full world viewing behavior
	- the broad terrain/WMO/MDX/M2 compatibility surface
	- many user-facing investigation and editor workflows
	- a large amount of real rendering behavior, even where `wow-viewer` already owns part of the runtime foundation

That means the program is now in the middle, not the beginning: enough foundation exists to stop planning abstractly and start cutting away whole ownership zones.

## Numbered Plans

### Plan 01 - Program Control, Parity Matrix, And Exit Gates

- goal:
	- keep the cut-away effort honest and measurable instead of “feels close”
- owns:
	- the feature inventory
	- parity acceptance gates
	- per-lane signoff criteria
- concrete outputs:
	- maintain the parity matrix in `gillijimproject_refactor/plans/wow_viewer_format_parity_matrix_2026-03-28.md`
	- add viewer-feature parity rows where the current matrix is still too format-centric
	- define for every major user-facing workflow whether the end state is:
		- fully migrated
		- compatibility-only
		- intentionally dropped
- done means:
	- no major old-viewer capability is “floating” without an owner or a disposition
	- every later numbered plan has concrete exit criteria tied to real user workflows
- should not include:
	- new renderer or parser implementation by itself

### Plan 02 - Shared Format Ownership Closure For Active Viewer Data

- goal:
	- finish the shared file/data seams that still force `MdxViewer` to own too much viewer behavior
- primary families:
	- `ADT` root
	- split `ADT` companions
	- `WDT`
	- `WDL`
	- `WMO`
	- `BLP`
	- actively used `DBC`/`DB2`
- key dependency plans:
	- `wow_viewer_shared_io_library_plan_2026-03-26.md`
	- `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- done means:
	- the current viewer and tools no longer rely on `MdxViewer`-local parsers for active terrain/world/model bootstrap families
	- `WowViewer.App`, inspect, and converter consume the same owned format services
- should not include:
	- app-shell-only polish
	- editor save workflows before the read/runtime seams are real

### Plan 03 - M2 Runtime And Renderer Final Closure

- goal:
	- finish the M2 lane so old `MdxViewer` M2 behavior is no longer a design dependency
- current status:
	- partially strong already; parser/runtime foundations are real
- remaining target surface:
	- residual section/material fidelity
	- full lighting/effect behavior in real consumers
	- particle/ribbon completion
	- scene submission/batching closure
	- parity harnesses strong enough to retire old-viewer ownership
- key dependency plan:
	- `wow_viewer_m2_runtime_plan_2026-03-31.md`
- done means:
	- `MdxViewer` M2 rendering is only a compatibility consumer or removable reference path
	- `WowViewer.App` and supporting tools can prove real M2 parity on fixed assets/builds
- should not include:
	- broad world-scene replacement in the same slice

### Plan 04 - MDX Runtime And Renderer Closure

- goal:
	- finish the classic `MDX` lane from bounded standalone preview into full old-viewer-equivalent ownership where that behavior is still required
- current status:
	- strong bounded standalone groundwork exists, but world/runtime ownership is incomplete
- remaining target surface:
	- helper/attachment/event ownership
	- particles/ribbons
	- material/effect parity gaps
	- world-scene/runtime consumers
	- stronger visual and behavior regression coverage
- upstream dependencies:
	- Plan 02 shared format ownership
	- the new visual-regression harness already landed in `WowViewer.App`
- done means:
	- classic `MDX` behavior no longer depends on `MdxViewer` parser/runtime/render ownership
	- the remaining old-viewer `MDX` path is compatibility-only or removed
- should not include:
	- re-implementing old `ModelRenderer` architecture verbatim inside `wow-viewer`

### Plan 05 - WMO Runtime And Rendering Closure

- goal:
	- move WMO root/group/material/liquid/render behavior into `wow-viewer` deeply enough that old `WmoRenderer` stops being the owner
- target surface:
	- root/group payload ownership
	- material and texture semantics
	- liquid behavior
	- visibility/topology/runtime-facing service contracts
	- standalone and world consumers
- dependencies:
	- Plan 02 shared format ownership
	- Plan 06 world runtime for final world-consumer closure
- done means:
	- `WmoRenderer` is no longer where new WMO behavior is designed first
- should not include:
	- PM4/editor work unless required for a narrow compatibility proof

### Plan 06 - World Runtime And 3D World Consumer Cutover

- goal:
	- replace `WorldScene` as the long-term owner of world-view runtime behavior
- current status:
	- partial runtime extraction already exists
	- `WowViewer.App` world consumer is still bounded and top-down
- remaining target surface:
	- true 3D world camera and renderer consumption
	- terrain/WDL/liquid/WMO/MDX pass ownership
	- host thinning until `WorldScene` becomes compatibility-only
	- Alpha-era world bring-up where practical, not just Wrath-only proof
- key dependency plans:
	- `wow_viewer_world_runtime_service_plan_2026-03-31.md`
	- `wow_viewer_viewer_app_cutover_plan_2026-04-17.md`
- done means:
	- `WowViewer.App` is the canonical world viewer
	- `WorldScene` is no longer the architecture center of gravity
- should not include:
	- editor save/publish work before basic viewer runtime closure

### Plan 07 - Viewer Shell, UX, And Workflow Surface Parity

- goal:
	- rebuild the actually-used viewer workflows in `WowViewer.App` once their runtime/data seams are real
- target surface:
	- navigation
	- selection/inspection
	- status and diagnostics
	- camera workflows
	- startup/open flows
	- capture/proof automation
	- non-placeholder standalone workspaces for `M2`, `MDX`, `WMO`, and world
- key dependency plan:
	- `wow_viewer_viewer_app_cutover_plan_2026-04-17.md`
- done means:
	- users are not routinely forced back into old `ViewerApp` for normal viewing and investigation work
- should not include:
	- deep parser/runtime ownership that belongs in Plans 02-06

### Plan 08 - Tool, Inspect, Converter, And Dataset Cutover

- goal:
	- eliminate the split-brain where shared services exist in `wow-viewer` but practical tool ownership still lives in old tool roots
- target surface:
	- inspect verbs
	- converter/report jobs
	- dataset builder ownership
	- minimap/capture/export surfaces where they should remain first-class
- key dependency plans:
	- `wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
	- `wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md`
	- `wow_viewer_minimap_generation_plan_2026-04-08.md`
- done means:
	- old tool-specific parsing and export logic is either cut over or explicitly legacy-only
- should not include:
	- editor transaction/save semantics unless they are strictly downstream consumers

### Plan 09 - Editor Foundation And Save-Capable Cutover

- goal:
	- migrate the modern editor direction into `wow-viewer` after viewer/runtime ownership is real enough to support it
- target surface:
	- dirty-map state
	- object add/move/remove ownership
	- terrain edit/write ownership
	- publish/save packaging
	- editor workspace organization
- key dependency plan:
	- `wow_viewer_editor_plan_2026-04-03.md`
- done means:
	- editor architecture is no longer being invented in old `MdxViewer` panel code
	- save-capable behavior uses shared `wow-viewer` writers/services first
- should not include:
	- rebuilding every archaeology/debug panel just because it exists in the old app

### Plan 10 - Compatibility Retirement And Final De-Ownership

- goal:
	- explicitly retire `MdxViewer` as the default owner once the major lanes are closed
- target surface:
	- mark remaining old-viewer features as:
		- migrated
		- compatibility-only
		- legacy-only
		- intentionally dropped
	- remove or freeze duplicate ownership paths
	- update repo guidance so future work cannot drift back
- dependencies:
	- Plans 02 through 09 substantially complete
- done means:
	- `MdxViewer` is no longer where new viewer features are expected to land
	- remaining use of the old app is deliberate and narrow

## Recommended Execution Order

The numbered plans are not all serial, but the default execution order should be:

1. Plan 01 immediately and continuously
2. Plan 02 as the base data-ownership lane
3. Plans 03, 04, and 05 in parallel where their dependencies are satisfied
4. Plan 06 once enough of Plans 02 through 05 are real to support a serious world consumer
5. Plan 07 alongside late Plan 06, but only when it closes real user workflows
6. Plan 08 once shared service ownership is strong enough to stop proliferating tool-local logic
7. Plan 09 after the viewer/runtime lanes are stable enough not to build editor work on sand
8. Plan 10 last

## Immediate Next Execution Recommendation

If the goal is practical momentum toward full old-viewer parity instead of another planning loop, the best immediate execution lanes are:

1. Plan 04 - MDX runtime and renderer closure
2. Plan 02 - shared format ownership closure for the world/model families still blocking app cutover
3. Plan 06 - world runtime and 3D world consumer cutover

Why this order now:

- `MDX` already has real momentum in `wow-viewer` and now has visual regression infrastructure, so it is the cleanest parity lane to keep cutting away
- shared format ownership is still the structural blocker for the rest of the world-viewer cutover
- the final credibility test for “feature parity with old MdxViewer” is not another standalone proof window; it is a real world consumer in `WowViewer.App`

## Proof Standard

For any numbered plan, proof must be stated precisely as one of:

- build-validated only
- test-validated only
- inspect/CLI proof on fixed real data
- visual regression proof on fixed real data
- runtime app proof on fixed real data
- compatibility-only old-viewer proof

Do not collapse those together when reporting progress.

## Bottom Line

The parity target is no longer “keep porting nice pieces over.”

The target is:

- `wow-viewer` owns the viewer stack
- `MdxViewer` gets cut away lane by lane
- each lane closes with a numbered plan, a proof standard, and an explicit ownership transfer
