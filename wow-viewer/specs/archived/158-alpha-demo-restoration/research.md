# Phase 0 Research: Alpha Demo Restoration

## 1. US1 (WTF reading) is already delivered

**Decision**: Adopt `WowViewer.Core.IO.Wtf` (Spec 159, committed `f0dffdaa`) unmodified for reading and
classifying WTF file content. No new reader is built for this spec.

**Rationale**: Spec 159's `WtfSweeper`/`WtfLineClassifier` already reads a WTF file's `SET`, `bind`, and
`PortCommandCandidate`-shaped lines, with real, direct validation against 0.5.3.3368 and 2.0.0.5610
(committed real output: 245/245 and 153/153 lines recognized). Rebuilding this would violate Format
Reader/Writer Ownership and duplicate already-working, already-tested code.

**Alternatives considered**: A dedicated reader scoped to this spec. Rejected — there is nothing left to
build; US1's acceptance scenarios are already met by existing code.

## 2. Alt+P binding is real, not assumed

**Decision**: Bind Alt+P to `_showPerfWindow` exactly as `WTF\DefaultBindings.wtf` declares it.

**Rationale**: Spec 159 read this file directly from 0.5.3.3368's real archives:
`bind ALT-P TOGGLEPERFORMANCEDISPLAY`. This is measured, not remembered — the same file also confirms
`ALT-O TOGGLEPERFORMANCEVALUES`, `CTRL-P RESETPERFORMANCEVALUES`, `CTRL-R TOGGLEFPS`,
`CTRL-Y TOGGLESTATS`, `CTRL-Q TOGGLETRIS`, `CTRL-W TOGGLEPORTALS`, `CTRL-E TOGGLECOLLISION`,
`CTRL-T TOGGLECOLLISIONDISPLAY`, `ALT-B TOGGLEPLAYERBOUNDS` — none of which are in this spec's scope
(only `TOGGLEPERFORMANCEDISPLAY` maps to something this viewer already has, the perf overlay), but all of
which are recorded here as a known, real, future candidate list rather than re-discovered later.

**Alternatives considered**: None — the user specified this exact binding from memory, and it is now
independently confirmed against real client data, so there is nothing to weigh.

## 3. `Key.AltLeft`/`Key.AltRight` exist in the vendored input library

**Decision**: Use `Key.AltLeft`/`Key.AltRight` directly, mirroring the existing `Key.ControlLeft`/
`Key.ControlRight` modifier-check pattern at `ViewerApp.cs:1305-1312`.

**Rationale**: Confirmed directly against the restored `Silk.NET.Input.Common` 2.21.0 assembly (the
version this project's `packages.lock.json` resolves) — both enum members are present. This was flagged
as unverified in spec.md and is resolved here rather than left for implementation time to discover.

**Alternatives considered**: None needed; the check was binary (exists or doesn't) and it exists.

## 4. Validation build for US2/US4/US5

**Decision**: Use 0.5.3.3368 as the primary validation build for worldport/teleport execution,
camera-follow, and attachment/lighting work.

**Rationale**: It is the build this session has the deepest existing tooling and measured grounding
against (Spec 155's asset sweep, Spec 159's WTF sweep, WMO corpus work all already validated here), its
model route reads without the Spec 154 MD20 era blockers (Alpha `MDLX` route works), and it is the build
`WTF\DefaultBindings.wtf` was actually read from. Using a second build (2.0.0.5610) as a cross-check for
US2/US3 is worthwhile once the first pass is validated, but is not required to call any phase done.

**Alternatives considered**: 2.0.0.5610 as primary — rejected only as the *primary* target because its
model route (MD20 0x100/0x101, Spec 154) has not been confirmed fully working the way 0.5.3.3368's has;
using it as primary would risk conflating a model-reading defect with a camera/lighting defect during
validation.

## 5. US6 status — carried forward unchanged

**Decision**: Phase 6 does no work. US6 remains blocked on Spec 159 finding a real, uncatalogued WTF
source file (or the user providing one).

**Rationale**: Nothing in Phases 1-5 changes this dependency, and nothing in this plan should imply
otherwise. Spec 159's `wtf probe` capability (candidate-name testing directly against a build's archive
hash table, bypassing every listfile) remains the live, ready mechanism for when a real name is available.

**Alternatives considered**: Fabricating a synthetic "demo" command sequence to exercise US2's dispatcher
end-to-end for demo-flavored testing. Rejected — Phase 2's own hand-written test commands (spec.md's
Independent Test for US2) already cover this without pretending a fabricated file is the real thing US6
asks for.

## 6. Camera follow-target design

**Decision**: A `CameraFollowTarget` holding a model instance reference and a bone index/KeyBoneId,
resolved to a world transform each frame via the pipeline-appropriate bone-matrix source (legacy
`MdxAnimator.BoneMatrices`, already public; modern `M2BonePoseState.Matrices`, needs one new small public
accessor on `M2Renderer`). `Camera` gains an optional follow-target field; when set, position/orientation
are derived from the target each frame instead of from WASD input — the same per-frame external-drive
shape already proven by `ViewerApp_CameraPaths.cs:679-688`.

**Rationale**: Reuses already-computed data (no new bone math) and an already-proven camera-drive
pattern. The only genuinely new code is the small accessor and the follow-target/detach state machine.

**Alternatives considered**: A parallel bone-evaluation path dedicated to camera-follow. Rejected — both
pipelines already compute full per-bone world matrices every frame for rendering; a second computation
would be pure duplication.

## 7. Dynamic lighting first-pass approach

**Decision**: A small, fixed-capacity point-light array (exact count decided against real frame-time
measurement during Phase 5b, not fixed in advance) uploaded as a shader uniform/buffer, additive with the
existing directional+ambient term. Nearest-N selection (or an equivalent cheap bound) keeps cost scoped to
lights actually near the camera rather than total scene light count.

**Rationale**: This mirrors how every other real-time renderer bounds dynamic light cost, and keeps the
first version simple enough to actually ship and validate rather than over-building a general lighting
system this spec does not need. `TerrainShader.cs` currently has zero point-light infrastructure of any
kind, so any bounded approach is strictly additive risk, never a regression path, provided it is gated
correctly (Phase 5b step 7's explicit zero-lights regression check against the Terrain Alpha Risk Area
baseline).

**Alternatives considered**: Full deferred/tiled light culling. Rejected as premature — this spec needs
"a torch visibly lights the area around it," not a general many-light renderer; that scope belongs to a
future spec if this project ever needs more than a handful of simultaneous dynamic lights.

## Open Research Boundaries

- Exact validation model(s) for M2Era100 attachment parsing (Phase 5a) are not pinned here — "a
  torch-carrying NPC or any hand-slot-bearing humanoid" is deliberately left to be selected from what's
  actually staged and loadable at implementation time, not guessed now.
- Exact dynamic point-light count/falloff constants are implementation-time decisions validated against
  real frame timing, not specified here as fixed numbers.
