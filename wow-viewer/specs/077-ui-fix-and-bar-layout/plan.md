# Plan: 077 UI Fix — Animation Wiring, Toolbar, Layout Consolidation

**Generated from**: `spec.md`
**Branch**: `077-ui-fix-and-bar-layout` (cut from `071-left-right-sidebar-split`)
**Phases**: 5 phases, each independently validatable.

## Phase Map

| Phase | Scope | Risk | Commit Size |
|-------|-------|------|-------------|
| A | Fix animation wiring (standalone M2/MDX) | High | Large |
| B | Fix top toolbar (full-width, no cutouts) | Low | Small |
| C | Add bottom bar (above status bar) | Low | Medium |
| D | Minimap relocation + right sidebar dedup | Medium | Medium |
| E | Chunk Manipulation tab + UI audit | Medium | Medium |

## Order of execution

1. **A** (animation fix): Most critical — regression. Fix `M2Renderer.Animator` and `MdxRenderer.Animator` creation.
2. **B** (toolbar fix): Second most critical — user can't toggle options.
3. **C** (bottom bar): New surface for control checkboxes.
4. **D** (minimap + dedup): Cleanup — no functional regression, just relocation.
5. **E** (chunks + audit): Polish — reorganize terrain editing controls, audit dead UI.

## Phase dependencies

- A: independent (animation system, no layout dependency)
- B: independent of A
- C: must follow B (bar layout)
- D: must follow C (bottom bar is new anchor point for some minimap controls)
- E: must follow D (chunk manipulation controls are the ones being moved out of World > Tiles)

## Validation per phase

- A: build, load an M2 model, check Animation tab shows sequences and Play works
- B: build, verify toolbar spans full width, checkboxes visible at any sidebar width
- C: build, verify bottom bar renders above status bar with grid toggles
- D: build, verify no minimap in World > Tiles, no file browser in right sidebar
- E: build, verify chunk editing tools accessible, no dead controls in main tabs

## Animation Investigation (Phase A approach)

- `M2Renderer.Animator` returns `_legacyRenderer?.Animator ?? _runtimeAnimator`
- For native static renderer path: `_runtimeAnimator` created when `runtimeModel.Model.SequenceCount > 0`
- For MDX path: `MdxRenderer.Animator` returns `_animator`, created when `_enableM2Animation && (mdx.Bones.Count > 0 || MdxAnimator.HasAnimationData(mdx))`
- **Fix 1**: If `_runtimeAnimator` is null but `_legacyRenderer?.Animator != null`, use legacy animator in ALL M2Renderer constructors
- **Fix 2**: If `_runtimeAnimator` is null because `SequenceCount==0` but sequences exist in the model data, use a fallback detection path
- **Fix 3**: Add diagnostic logging when animator is null for a loaded model
- **Fix 4**: Ensure `MdxAnimator` is created for all MDX files that have ANY animation data (not just bones)
