# M2 Implementation Contract

## Scope

This contract describes what the current evidence requires the M2 reader and runtime to do.

It is implementation-facing. It should be the main handoff for building or correcting:

- M2 parsing in `wow-viewer`
- M2 runtime state and render preparation in `wow-viewer`
- narrow compatibility reuse in `MdxViewer`

## Ownership

### Canonical owner

`wow-viewer` owns the design for:

- model identity normalization
- root-model parse and validation
- skin-profile selection and loading
- active section and effect materialization
- external animation ownership
- runtime flags and effect-family selection
- future scene-submission and batching seams

### Compatibility-only consumer

`MdxViewer` can consume extracted `wow-viewer` M2 seams when needed for:

- parity probes
- active-viewer smoke validation
- stopgap compatibility work

It should not keep accumulating its own separate long-term M2 design.

## Current Contract By Seam

### 1. Model identity and extension gate

Current contract:

- accepted model-family requests may enter as `.mdl`, `.mdx`, or `.m2`
- later native clients normalize model-family identity to canonical `.m2`
- this normalization is separate from parser support

Implementation rule:

- `wow-viewer` should resolve canonical model identity before parser dispatch
- successful path normalization must not be mistaken for successful format support

Current evidence:

- `2.0.0.5610`: static-only
- `3.3.5`: static + runtime
- `4.0.0.11927`: static-only

### 2. Root-model validation

Current contract:

- early beta `2.0.0.5610` already uses `MD20`, but its active profile path differs from later numbered-skin clients
- `3.0.1.8303` is its own pre-release parser track with strict `MD20` and version `0x104..0x108`
- later Wrath and early Cataclysm evidence keeps strict `MD20`-family root validation and explicit structured setup

Implementation rule:

- parser dispatch must be build-aware
- do not collapse `2.0.0`, `3.0.1`, `3.3.5`, and `4.0.0` into one permissive parser path just because they all say `MD20`

### 3. Skin-profile ownership

Current contract:

- `2.0.0.5610`: active profile is root-contained; no `%02d.skin` proof on the traced path
- `3.0.1.8303`: do not assume missing external `.skin` means failure; root-contained profile evidence remains important
- `3.3.5` and `4.0.0.11927`: choose -> exact `%02d.skin` load -> validate/init -> rebuild live instances is explicit

Implementation rule:

- later native clients need first-class numbered-skin ownership
- early beta or pre-release builds need explicit separate handling instead of inheriting later `.skin` behavior blindly
- a skin file is structural runtime state, not just an index-buffer sidecar

### 4. Active section and effect materialization

Current contract:

- skin init copies and remaps active section records
- effect handles are built during or immediately after skin initialization
- unresolved flags such as `0x20` and propagated `0x40` remain architecturally important

Implementation rule:

- keep typed active-section state in the runtime layer
- do not flatten section, batch, and effect metadata too early
- preserve unresolved native flags in runtime-facing contracts until their semantics are closed

Current strongest evidence:

- `3.3.5`: static + runtime
- `4.0.0.11927`: static-only

### 5. External animation ownership

Current contract:

- later native clients use explicit `%04d-%02d.anim` sidecars
- sequence alias chains and ready-state behavior are part of the runtime seam

Implementation rule:

- external animation files belong in `wow-viewer` runtime ownership, not as an afterthought glued onto a renderer
- the runtime contract should preserve alias and readiness metadata

### 6. Effect-family and material routing

Current contract:

- native clients synthesize `Diffuse_*` and `Combiners_*` effect-family names explicitly
- Wrath and early Cataclysm both expose a rich effect vocabulary rather than one flat blend-mode path
- later Cataclysm also exposes a broader shader or effect stack around `.bls` assets and `ShaderEffectManager`

Implementation rule:

- keep an explicit runtime-owned effect recipe layer
- do not collapse everything into one generic local blend enum too early
- keep proof levels separate: some effect families are directly confirmed, some remain partial or research-only

### 7. Runtime flags

Current contract:

- visible user-facing toggles include z-fill, clip planes, threads, doodad batching, particle batching, additive particle sort, and optimization masks
- Wrath and early Cataclysm share the same broad low-bit option surface
- early Cataclysm currently differs in its observed default-return mask (`0x2008` versus the current Wrath reading centered on `0x8`)

Implementation rule:

- runtime flags need their own explicit contract
- do not hide them inside random renderer globals
- preserve cross-build differences when they are directly observed

### 8. Lighting and model-local animated state

Current contract:

- M2 lighting and emissive state is runtime-evaluated, not purely static file data
- material, color, alpha, and animation state interact with lighting and effect selection

Implementation rule:

- future lighting evaluation belongs in the M2 runtime seam, not as renderer-only ad hoc state
- the runtime contract should keep diffuse or emissive and animated material state explicit

### 9. Scene submission and batching

Current contract:

- native M2 rendering uses family-aware submission and batching
- doodads, particles, ribbons, and hit testing are not one monolithic render path
- final draw-submission closure is still incomplete in the current evidence set

Implementation rule:

- future `wow-viewer` M2 runtime should own typed render-entry families
- do not force every M2-derived draw through one blunt generic renderer path

## Unresolved But Important

- final user-facing label for the `0x20` shared-record class
- final semantics of propagated `0x40`
- exact final draw-submission path that consumes the compact runtime list
- precise mapping from some section or material flags into effect-family selection
- wider Cataclysm, Mists, and Warlords runtime confirmation

These must stay visible as unresolved. They should not be flattened away in the public runtime contract.

## Recommended Implementation Order

1. Keep `wow-viewer` as the only design owner for new M2 seams.
2. Complete the library-owned parser and skin-runtime foundations before deeper renderer work.
3. Land typed active-section and effect metadata before attempting full parity rendering.
4. Add external animation ownership and model-local runtime state next.
5. Only then expand into scene submission and batching.
6. Use `MdxViewer` only as a compatibility consumer or proof harness after extraction.

## MdxViewer Consumer Rules

- use extracted `wow-viewer` seams where possible
- keep compatibility-only stopgaps narrow and explicitly temporary
- do not claim `MdxViewer` build success as proof of runtime parity
