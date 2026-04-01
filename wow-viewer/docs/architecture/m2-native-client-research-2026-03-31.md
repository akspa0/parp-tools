# Native M2 Client Research - Mar 31, 2026

## Scope

This note records native-client M2 findings gathered from Ghidra against the 3.3.5 OS X client and frames them as `wow-viewer` library work, not as more long-term ownership in `MdxViewer`.

The current confirmed findings now come from three native/reverse-engineering passes completed on Mar 31, 2026:

- the earlier 3.3.5 OS X native client pass
- a follow-up pass against the 3.3.5 PTR OS X PowerPC raw binary, after reconnecting the Ghidra bridge
- a focused Ghidra pass against the Win32 beta `2.0.0.5610` client to answer whether early `MD20 0x100` models really require external `.skin` companions

## Ownership Rule

- Treat these findings as input for the future `wow-viewer` M2 library and renderer path.
- Do not treat `MdxViewer` as the canonical implementation target for new M2 semantics.
- Any future parser, skin, section, or render-state work derived from this note should land in `wow-viewer` first, with `MdxViewer` only as an optional compatibility consumer.

## Early 2.0.0.5610 Boundary

The later native-client `%02d.skin` findings in this note are not universal across every M2 era.

Confirmed in the Win32 beta `2.0.0.5610` binary:

- no `.skin` or `%02d.skin` string was present in the loaded `Model2` code path during this pass
- `FUN_0072ee30` in `M2Shared.cpp` selects an active embedded profile from the root model table at `+0x4C` / `+0x50` and stores the chosen profile pointer at shared offset `+0x138`
- `FUN_0072f220` then builds `CM2Shared_idx` directly from that selected embedded profile
- `FUN_0072f3f0` builds `CM2Shared_vtx` directly from the same selected embedded profile

Implication for `wow-viewer`:

- the exact numbered `%02d.skin` ownership model remains correct for the confirmed later native passes in this note
- early beta `MD20 0x100` clients need a separate embedded-root-profile path instead of inheriting the later `.skin` rule by default

## Confirmed Native Pipeline Findings

### 1. Model cache entry normalizes all model-family inputs to `.m2`

Confirmed in native function `FUN_00957ca0`.

- `.mdl`, `.mdx`, and `.m2` requests are canonicalized to the base `.m2` path.
- The cache key is derived from the normalized basename.
- The cache creates or reuses a `CM2Model` entry, then queues the real file load asynchronously.

Implication for `wow-viewer`:

- the canonical identity for classic M2-family runtime loading should be the `.m2` asset, not a loose mixture of `.mdl` or `.mdx` compatibility guesses.

### 2. The model bootstrap is strict about `MD20`

Confirmed in native function `FUN_00984b00`, which calls `FUN_00987830`.

- the client validates the main model payload as `MD20`
- the current 3.3.5 native parser path accepted the expected Wrath-era version range and specifically handled the `0x108`-era structure

Implication for `wow-viewer`:

- `wow-viewer` should keep build/profile-aware M2 parsing strict and explicit instead of relying on broad fallback parsing that hides format mismatches.

### 3. In the 3.3.5 native pass, the active skin file is an exact numbered companion, not a fuzzy fallback

Confirmed in native function `FUN_00983970`.

- the client derives the skin path from the base model path using `baseName + %02d.skin`
- it opens the exact numbered skin chosen by the active skin-profile selection
- this is not a best-effort search over arbitrary companion skin names

Implication for `wow-viewer`:

- the library should model numbered skin ownership explicitly
- fallback behavior should stay clearly labeled as compatibility or recovery behavior, not source-of-truth behavior

### 4. Skin parse completion rebuilds live instances

Confirmed in native function `FUN_00983e40`.

- after the numbered skin profile is parsed and validated, the client initializes the active skin-profile state
- the client then rebuilds all live instances for that model by calling instance-reset and instance-init paths

Implication for `wow-viewer`:

- skin data is not a passive index buffer sidecar
- the runtime instance layout depends on the chosen skin profile and must be rebuilt from it

### 5. Skin-profile initialization materially reshapes the active renderable section data

Confirmed in native function `FUN_009833d0`.

Observed behavior:

- the skin profile copies section records into active shared storage
- it builds per-section helper arrays from skin records
- it remaps section data using the skin indices and bone-palette related data
- it propagates dependency state through a `0x40` flag across related records
- when necessary, it replaces or expands the shared section table pointer based on the active skin-profile section count

Implication for `wow-viewer`:

- the `.skin` file is defining active renderable section layout, not merely supplying triangle indices
- flattening batches or sections too early in the library would lose real native behavior
- section/bone palette remapping needs a first-class library seam

### 6. The native model bootstrap excludes a special class of shared records from the compact runtime list

Confirmed in native function `FUN_00984b00`.

- after model parse and skin initialization, the client allocates a compact runtime list only for shared records where a flag `0x20` is not set

Implication for `wow-viewer`:

- there is a real native distinction between generic render-list entries and a special-case class of records
- treating every submesh or batch as one uniform geoset path is likely wrong
- `0x20` needs to stay visible as an unresolved but important semantic flag during the first library-owned implementation

## PowerPC 3.3.5 PTR Confirmation Pass

The PowerPC raw binary did not preserve friendly function symbols, but it did preserve enough source-file paths, assert strings, and behavior to confirm the same core M2 ownership model.

### 1. PowerPC cache open confirms the `.mdl` and `.mdx` to `.m2` normalization path

Confirmed in `FUN_009d91c4`.

- the raw PowerPC binary strips the incoming extension, accepts `.mdl`, `.mdx`, or `.m2`, and normalizes legacy requests to `.m2`
- it hashes the normalized basename into a cache bucket
- it opens the normalized file and constructs the model cache entry from that path

This is the same cache ownership boundary found in the earlier native pass, not an alternate platform-specific loader.

### 2. PowerPC has an exact numbered skin loader and an explicit skin warm-up probe

Confirmed in `FUN_00a06a10` and `FUN_005cead8`.

- `FUN_00a06a10` strips the current model extension, appends `%02d.skin`, opens the exact numbered skin file, allocates the async load job, and guards against reloading after `m_skinProfileLoaded`
- `FUN_005cead8` is an auxiliary probe path that normalizes `.mdl` and `.mdx` to `.m2`, then checks for `%02d.skin` companions from `00` through `03`

The important point is unchanged for this later native client: the runtime still treats numbered skin ownership as explicit and structured, not as a loose best-effort search over arbitrary sidecar files.

### 3. PowerPC preserves an explicit choose, load, and initialize split for skin profiles

Confirmed across `FUN_00a06bc8`, `FUN_00a06a10`, `FUN_00a0809c`, and `FUN_00a064b8`.

- `FUN_00a06bc8` chooses a skin profile based on runtime/model state, then hard-fails with native diagnostics if choose or load or texture-array allocation fails
- `FUN_00a0809c` validates the loaded skin payload, including byteswap-aware parsing in the non-native-endian case, then calls the active skin-profile initializer
- `FUN_00a064b8` performs the section-copy and remap work that turns skin data into the active shared runtime section state

This confirms that the native client still treats skin ownership as a staged pipeline:

1. choose skin profile
2. load exact numbered skin file
3. validate and relocate profile payload
4. build active shared/runtime section state
5. rebuild live instances

### 4. PowerPC confirms bone-palette remap and `0x40` propagation in the skin initializer

Confirmed in `FUN_00a064b8`.

- the function copies active 0x30-byte section records from skin data into working shared storage
- it remaps per-section palette or influence references through the model-side lookup tables
- if the active section count exceeds current shared capacity, it swaps in a replacement section table; otherwise it copies into the existing shared table
- it propagates a `0x40` flag from referenced section relationships onto dependent records

This is direct platform confirmation that the skin initializer is doing structural render-state work, not just uploading an index buffer.

### 5. PowerPC confirms post-load instance rebuild after skin-profile activation

Confirmed in `FUN_00a0809c`.

- once the skin profile validates and initializes successfully, the client marks the skin profile as loaded
- it then walks the callback list and rebuilds each live model instance through the existing callback/reset paths

This is the same runtime contract seen in the earlier OS X native pass and remains one of the strongest reasons not to flatten skin handling into a passive renderer detail.

### 6. PowerPC exposes extra native invariants worth preserving in the library design

Confirmed in `FUN_00a06bc8` and `FUN_00a064b8`.

- the choose/load path hard-fails on `Failed to choose skin profile`, `Failed to load skin profile`, `Failed to allocate texture array`, and `Failed to initialize model skin profile`
- the section setup path marks some records when `boneInfluences > 4`
- byteswap-aware validation is built into the skin parse path instead of being bolted on later

Implication for `wow-viewer`:

- keep the library seams strict and stateful
- keep endian-sensitive parsing in the parser layer
- keep section-class and influence-count distinctions visible in the runtime contract instead of flattening them away early

### 7. PowerPC confirms external animation-file ownership

Confirmed in `FUN_00a07c8c` and `FUN_00a084a4`.

- the client loads explicit `%04d-%02d.anim` files derived from the current model basename and sequence identifiers
- it follows sequence alias chains while loading and marks dependent sequence flags during that process
- the completion path validates the loaded sequence data and marks the resolved chain with ready state before runtime use

Implication for `wow-viewer`:

- external animation files should be treated as a first-class M2 runtime seam rather than as an afterthought glued on after parser completion
- sequence aliasing and ready-state flags should survive into the shared runtime contract

### 8. PowerPC exposes a shader or effect combiner system instead of one flat material mode

Confirmed in `FUN_00a03be8`.

- the client synthesizes effect names from texture-source mode plus blend-combiner mode, for example `Diffuse_T1`, `Diffuse_T2`, `Diffuse_Env`, `Diffuse_T1_T2`, and `Diffuse_T1_Env`
- it then pairs those with combiner families such as `Combiners_Opaque`, `Combiners_Mod`, `Combiners_Decal`, `Combiners_Add`, `Combiners_Mod2x`, `Combiners_Fade`, and two-layer variants such as `Combiners_Opaque_Opaque`, `Combiners_Mod_Add`, and `Combiners_Mod_Mod2x`
- if the effect does not already exist, the client creates it lazily and binds the derived state table into the effect object

Implication for `wow-viewer`:

- a future M2 runtime should not collapse everything into one generic blend enum too early
- the native client appears to own a small but meaningful effect vocabulary that is closer to explicit pass recipes than to loose material flags
- effect caching should live in a shared runtime-facing seam, not inside one legacy viewer path

### 9. PowerPC confirms model-local lighting and emissive state are evaluated per runtime model state

Confirmed in `FUN_009e7d08`, `FUN_009e92cc`, and the alternate path `FUN_009e8ccc`.

- the client computes `m_currentDiffuse` and `m_currentEmissive` dynamically from model-local color state, scene/view transforms, and additional flag-controlled lighting paths
- the same update path also refreshes animated scalar or matrix state that feeds section visibility, material values, and texture or alpha behavior
- parented or attached models can inherit transform or lighting context from their parent model before final runtime state is committed

Implication for `wow-viewer`:

- lighting evaluation is not separable from model animation state in the native pipeline
- performance work should avoid repeatedly rebuilding lighting or animated material state when the same model state can be cached across equivalent frame contexts
- visual parity work should keep diffuse or emissive state explicit in the runtime contract rather than hiding it inside ad hoc renderer globals

### 10. PowerPC exposes scene-side batching for doodads, particles, ribbons, and hit testing as separate paths

Confirmed in `FUN_009fe2c4`, `FUN_009fea98`, `FUN_009fcbf8`, `FUN_009fb9cc`, `FUN_009f89fc`, and `FUN_009f9610`.

Observed behavior:

- doodad geometry is submitted in chunked batches, with runtime code choosing between a skinned upload path and a software-expanded bone-matrix path depending on current conditions
- particle emitters are merged into compatible batches only while format, material, and capacity constraints remain compatible
- ribbons use their own effect path and their own scene submission logic instead of sharing the generic doodad batch path
- scene hit testing is a separate mode that builds a candidate list, sorts by distance or priority, and resolves nearest hits without reusing the normal render submission path
- scene-side render entries are classified into families before a comparator-driven ordering pass assigns final batch group indices

Implication for `wow-viewer`:

- the native runtime already treats doodads, particles, ribbons, and hit testing as distinct orchestration problems
- one monolithic "model renderer" abstraction in the future wow-viewer runtime would be too blunt
- batching policy should stay family-aware and capacity-aware instead of forcing every M2-derived draw through one submission path

### 10a. PowerPC also exposes a single classified scene-submission loop above those handlers

Confirmed in `FUN_009ff5a4` and the entry handlers it dispatches.

Observed behavior:

- the scene walks one sorted, classified render-entry list and dispatches each entry by a small integer type
- the recovered switch targets currently map to at least these handler families:
	- `FUN_009fd830` - core batch submission
	- `FUN_009fcee8` - projected-batch submission
	- `FUN_009fe2c4` - doodad batch submission
	- `FUN_009fcbf8` - ribbon submission
	- `FUN_009ff114` - particle dispatch
	- `FUN_009fe16c` - callback-owned scene submission
- after the per-entry loop completes, the function unwinds a family of stacked GX state blocks rather than assuming one flat render state for the whole frame

Implication for `wow-viewer`:

- the future runtime should have one explicit scene-submission coordinator above the family-specific handlers
- render-family choice is a first-class native concern, not an incidental detail hidden inside one batch function
- render-state lifetime should be explicit and scoped because the native client is clearly pushing and unwinding family-specific state stacks

### 10b. Particle submission has both a direct path and a merged-batch path

Confirmed in `FUN_009ff114` and `FUN_009fea98`.

- when batching is disabled or incompatible, the client drives a direct single-emitter render path
- when batching is enabled, it merges adjacent compatible emitters only while effect, capacity, and per-emitter constraints remain valid
- the recovered strings and behavior match the preserved runtime knobs for particle batching and additive sorting

Implication for `wow-viewer`:

- particle performance work should preserve a deliberate fallback path instead of assuming batching is always legal or always faster
- additive or transparent particle ordering should remain a visible runtime policy knob during early wow-viewer implementation

### 10c. The scene ordering pass is primarily a state-batching comparator, not a pure depth sort

Confirmed in the PowerPC scene classifier `FUN_009f9610` and its comparator body recovered in `FUN_009f9960`.

Observed behavior:

- `FUN_009f9610` classifies scene entries into render families, assigns intermediate group ids, and then sorts the resulting list through the generic heap-sort helper `FUN_009d8cec`
- the comparator first groups entries by model or resource identity through the renderable object pointer chain
- it then compares the classifier-assigned per-entry batch key stored in the scene record
- for otherwise similar entries it continues through material or section-state payloads, including memcmp-style comparisons over multiple 12-byte state blocks and a float fallback term near the end of the chain
- the recovered ordering is therefore built to maximize stable state grouping before submission, not to implement one simple global back-to-front rule

Implication for `wow-viewer`:

- the future M2 scene layer should separate state-batching order from final transparent or additive submission policy instead of trying to treat the whole native ordering pass as one depth comparator
- batching keys should remain explicit in the runtime contract because the native client is clearly sorting on more than one dimension before draw dispatch

### 10d. The lower CSimpleRender batching path also splits on state changes and buffer limits

Confirmed in `FUN_008692f8` and its main caller `FUN_0086e804`.

Observed behavior:

- the helper walks prebuilt render entries and starts a new batch whenever `entry->m_texSort` changes
- it also splits when the effective blend or state bucket changes, when another per-entry state word changes, or when the accumulated vertex or index counts would overflow the current hardware limits
- this is a generic lower rendering layer rather than an M2-only function, but it matches the same pattern seen higher in the M2 scene code: keep compatible state together, then flush deliberately at state or capacity boundaries

Implication for `wow-viewer`:

- if wow-viewer wants native-like performance behavior, it needs explicit batch-boundary logic instead of assuming one continuously growing transparent or opaque list is correct
- texture-sort and state-bucket ownership should stay visible through the runtime seam instead of being hidden inside one renderer-local side effect

### 10e. The preserved M2 option family is now confirmed as real registration code that feeds a shared runtime flag word

Confirmed from the currently unlabeled PowerPC code region spanning the M2 option strings near `0x00a1a5ec` through the follow-up flag-application blocks near `0x00a1850c` through `0x00a18644`.

Observed behavior:

- the M2 options are not just passive data descriptors; the same raw region also contains executable registration code that calls the generic cvar-registration helper `FUN_000abf70`
- that registration path walks the preserved M2 family in order and stores the resulting option handles into globals:
	- `DAT_010a538c` - `M2UseZFill`
	- `DAT_010a5388` - `M2UseClipPlanes`
	- `DAT_010a5384` - `M2UseThreads`
	- `DAT_010a5380` - `M2BatchDoodads`
	- `DAT_010a537c` - `M2BatchParticles`
	- `DAT_010a5378` - `M2ForceAdditiveParticleSort`
	- `DAT_010a5374` - `M2Faster`
	- `DAT_010a5370` - `M2FasterDebug`
- follow-up raw blocks in the `0x00a185xx` through `0x00a186xx` range read and write the shared M2 runtime flag word through the helpers:
	- `FUN_009d911c` - returns `DAT_010a3b2c`
	- `FUN_009d9128` - writes `DAT_010a3b2c`
	- `FUN_009d90a4` - ORs `param & 0xe000` into `DAT_010a3b2c`
- the additive-particle callback block is now concretely anchored: it updates the shared M2 runtime flag word and emits the preserved logs `Sorting all particles as though they were additive.` and `Sorting particles normally.`
- the registration or initialization path also references the labels `Internal` and `ClipPlanes`, which is the strongest current hint that clip-plane behavior is applied as a named scene-optimization mode rather than as an incidental low-level toggle
- the shared flag word is already consumed in live native code:
	- `FUN_009ef224` and `FUN_009dc1c4` gate combinable doodad-batch behavior on bits `0x20` and `0x40`
	- `FUN_00768604` reads bit `0x8` during a shader or effect initialization path that reloads `MapObj.wfx`, `MapObjU.wfx`, `Model2.wfx`, `Particle.wfx`, and `ShadowMap.wfx`

Implication for `wow-viewer`:

- the future runtime should preserve a small explicit option-to-flag application layer instead of hardcoding clip planes, batching, additive particle sorting, or faster-scene modes directly inside render handlers
- the M2 runtime flag word is a real native coordination seam between option setup and live renderer behavior
- the first library-owned implementation should keep the flag word semantically explicit even before every bit is fully named, because at least some bits already control doodad batching and shader-path initialization

### 11. PowerPC keeps explicit optimization knobs for batching, z-fill, clip planes, threads, and additive particle sorting

Observed directly from preserved strings in the PowerPC binary:

- `M2UseZFill`
- `M2UseClipPlanes`
- `M2UseThreads`
- `M2BatchDoodads`
- `M2BatchParticles`
- `M2ForceAdditiveParticleSort`
- `M2Faster`
- `M2FasterDebug`

The exact registration function for these controls was not fully recovered in this pass, but the strings are real and tightly grouped with the `Model2` runtime. They are not generic engine noise.

Additional PowerPC evidence from the current pass:

- the option-name strings sit in a currently unlabeled raw-code region that includes both registration-time string references and follow-up handle writes, not just isolated passive data descriptors
- nearby named functions in the `0x00a19...` and `0x00a1a...` range that were tested so far turned out to be unrelated startup, logging, or signature-validation helpers, which is why the real M2 option owner stayed hidden until the raw `xrefs_from` pass
- the clip-plane description string `use clip planes for sorting transparent objects` lands in the same descriptor-style region
- the additive-particle logging strings `Sorting all particles as though they were additive.` and `Sorting particles normally.` also resolve through parameter-style references
- the z-fill enabled or disabled strings were found separately in a broader engine-config area, which suggests the underlying z-fill toggle may be registered through a more global config path even if it still affects M2 behavior

Implication for `wow-viewer`:

- future performance work should preserve room for explicit runtime feature switches around batching, clip-plane use, z-fill behavior, additive-particle ordering, and threaded work submission
- do not assume the optimal path is always the same for all model families or all passes
- build the wow-viewer runtime so these policies can be surfaced and measured instead of being hardcoded into one opaque renderer path

## Native Function Anchors

These are the key native functions identified and commented in Ghidra across the two completed passes.

| Address | Function | Confirmed role |
| --- | --- | --- |
| `0x00957ca0` | `FUN_00957ca0` | model-family canonicalization, cache lookup, async model load queue |
| `0x00984b00` | `FUN_00984b00` | core `CM2Model` bootstrap |
| `0x00987830` | `FUN_00987830` | raw `MD20` parser and relocation/validation path |
| `0x00983970` | `FUN_00983970` | exact numbered `%02d.skin` load |
| `0x00983e40` | `FUN_00983e40` | skin parse completion and live-instance rebuild |
| `0x009833d0` | `FUN_009833d0` | active skin-profile section initialization/remap |
| `0x009d91c4` | `FUN_009d91c4` | PowerPC cache open and `.mdl`/`.mdx` to `.m2` normalization |
| `0x005cead8` | `FUN_005cead8` | PowerPC numbered skin probe from `00.skin` through `03.skin` |
| `0x00a06a10` | `FUN_00a06a10` | PowerPC exact numbered skin loader |
| `0x00a06bc8` | `FUN_00a06bc8` | PowerPC skin profile choose/load wrapper and texture-array setup |
| `0x00a064b8` | `FUN_00a064b8` | PowerPC active skin-profile section initialization/remap |
| `0x00a0809c` | `FUN_00a0809c` | PowerPC skin parse completion and live-instance rebuild |
| `0x00a07c8c` | `FUN_00a07c8c` | PowerPC external `%04d-%02d.anim` loader and alias-chain flagging |
| `0x00a084a4` | `FUN_00a084a4` | PowerPC external animation completion and sequence ready-state validation |
| `0x00a03be8` | `FUN_00a03be8` | PowerPC effect-combiner builder and simple-effect cache |
| `0x009e7d08` | `FUN_009e7d08` | PowerPC model animated material or skin-state update path |
| `0x009e92cc` | `FUN_009e92cc` | PowerPC model transform or parent-propagation animation path |
| `0x009e8ccc` | `FUN_009e8ccc` | PowerPC alternate lighting or material-evaluation path |
| `0x009f89fc` | `FUN_009f89fc` | PowerPC scene hit-list growth helper |
| `0x009fb9cc` | `FUN_009fb9cc` | PowerPC scene hit-test resolve path |
| `0x009f9610` | `FUN_009f9610` | PowerPC scene render-list classifier and batch-group ordering prep |
| `0x009f9960` | `FUN_009f9960` | PowerPC M2 scene comparator body recovered for state-batching order |
| `0x009fd830` | `FUN_009fd830` | PowerPC core batch submission path |
| `0x009fcee8` | `FUN_009fcee8` | PowerPC projected-batch submission path |
| `0x009fe2c4` | `FUN_009fe2c4` | PowerPC doodad batch submission path |
| `0x009fea98` | `FUN_009fea98` | PowerPC particle batch submission path |
| `0x009fcbf8` | `FUN_009fcbf8` | PowerPC ribbon submission path |
| `0x009ff114` | `FUN_009ff114` | PowerPC particle dispatch path |
| `0x009ff5a4` | `FUN_009ff5a4` | PowerPC main scene submission loop |
| `0x008692f8` | `FUN_008692f8` | Lower `CSimpleRender` batch splitter by `m_texSort`, state bucket, and buffer limits |
| `0x009d911c` | `FUN_009d911c` | shared M2 runtime-flag getter for `DAT_010a3b2c` |
| `0x009d9128` | `FUN_009d9128` | shared M2 runtime-flag setter for `DAT_010a3b2c` |
| `0x009d90a4` | `FUN_009d90a4` | helper that ORs `0xe000`-class scene-optimization bits into the shared M2 runtime flag word |

## Option Descriptor Anchors

The exact unlabeled function boundaries are still open, but the current M2 option region and its handle writes are now concrete.

| Address | Kind | Notes |
| --- | --- | --- |
| `0x00a1a5ec` | raw code anchor | loads `M2UseZFill` for registration and later writes handle `DAT_010a538c` |
| `0x00a1a620` | raw code anchor | loads `M2UseClipPlanes` and later writes handle `DAT_010a5388` |
| `0x00a1a654` | raw code anchor | loads `M2UseThreads` and later writes handle `DAT_010a5384` |
| `0x00a1a68c` | raw code anchor | loads `M2BatchDoodads` and later writes handle `DAT_010a5380` |
| `0x00a1a6c4` | raw code anchor | loads `M2BatchParticles` and later writes handle `DAT_010a537c` |
| `0x00a1a6fc` | raw code anchor | loads `M2ForceAdditiveParticleSort` and later writes handle `DAT_010a5378` |
| `0x00a1a734` | raw code anchor | loads `M2Faster` and later writes handle `DAT_010a5374` |
| `0x00a1a75c` | raw code anchor | loads `M2FasterDebug` and later writes handle `DAT_010a5370` |
| `0x00a1a608` | raw code anchor | loads `use clip planes for sorting transparent objects` |
| `0x00a18628` | raw callback anchor | emits `Sorting all particles as though they were additive.` |
| `0x00a18644` | raw callback anchor | emits `Sorting particles normally.` |

## What This Means For wow-viewer

The future `wow-viewer` M2 source of truth should be built around native ownership boundaries that the client clearly expresses.

### Recommended library seams

1. A library-owned M2 document model that represents the main `MD20` payload and the chosen numbered skin as one runtime-bound unit.
2. A library-owned active skin-profile builder that produces the renderable section list from the chosen skin, rather than treating the skin as a later renderer convenience.
3. A library-owned section classification layer that preserves unresolved but important native flags such as `0x20` and `0x40` instead of collapsing them away.
4. A library-owned runtime section/bone-palette remap step before renderer submission.
5. A renderer-facing pass classifier that keeps opaque/transparent/special-case section routing explicit rather than flattening everything into one generic geoset path.
6. A runtime-owned effect recipe layer that maps M2 material or combiner state into explicit pass/effect choices instead of burying that logic in one legacy renderer.
7. Distinct runtime submission services for doodads, particles, ribbons, and hit testing rather than a single undifferentiated draw path.
8. A runtime-owned animation asset seam that includes external `%04d-%02d.anim` files, sequence aliases, and readiness state.
9. A scene-owned submission coordinator that dispatches classified render entries into family-specific handlers and owns family-scoped render-state lifetimes.

### Recommended project direction

- keep this work out of `MdxViewer` as the design owner
- implement the next M2-native slice in `wow-viewer` as the canonical path
- if compatibility is needed later, wire `MdxViewer` to consume the `wow-viewer` library result rather than cloning the semantics again

## Current Gaps

The completed pass did not yet close these items.

- exact semantic meaning of section flag `0x20`
- exact semantic meaning of the propagated section dependency flag `0x40`
- the exact final draw submission path for the compact runtime section list after scene classification
- the exact unlabeled function boundaries for the recovered M2 option-registration and option-application blocks
- the exact runtime switch points for z-fill, clip-plane use, threads, and additive particle sorting
- the precise mapping from section or material flags into the native combiner/effect families
- how transparent or additive submission policy interacts with the recovered scene batching comparator and the lower state-split batch helpers

## Remaining PowerPC Follow-up

The main PowerPC confirmation pass is now complete enough to support library design, but a few questions are still open.

Next PowerPC follow-up targets:

1. recover the final draw-submission path that consumes the compact runtime section list
2. determine whether the PowerPC build exposes clearer semantics for the unresolved `0x20` section class
3. determine whether the PowerPC build exposes a clearer semantic name for the propagated `0x40` dependency flag
4. recover the missing function boundaries around the M2 option-registration and option-application blocks near `0x00a1a5ec` and `0x00a1850c`
5. trace the shared M2 runtime flag bits from `DAT_010a3b2c` into the remaining live transparent, clip-plane, z-fill, and particle-ordering paths

## Next wow-viewer Slice

The next correct implementation slice is not another `MdxViewer` workaround.

The next slice should be a narrow `wow-viewer` library-first M2 runtime seam that:

- loads `MD20` plus exact numbered skin ownership explicitly
- builds the active skin-profile section list in library code
- preserves unresolved native section flags instead of flattening them away
- exposes a renderer-ready section contract for later pass routing

That should become the definitive source of truth for M2 runtime ownership in `wow-viewer`.