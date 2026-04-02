# Native M2 Client Research - Mar 31, 2026

Implementation-facing readers should start with `docs/architecture/m2/`.

This file remains the raw native evidence log and build-by-build behavior notebook behind that consolidated doc set.

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

Later Win32 `3.3.5.12340` decompilation tightened that boundary further:

- the same `0x20` bit repeatedly gates nested pointer or track relocation in helper families whose record sizes line up with track-bearing root-model blocks, not `.skin` texture-unit flags
- exact size matches now confirmed from the in-repo wowdev docs:
	- `0x14` -> single `M2Track` payloads such as `M2TextureWeight`
	- `0x3c` -> exact `M2TextureTransform`
	- `0x9c` -> exact `M2Light`
- this makes the best current reading: `0x20` marks a shared-record class with attached animated payloads or nested track state that stays out of the compact renderable section list and receives special relocation handling during bootstrap

Implication for `wow-viewer`:

- there is a real native distinction between compact render-list entries and a special shared-record class that owns richer animated payloads
- treating every submesh or batch as one uniform geoset path is likely wrong
- the exact user-facing name of that class is still open, but `0x20` is no longer just an unknown batch flag and should stay explicit in the first library-owned implementation

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

## Win32 3.3.5.12340 x64dbg Breakpoint Anchors (Apr 01, 2026)

These anchors were confirmed from offline Ghidra string xrefs in the loaded Win32 `WoW.exe` (`3.3.5.12340`) and are intended for live runtime validation through `x64dbg-mcp`.

| Address | Current Label | Confirmed role |
| --- | --- | --- |
| `0x00835a80` | `M2_FormatSkinFilename_02d` | builds exact `%02d.skin` companion path from model basename |
| `0x0083cc80` | `M2_ChooseAndLoadSkinProfile` | selects skin profile, loads skin payload, allocates texture array; emits `Failed to choose/load skin profile` diagnostics |
| `0x00838490` | `M2_InitializeSkinProfileAndRebuildInstances` | validates skin payload blocks, initializes active skin profile, emits `Corrupt skin profile data` and `Failed to initialize model skin profile`, rebuilds live instances |
| `0x00835a20` | `M2_FormatAnimFilename_04d_02d` | builds external animation filename `%04d-%02d.anim` |
| `0x00836600` | `M2_BuildCombinerEffectName` | maps blend/flags into `Diffuse_*` + `Combiners_*` effect families |
| `0x00402760` | `M2_RegisterRuntimeFlags` | registers `M2UseZFill`, `M2UseClipPlanes`, `M2UseThreads`, `M2BatchDoodads`, `M2BatchParticles`, `M2ForceAdditiveParticleSort`, `M2Faster`, `M2FasterDebug` and folds enabled state into a runtime bitfield |
| `0x0053c430` | `M2_NormalizeModelPathAndProbeSkins` | normalizes `.mdl`/`.mdx` to `.m2` and probes `00.skin` through `03.skin` companions |

Practical debugging order for runtime capture:

1. `0x0083cc80` (choose/load skin)
2. `0x00838490` (skin init/rebuild)
3. `0x00836600` (combiner/effect routing)
4. `0x00835a80` and `0x00835a20` (path formatting confirmation)

This gives one contiguous evidence chain from skin ownership to draw-state family routing without requiring live Ghidra process attach.

### Win32 first world-path choose-load capture (Apr 02, 2026)

Live x64dbg sampling in an in-world client state now confirms the choose/load side of the chain for a real world doodad path on Win32 `3.3.5.12340`:

- model object path at formatter entry:
	- `world\expansion02\doodads\generic\barbershop\barbershop_mirror_01.m2`
- exact numbered skin formatter output from the stack-local destination buffer:
	- `world\expansion02\doodads\generic\barbershop\barbershop_mirror_0100.skin`
- formatter call-frame evidence:
	- source path pointer was the world-model basename
	- skin profile argument was `0`
- post-load result at `0x0083cd32` remained success (`EAX=1`) for the same world-model chain
- callback rebuild activity surfaced immediately after successful world-path load with repeated hits at `0x00832ea0`

Current boundary on this first world-path pass:

- this proves real world-path numbered skin ownership and successful load on Win32 Wrath, not just UI or portrait traffic
- the explicit `0x00838490` skin-init entry did not surface cleanly in this noisy in-world run even though downstream callback-rebuild hits did
- `0x00836600` combiner-family capture is still only confirmed from the earlier UI-path chain; a world-path effect-family sample is still pending

### Win32 follow-up world-path capture and init reachability (Apr 02, 2026)

The same in-world session produced a second confirmed world-path choose-load chain and also proved that the explicit init path is reachable once noisy choose/load breakpoints are removed:

- second confirmed world-path model:
	- `world\expansion02\doodads\generic\barbershop\barbershop_shavecup.m2`
- exact numbered companion output from the formatter destination buffer:
	- `world\expansion02\doodads\generic\barbershop\barbershop_shavecup00.skin`
- post-load result again remained success (`EAX=1` at `0x0083cd32`)
- downstream callback rebuild again surfaced immediately after successful world-path load at `0x00832ea0`
- after deleting the noisy choose/load breakpoints and leaving downstream stops armed, execution reached:
	- `0x00838490` (`M2_InitializeSkinProfileAndRebuildInstances`)
	- `0x00838561` (loaded-state write inside the same init path)

Current boundary on this follow-up pass:

- the explicit init path is now proven reachable in the active in-world session, but the isolated init samples captured after breakpoint cleanup were still dominated by UI-model callers such as `ui_nightelf.m2` and `ui-autocastbutton.m2`
- `0x00836600` combiner-family routing was also re-hit after cleanup, but the sampled caller remained UI-path in this pass rather than one of the confirmed world doodads
- the x64dbg MCP session timed out and dropped before a clean world-attributed `0x00838490` or world-attributed `0x00836600` sample could be harvested

### Win32 reattach pass: first world-attributed combiner and init-completion samples (Apr 02, 2026)

After restarting x64dbg and reattaching with only narrow downstream breakpoints, the next pass finally produced clean world-attributed samples for both effect routing and init completion.

- world-attributed combiner caller object:
	- `world\generic\human\passive doodads\beds\duskwoodbed.m2`
- combiner callsite:
	- `0x00836600` (`M2_BuildCombinerEffectName`)
- resolved world-path selection inside that call:
	- texture-side selection set `[ebp+0x0C] = 0x00A45854` -> `Diffuse_T2`
	- combiner-family dispatch selected `0x00A45838` -> `Combiners_Mod2x`
- this gives the first clean world-path effect-family sample in the Win32 Wrath session instead of only UI-path effect routing

- world-attributed init-completion sample:
	- the same `duskwoodbed.m2` object was later observed at `0x00838561` (`or dword ptr ds:[edi+0x08], 0x02`), the loaded-state write inside `M2_InitializeSkinProfileAndRebuildInstances`
	- `EDI` at that stop pointed back to the same world model object that carried the `duskwoodbed.m2` path

Current boundary after the reattach pass:

- world-path choose/load is now confirmed on multiple doodads
- world-path effect routing is now confirmed with one concrete sample: `Diffuse_T2` + `Combiners_Mod2x`
- world-path init completion is now confirmed at the loaded-state write inside the init routine
- a clean world-attributed stop at the entry of `0x00838490` itself is still optional follow-up, but the stronger practical boundary is now closed because the init routine and its loaded-state write have been observed on a world object

### Win32 static contract consolidation from decompilation (Apr 02, 2026)

The Wrath Win32 M2 runtime contract is now backed by direct decompilation for the main choose/load/init/effect seams, not just string-xref and runtime sampling.

#### 1. Skin-profile choose logic uses a real threshold ladder

`M2_ChooseAndLoadSkinProfile` (`0x0083cc80`) selects the active profile by comparing the current runtime threshold against the model-side table under `+0x44`.

The live data table at `0x00A45644` is:

- `0x100`
- `0x40`
- `0x35`
- `0x15`

This matches the earlier runtime observations where live world and UI samples repeatedly showed the active threshold path `0x40`.

#### 2. Skin load is staged and asynchronous before init

`FUN_0083cb40` is the exact numbered-skin load wrapper:

- builds `%02d.skin` through `M2_FormatSkinFilename_02d`
- opens the file through the model file-open path
- allocates the skin payload block into `+0x170`
- allocates an async job record at `+0x0c`
- sets completion callback `FUN_0083cb10`

`FUN_0083cb10` then calls `M2_InitializeSkinProfileAndRebuildInstances` on completion and frees the async job state.

#### 3. Init path is strict relocation plus rebuild

`M2_InitializeSkinProfileAndRebuildInstances` (`0x00838490`) performs a strict block-relocation and validation chain before the runtime accepts the skin profile.

The decompiled order is:

- `FUN_00835df0`
- `FUN_00835df0`
- `FUN_00835c20`
- `FUN_00835ae0`
- `FUN_00835b30`
- `FUN_00837a40`

If any stage fails, the client logs either:

- `Corrupt skin profile data: %s`
- `Failed to initialize model skin profile: %s`

On success it:

- sets the loaded bit at `param_1 + 8 |= 2`
- drains the live callback list through `FUN_00824510` and `FUN_00832ea0`

#### 4. `FUN_00837a40` is the real section or batch materialization seam

`FUN_00837a40` is not a trivial helper. It materializes the active skin-driven runtime state by:

- allocating the active copied section block at `+0x18c`
- calling `FUN_00836980` and `FUN_00837680` to classify and merge section-state flags
- building effect handles into `+0x188` by iterating skin records through `FUN_00836c90` and `M2_BuildCombinerEffectName`
- copying and remapping section records into the shared runtime table at the model-side `+0x40`
- propagating `0x40` relationship bits across dependent entries
- replacing the shared table pointer when the active section count exceeds capacity, otherwise copying into the existing table

This directly reinforces the earlier conclusion that `.skin` ownership is structural runtime state, not just an index-buffer sidecar.

#### 5. `M2_BuildCombinerEffectName` now has a decompiled decision tree

For `param_2 == 1`, the function chooses one of:

- `Diffuse_T1`
- `Diffuse_T2`
- `Diffuse_Env`

paired with one of:

- `Combiners_Opaque`
- `Combiners_Mod`
- `Combiners_Decal`
- `Combiners_Add`
- `Combiners_Mod2x`
- `Combiners_Fade`

For the two-layer route it chooses:

- `Diffuse_T1_T2`
- `Diffuse_T1_Env`
- `Diffuse_Env_T2`
- `Diffuse_Env_Env`

paired with families including:

- `Combiners_Opaque_Opaque`
- `Combiners_Opaque_Mod`
- `Combiners_Opaque_Add`
- `Combiners_Opaque_Mod2x`
- `Combiners_Opaque_Mod2xNA`
- `Combiners_Opaque_AddNA`
- `Combiners_Mod_Opaque`
- `Combiners_Mod_Mod`
- `Combiners_Mod_Add`
- `Combiners_Mod_Mod2x`
- `Combiners_Mod_Mod2xNA`
- `Combiners_Mod_AddNA`
- `Combiners_Add_Mod`
- `Combiners_Mod2x_Mod2x`

The world-attributed `duskwoodbed.m2` runtime sample matches this tree exactly and resolved to:

- `Diffuse_T2`
- `Combiners_Mod2x`

#### 6. Special-case wrapper above the normal combiner path

`FUN_00836c90` wraps `M2_BuildCombinerEffectName` and adds a special high-bit route when the batch flags carry `0x8000`.

Recovered explicit special families include:

- `Combiners_Opaque_Mod2xNA_Alpha`
- `Combiners_Opaque_AddAlpha`
- `Combiners_Opaque_AddAlpha_Alpha`

with `Diffuse_T1_Env` on that special route before the function falls back to a normal `M2_BuildCombinerEffectName(..., 0x11, ...)` call when the first effect lookup fails.

#### 7. Runtime flag word semantics are now decompiled, not inferred

`M2_RegisterRuntimeFlags` (`0x00402760`) registers:

- `M2UseZFill` -> bit `0x1`
- `M2UseClipPlanes` -> bit `0x2`
- `M2UseThreads` -> bit `0x4`
- always returns with startup bit `0x8` set
- `M2BatchDoodads` callback -> bit `0x20`
- `M2BatchParticles` callback -> bit `0x80`
- `M2ForceAdditiveParticleSort` callback -> bit `0x100`

`M2Faster` and `M2FasterDebug` do not directly replace the low bits; they OR optimization masks through `FUN_0081c060`, which only preserves high bits `0x2000`, `0x6000`, or `0xe000`.

The callback bodies now confirm the user-facing semantics too:

- doodad batching enabled or disabled
- particle batching enabled or disabled
- additive-particle forced-sort enabled or disabled

#### 8. The fallback `0x40` path is real and still tied to batching eligibility

`FUN_0081c0d0` sets fallback bit `0x40` only when startup bit `0x8` is not active and the host capability flags also permit it.

`FUN_00824550` still consults that same `0x40` bit inside combinable-doodad eligibility:

- batching requires global `0x20`
- doodad flag `0x10`
- either more than one shared record or fallback `0x40`
- candidate flag `0x10`
- no blocking runtime owner state at `+0x2a8`

This keeps the previously suspected `0x40` relationship grounded in real batching control flow rather than only speculative flag archaeology.

#### 9. The Win32 cache-open path is now decompiled and confirms the strict extension gate

`FUN_0081c390` is the real Win32 model cache-open path behind the earlier `Model2` string evidence.

Recovered behavior:

- copies and normalizes the incoming path to lowercase
- accepts `.mdl`, `.mdx`, and `.m2`
- rewrites `.mdl` and `.mdx` to `.m2`
- rejects missing or unsupported extensions with:
	- `Model2: Invalid file extension: %s`
- opens the normalized `.m2` path and rejects missing payloads with:
	- `Model2: File not found: %s`
- hashes the basename into the cache bucket and either reuses the existing entry or creates a new cache object
- preserves loader flags such as `0x10`, `0x40`, and the non-linking `0x8` path that had already shown up in earlier runtime notes

This is no longer just a string-xref conclusion; the Win32 decompilation now confirms the canonical `.m2` identity rule directly.

#### 10. External animation naming is fully explicit on Win32

`M2_FormatAnimFilename_04d_02d` (`0x00835a20`) is now decompiled and does exactly what the earlier runtime sampling implied:

- copies the current model basename into the destination buffer
- strips the extension
- appends `%04d-%02d.anim`

So the external animation filename contract is directly confirmed on Win32 as:

- `basename + "%04d-%02d.anim"`

#### 11. Animation-track relocation is also first-class in the model bootstrap

`FUN_00837ee0` is one of the larger animation-track relocation seams called during the main `FUN_0083cf00` model bootstrap.

Important behavior from the decompilation:

- validates and relocates multiple per-sequence pointer families in `0x28`-byte records
- handles inline and out-of-line forms, including the `-1` sentinel cases seen in native structures
- applies per-record pointer fixups against the loaded model base
- respects section records carrying flag `0x20` when walking associated animation or track payloads

This strengthens the earlier architectural boundary: animation ownership is part of the main strict model bootstrap, not a late optional post-process bolted on after the root model is already considered fully initialized.

#### 12. The `0x20` shared-record flag is now partially resolved by exact record-size matches

The follow-up Win32 decompilation pass against `FUN_00838b10`, `FUN_00839080`, and `FUN_00839270` materially improved the meaning of `0x20`.

Recovered pattern:

- each helper repeatedly performs nested pointer or track fixups only when `(*(byte *)(... + 0xc + iVar4) & 0x20) != 0`
- the helper record widths line up with real root-model track-bearing structures from the in-repo wowdev docs:
	- `0x14` -> `M2Track<T>`
	- `0x28` -> `M2Color`
	- `0x3c` -> `M2TextureTransform`
	- `0x9c` -> `M2Light`
- this means the flag is showing up on shared record families that own nested animation payloads, not on the compact renderable section path itself

Current best reading:

- `0x20` marks a shared-record class whose associated track-bearing substructures require special relocation and are excluded from the compact runtime render list
- it should not currently be named as a skin texture-unit `0x20` meaning or any other `.skin`-local material flag; the native bootstrap evidence is pointing at root-model shared blocks instead

What is still open:

- the exact user-facing label for that class in `wow-viewer` terms
- whether every `0x20`-bearing family is purely non-renderable support state or whether some participate indirectly in section/material evaluation later in the pipeline

### Win32 live deep-capture update (Apr 01, 2026)

Live x64dbg captures now confirm the expected choose/load/init/rebuild/effect chain on the Win32 `3.3.5.12340` process:

- at `0x0083cd2a`, profile index `1` was selected with active threshold path `0x40`
- at `0x0083cb60`, the formatter output was exact numbered skin ownership:
	- `interface\\glues\\models\\ui_mainmenu_northrend\\ui_mainmenu_northrend01.skin`
- at `0x0083cd32`, the post-load path reported success (`EAX=1`)
- at `0x00838490` and `0x00838561`, skin-init completion set the loaded state bit (`...01` -> `...03`)
- callback rebuild loop executed in the same chain with hits at `0x00824510` and `0x00832ea0`
- combiner path returned a live effect handle from `M2_BuildCombinerEffectName` (`0x00836dab`, non-null `EAX`)

Current boundary on that live pass:

- captures were still in UI-model context, not world-path doodad context
- world-path capture is still required for final renderer-parity decisions

### Win32 hidden option and likely-dead branch notes (Apr 01, 2026)

The Win32 pass now also recovered additional runtime-flag behavior and hidden-path seams:

- shared runtime flag word: `DAT_00d3fcf4`
	- getter `FUN_0081c0b0`
	- setter `FUN_0081c0c0`
	- high-bit OR helper `FUN_0081c060`
- callback-owned low-bit toggles are explicit in Win32:
	- `FUN_00402410` (`M2BatchDoodads`) -> bit `0x20`
	- `FUN_00402470` (`M2BatchParticles`) -> bit `0x80`
	- `FUN_004024d0` (`M2ForceAdditiveParticleSort`) -> bit `0x100`
- registration nuance in `M2_RegisterRuntimeFlags` disassembly:
	- `M2UseZFill`, `M2UseClipPlanes`, and `M2UseThreads` are registered with null callbacks
	- batching/sort/faster controls register non-null callbacks
	- implication: some controls may be startup-applied state rather than full runtime hot-toggle state unless another path re-applies them
- `M2Faster` and `M2FasterDebug` callbacks (`FUN_004021c0`, `FUN_00402210`) route through `FUN_00402100` and can drive `0x2000`, `0x6000`, and `0xe000` optimization masks
- likely dead startup fallback branch:
	- `FUN_0081c0d0` only sets fallback bit `0x40` when `(flags & 0x8) == 0`
	- `M2_RegisterRuntimeFlags` returns `uVar1 | 8`, and startup calls it directly before `FUN_0081c6e0`
	- implication: this fallback branch is likely unreachable in normal startup-driven init unless flags are injected by a different path
	- downstream note: `FUN_00824550` consults bit `0x40` in combinable-doodad gating, so an unreachable `0x40` setter changes effective runtime branch behavior
- non-primary prewarm chain discovered:
	- `M2_NormalizeModelPathAndProbeSkins` is called repeatedly from `FUN_0053c520`, `FUN_0053e810`, `FUN_0053e930`, and `FUN_0053eaa0`
	- `FUN_006e7d60` and `FUN_006e7e00` feed this chain through `FUN_0053eaa0` during player-object update flows
	- this appears to be an aggressive probe or warm-up path distinct from the strict choose/load/init runtime ownership path
- secondary portrait-only render path observed:
	- `FUN_00619580` performs dedicated portrait texture rendering using M2 cache/runtime state but a separate submission path and callback setup
	- parity work should treat portrait and world paths as related but not equivalent rendering pipelines
- strict load rejection path confirmed in Win32 cache-open routine (`FUN_0081c390`):
	- logs `Model2: Invalid file extension: %s` on unsupported extensions
	- logs `Model2: File not found: %s` on open failure
	- still normalizes `.mdl`/`.mdx` to `.m2` before open
	- includes additional cache behavior switches via loader flags (`0x10`, `0x8`, `0x40`) that can change keying/linkage/state behavior

### Win32 subsystem sweep: rendering, shaders, liquids, particles, lighting, LIT status (Apr 02, 2026)

This pass was a static/decompilation expansion in the same Win32 `3.3.5.12340` binary context, focused on subsystem ownership boundaries that should shape `wow-viewer` runtime design.

#### Rendering and shader pipeline anchors

- `FUN_00780f50` (world render init) explicitly reloads effect packs:
	- `MapObj.wfx`
	- `MapObjU.wfx`
	- `Model2.wfx`
	- `Particle.wfx`
	- `ShadowMap.wfx`
- `FUN_00876d90` loads `Shaders\Effects\%s` and routes through `ShaderEffectManager.cpp`-owned parse/bind setup.
- `FUN_00876be0` creates or reuses effect objects keyed by effect name hash.
- `FUN_00872d30` and `FUN_008728c0` bind explicit vertex/pixel shader names and combiner argument tables into effect objects.
- `M2_BuildCombinerEffectName` (`0x00836600`) and `FUN_00836c90` remain the decisive M2 combiner-family selection seam.

#### Shader-capability and feature gating anchors

- `FUN_0068a9a0` and `FUN_00684c40` enumerate and log pixel/vertex shader capability and selected targets (`pixelShaderTarget`, `vertexShaderTarget`, shader constants).
- `FUN_0078de60` (`specular`) hard-gates specular enablement on pixel shader capability (`Specular not enabled.  Requires pixel shaders.`).
- `FUN_00787780` (`MapWeather.cpp`) registers `useWeatherShaders` and weather-density controls; weather rendering is shader-gated rather than fixed-function only.

#### Liquid system anchors

- Liquid shader/material constructors are explicit and split by family:
	- `FUN_008a3e00` -> `vsLiquidProcWater%s` + `psLiquidProcWater%s`
	- `FUN_008a3f70` -> `vsLiquidWater` + `psLiquidWater`
	- `FUN_008a4070` -> `vsLiquidWaterNoSpec` + `psLiquidWaterNoSpec`
	- `FUN_008a4190` -> `vsLiquidMagma` + `psLiquidMagma`
- `FUN_008a1fa0` (`Material Bank`) and `FUN_008a28f0` (`Settings Bank`) resolve liquid type from DBC-backed banks and fall back to water when missing.
- `FUN_00793d20` builds runtime liquid instances, resolves liquid type/material/settings, and logs missing-type fallback in WMO contexts.
- `FUN_007cefd0` and `FUN_007cf790` show dedicated `MapChunkLiquid.cpp` buffer/object ownership (`CChunkBuf_Vertex`, `CChunkBuf_Index`, map chunk liquid object lifecycle).
- `FUN_0079e1a0` loads water ripple shaders (`WaterRipples` vertex/pixel) and textures; `FUN_0079d460` gates ripple emission via runtime toggle state (`DAT_00adf7f0`) and water LOD state.

#### Particle system anchors

- `FUN_00821100` is the particle emitter merge path (`ParticleBatch`) with strict compatibility checks and vertex/index capacity limits.
- `FUN_008214e0` dispatches either batched or direct particle submission depending on emitter flags; direct path logs `Particle: model=%s` and binds particle render state.
- `FUN_0081f330` initializes particle effect-handle set (`Particle`, `Particle_Unlit`, `Projected_ModMod`, `Projected_ModMod_Unlit`, `Projected_ModAdd`, `Projected_ModAdd_Unlit`).
- `FUN_00979170` allocates and manages `CParticleEmitter2_idx` pools.
- `FUN_0078e400` and `FUN_0078d860` confirm live `particleDensity` cvar registration and range checking (`0.1..1.0`).

#### Lighting anchors

- Map/world lighting DB ownership is explicit through Light-family DBC seams:
	- `FUN_008bdfc0` -> `DBFilesClient\\LightSkybox.dbc`
	- `FUN_008bdfd0` -> `DBFilesClient\\LightParams.dbc`
	- `FUN_008bdfe0` -> `DBFilesClient\\Light.dbc`
	- `FUN_008be100` -> `DBFilesClient\\LightIntBand.dbc`
	- `FUN_008be1a0` -> `DBFilesClient\\LightFloatBand.dbc`
- `FUN_0079e7c0` allocates map lighting runtime classes (`WLIGHT`, `WCACHELIGHT`) during world/map system init.
- Debug/script lighting command table is still live:
	- `FUN_004e6c60` (`AddLight`)
	- `FUN_004e6d60` (`AddCharacterLight`)
	- `FUN_004e6e60` (`AddPetLight`)
	- `FUN_004e6be0` (`ResetLights`)
	- `FUN_00960d20` (`SetLight`)
	- `FUN_00960dd0` (`GetLight`)
- `FUN_0078ded0` enforces `mapObjLightLOD` range (`0..2`).

#### LIT support status in this pass

Current Win32 evidence says:

- no discovered string/path references to standalone `.lit` or `.LIT` files
- no discovered `%s.lit` formatter seam
- only discovered `*Unlit*` usage is effect-mode naming (`Particle_Unlit`, `Projected_*_Unlit`), not a file-family loader
- discovered lighting ownership is DBC-driven (`Light*.dbc`) plus shader/effect selection, not an obvious external `*.lit` asset seam

Current classification for this binary pass:

- no positive evidence of direct standalone `.lit` asset-file loading in the recovered Win32 renderer/runtime path
- this is still an evidence-bounded statement, not a proof that no hidden path exists under all startup permutations

### Cataclysm next-build setup status: substitute `4.0.0.11927`, but static-only for now (Apr 02, 2026)

The next cross-build slot after the Win32 Wrath baseline should be the first Cataclysm-era build.

The default ladder names `4.0.6a.13623`, but the nearest actually documented Cataclysm-era native evidence already present in this repo is Win32 `4.0.0.11927`.

Current confirmed static-only evidence for `4.0.0.11927` from the existing repo notes and prompts:

- Win32 x86 `WoW.exe` build `4.0.0.11927` was previously loaded in Ghidra for terrain and engine work
- the build is documented as keeping `MD20` / `MD21` M2-family continuity with `3.3.5`-style on-disk M2 expectations
- archived Cataclysm notes explicitly mark M2 as active in this build (`InvisibleStalker.m2`) while MDX is legacy
- the engine/performance guide already recovers Cataclysm-native effect-stack evidence including:
	- `./ShaderEffectManager.cpp`
	- `%s\%s\%s.bls`
	- `Shaders\Vertex`, `Shaders\Pixel`, and related shader directories
	- `A.\M2Cache.cpp`

Current blocker for a reproducible next-build pass in this session:

- a direct filesystem search under `I:\parp` only surfaced local testdata executables for `0.5.5` and `0.6.0`
- no Cataclysm or later `WoW.exe` path is currently available in the visible environment for live x64dbg attach or fresh offline anchor recovery

What this does and does not close:

- it is enough to justify substituting `4.0.0.11927` for the next Cataclysm slot when the exact `4.0.6a.13623` client is unavailable
- it is not enough to claim Cataclysm M2 choose/load/init/effect parity with the Wrath baseline yet
- no Cataclysm `%02d.skin`, `%04d-%02d.anim`, `Combiners_*`, or runtime world-path chain has been freshly confirmed in this session

### Cataclysm `4.0.0.11927` static M2 anchor recovery (Apr 02, 2026)

With the Cataclysm binary loaded in Ghidra, the first dedicated `4.0.0.11927` M2 anchor pass is now strong enough to treat this build as a real next-step baseline for static comparison.

#### 1. Exact numbered skin and anim filename builders are still explicit

Recovered string/xref anchors:

- `0x00a2cd9c` -> `%02d.skin`
- `0x00a2cd8c` -> `%04d-%02d.anim`

Recovered functions:

- `FUN_007242d0` copies the current model basename, strips the extension, and appends `%02d.skin`
- `FUN_00724270` copies the current model basename, strips the extension, and appends `%04d-%02d.anim`

This closes the first Cataclysm-era identity question cleanly: `4.0.0.11927` still owns exact numbered skin files and exact external anim filenames through dedicated formatter seams.

#### 2. Skin choose, exact load, completion callback, and strict init are all still staged

Recovered choose/load/init chain:

- `FUN_0072a740`
	- chooses the active skin profile
	- emits `Failed to choose skin profile: %s` on failure
	- allocates the texture array at `+0x174`
	- still carries the same runtime-facing texture flag setup and section-count walk shape seen in later Wrath notes
- `FUN_0072a620`
	- calls `FUN_007242d0` to build the exact numbered `%02d.skin` path
	- opens that path through the model/file open path
	- allocates the loaded skin payload at `+0x170`
	- allocates async job state at `+0x0c`
	- installs completion callback `FUN_0072a5f0` and failure cleanup `FUN_00724250`
- `FUN_0072a5f0`
	- calls `FUN_0072a4e0`
	- then clears the async job state
- `FUN_0072a4e0`
	- validates or relocates multiple payload blocks through `FUN_007245d0`, `FUN_00724620`, `FUN_00724670`, and `FUN_007246c0`
	- emits `Corrupt skin profile data: %s` or `Failed to initialize model skin profile: %s` on failure
	- calls `FUN_00725e00` as the main active-skin materialization seam
	- sets the loaded bit with `*(uint *)(param_1 + 8) |= 2`
	- drains the callback list through `FUN_007122c0` and `FUN_007223c0`

This is no longer just a generic “Cataclysm probably still has skins” statement. The static chain confirms that early Cataclysm still preserves explicit choose -> exact `%02d.skin` load -> completion callback -> strict init -> callback rebuild ownership.

#### 3. Active section/effect materialization still routes through a dedicated effect builder

Recovered functions:

- `FUN_00725e00`
	- allocates the active copied section block at `+0x18c`
	- copies `0x30`-byte section records from the loaded skin payload
	- allocates or zeros the effect-handle array at `+0x188`
	- iterates the skin-side effect records and calls `FUN_00724320` for each one
	- keeps the same broad ownership boundary as Wrath: the skin initializer is structurally building active runtime section/effect state, not passively loading sidecar data
- `FUN_00724850`
	- refreshes those effect handles again by re-running `FUN_00724320` across the active effect list when needed

Recovered effect builder:

- `FUN_00724320`
	- references `Combiners_*` and `Diffuse_*` string families directly
	- builds effect/material names like `Model2_%s`, `Model2Displ_%s`, or explicit `M2Effect %d`
	- falls back to `Diffuse_T1Combiners_Opaque` when the requested effect is missing

This is strong evidence that `4.0.0.11927` still has an explicit M2 effect-family layer and does not flatten M2 materials down to one generic renderer mode.

#### 4. External animation loads are still explicit runtime-owned assets

Recovered external-animation loader:

- `FUN_0072b3f0`
	- walks the model-side animation block for the requested sequence
	- follows alias chaining through the referenced sequence records
	- calls `FUN_00724270` to build the exact `%04d-%02d.anim` path
	- opens the external animation file and allocates async job state for it
	- marks related sequence records with runtime-ready bits during successful setup

So early Cataclysm still treats external animation files as a first-class runtime seam, not a dead legacy path.

#### 5. Runtime option registration survives, but the default mask differs from Wrath

Recovered function:

- `FUN_00402390`
	- registers `M2UseZFill`, `M2UseClipPlanes`, `M2UseThreads`, `M2BatchDoodads`, `M2BatchParticles`, `M2ForceAdditiveParticleSort`, `M2Faster`, and `M2FasterDebug`
	- maps the same low bits previously seen in Wrath:
		- `0x1` ZFill
		- `0x2` ClipPlanes
		- `0x4` Threads
		- `0x20` doodad batching
		- `0x80` particle batching
		- `0x100` additive particle sort
	- returns those flags ORed with `0x2008`

Current reading:

- the user-facing M2 runtime option surface is still recognizably the same in early Cataclysm
- unlike the Wrath note's current `0x8` startup default, this build appears to force `0x2008`, which is a real cross-build difference worth keeping visible until the corresponding optimization-mask path is traced more completely

#### 6. Auxiliary skin probe path is still present

Recovered helper:

- `FUN_007ed840`
	- normalizes `.mdl` and `.mdx` requests to `.m2`
	- probes `%02d.skin` companions from `00` through `03`
	- returns early when one exists

This mirrors the later native probe/warm-up pattern closely enough that the Cataclysm slot should not be treated as a fundamentally different skin-discovery era without stronger contradictory evidence.

#### Current Cataclysm boundary after this pass

- static anchor recovery for `4.0.0.11927` is now real for identity, exact `%02d.skin`, exact `%04d-%02d.anim`, choose/load/init, section/effect materialization, external anim ownership, and runtime flags
- what is still missing is live runtime confirmation in x64dbg for at least one contiguous choose/load/init/effect chain and at least one world-path sample
- the attempted x64dbg continuation in this session failed when the debug session dropped while trying to recover the live module base for rebased breakpoints

#### Live open-path trace update (Apr 02, 2026)

After restarting x64dbg and reattaching to the same Win32 process, a targeted live trace sampled the Storm open wrapper path at `FUN_004609b0`:

- sampled open path: `sound\\emitters\\Emitter_Stormwind_BehindtheGate_03.wav`
- sampled open path: `Shaders\\Pixel\\ps_3_0\\Desaturate.bls`

These live samples add positive evidence that active runtime open traffic is currently hitting expected audio/shader assets in this scene.

Static loader-gate corroboration in the same binary:

- `FUN_0081c390` is a strict extension gate for the `Model2` cache path and emits:
	- `Model2: Invalid file extension: %s`
	- `Model2: File not found: %s`
- recovered compare logic in `FUN_0081c390` and `M2_NormalizeModelPathAndProbeSkins` currently matches:
	- `.m2` accepted directly
	- `.mdl`/`.mdx` normalized to `.m2`
	- no positive `.lit` branch recovered in this path

Evidence boundary note:

- this does not prove no `.lit` loader exists anywhere in the client
- it does tighten the current conclusion for the recovered `Model2` load path and observed open traffic in this runtime window

#### wow-viewer implementation implications by subsystem

1. Rendering/shader ownership should keep a first-class effect registry seam (name-keyed cache plus explicit vertex/pixel binding), not ad hoc per-renderer string dispatch.
2. Liquid runtime should stay a dedicated service with DBC-driven type->material/settings resolution and explicit shader-family routing (water, no-spec, magma, procedural water).
3. Particle runtime should preserve dual paths (direct and merged-batch) with explicit compatibility/capacity gates instead of forcing unconditional merging.
4. Lighting runtime should separate world/map lighting tables from model-material/effect lighting and keep map-object light LOD policy explicit.
5. LIT handling should remain marked as unresolved or absent-until-proven; do not invent a `*.lit` loader seam in `wow-viewer` without native-positive evidence.

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

- reproducible access to a Cataclysm-or-later Win32 client binary in the current environment; the `I:\parp` search only surfaced local `0.5.5` and `0.6.0` testdata executables
- the final user-facing name for the `0x20` shared-record class; current evidence only closes that it gates track-bearing shared blocks such as exact-size `M2TextureTransform` and `M2Light` families during relocation and compact-list exclusion
- exact semantic meaning of the propagated section dependency flag `0x40`
- the exact final draw submission path for the compact runtime section list after scene classification
- the exact unlabeled function boundaries for the recovered M2 option-registration and option-application blocks
- the exact runtime switch points for z-fill, clip-plane use, threads, and additive particle sorting
- the precise mapping from section or material flags into the native combiner/effect families
- how transparent or additive submission policy interacts with the recovered scene batching comparator and the lower state-split batch helpers
- broader world-path Win32 combiner coverage beyond the confirmed `duskwoodbed.m2` sample is still pending; choose/load/init/effect closure itself is now proven on Wrath
- Cataclysm `4.0.0.11927` still lacks fresh runtime x64dbg confirmation for rebased breakpoints and world-path choose/load/init/effect capture even though the static anchors are now recovered

## Remaining PowerPC Follow-up

The main PowerPC confirmation pass is now complete enough to support library design, but a few questions are still open.

Next PowerPC follow-up targets:

1. recover the final draw-submission path that consumes the compact runtime section list
2. determine whether the PowerPC build exposes a cleaner semantic label for the now-partially-resolved `0x20` shared-record class
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