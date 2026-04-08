# wow-viewer Spell, Paperdoll, POI, And Converter Expansion Plan

## Scope

- plan the next shared-service and tool-suite slices for:
  - named spell visualization driven by `Spell.dbc` or `Spell.db2`-family data plus linked model assets
  - character profile and paperdoll composition with armor and weapon attachment
  - `WorldSafeLocs` graveyard POIs in world view and minimap
  - converter modernization and tool cutover so old format converters stop living as conflicting executables with stale assumptions
- treat this as a migration and sequencing plan, not proof that these systems already work

## Source-Of-Truth Rule

- `wow-viewer` is the canonical implementation target for new shared DBC or DB2 readers, spell or character-resolution services, POI data services, and converter ownership
- `gillijimproject_refactor/src/MdxViewer` is still the active compatibility and runtime host for the current viewer, but it should consume shared services instead of becoming the long-term owner of these seams
- old CLIs and archaeology tools in `WoWRollback`, `WoWMapConverter`, `parpToolbox`, `PM4Tool`, `DBCTool`, and related projects are reference inputs and temporary compatibility surfaces, not the long-term architecture

## Current Starting Point

### Shared seams already worth reusing

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` already proves the active DBCD plus WoWDBDefs direction for table-backed services
- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/DbcReader.cs` already gives a narrow fallback binary DBC reader for cases where DBCD-backed table loading is not available
- `wow-viewer/tools/converter/WowViewer.Tool.Converter` already owns `detect`, which should become the front door for version-aware conversion instead of adding more one-off executables
- `wow-viewer/src/core/WowViewer.Core/Files` plus current file-summary readers already provide the right place for broader version detection and family classification

### Active viewer footholds worth extracting instead of rewriting blindly

- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` already loads `CreatureDisplayInfo`, `CreatureDisplayInfoExtra`, and `ItemDisplayInfo`; this is the best current seed for character-display and gear-texture composition
- `gillijimproject_refactor/src/MdxViewer/Terrain/TaxiPathLoader.cs` already consumes DBCD-backed creature display data for taxi mounts; this proves the current viewer can already use table-driven display resolution in a real overlay workflow
- existing area-POI and taxi overlays in `MdxViewer` are the correct sibling surface for later `WorldSafeLocs` graveyard overlays

### Current gaps that should stay explicit

- there is no shared `Spell.dbc` or `Spell.db2` reader/resolver seam yet
- there is no shared `SpellVisual` or `SpellVisualKit` resolution path yet
- there is no shared paperdoll or character-composition runtime yet
- there is no shared `WorldSafeLocs` reader or POI service yet
- the converter stack is still split across `wow-viewer`, `WoWRollback`, `WoWMapConverter`, and older tool families with overlapping responsibilities

## Architecture Decisions

### 1. Shared table services first, not viewer-first widgets

- add new DBC or DB2 readers and resolvers under `wow-viewer/src/core/WowViewer.Core.IO/Dbc`
- keep higher-level resolution and composition under `wow-viewer/src/core/WowViewer.Core.Runtime`
- treat any future GUI spell browser, paperdoll, or POI layer as a consumer of shared services, not the owner of parsing or version rules

### 2. One converter front door

- converge on one version-aware conversion host under `wow-viewer/tools/converter/WowViewer.Tool.Converter`
- keep subcommands grouped by intent, not by historical executable identity
- required user flow:
  - user chooses one input file or directory
  - tool auto-detects file family and source version/profile
  - tool lists valid target output profiles
  - tool emits one explicit conversion plan before writing

### 3. Dual-surface by service, not duplicated apps

- spell inspection, paperdoll composition, POI loading, and conversion planning should all be exposed through shared first-party services
- CLI should stay first-class for batch inspection and conversion
- GUI should be a second surface over the same services for browsing, previewing, and interactive selection

### 4. Systems before UI recreation

- the older unreleased 2001 UI direction is a valid future presentation target, but only after spell browsing, character composition, and POI data services are real
- do not spend the next slices on UI skinning while spell resolution, paperdoll composition, and converter correctness are still missing

## Feature Buckets And First Anchors

### A. Spell browser and named-spell visualization

#### Goal

- browse named spells from any supported game data version
- show the resolved spell record, linked visual rows, and referenced model assets
- preview the assembled visual bundle when the underlying runtime can support it

#### First anchor files

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/`
- `wow-viewer/src/core/WowViewer.Core.Runtime/`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect`

#### First shared seams

- `SpellRecordReader` or equivalent shared table loader
- `SpellVisualResolver` that resolves spell rows to visual rows, model paths, and known effect references without pretending full native parity
- `SpellAssetBundle` contract that returns the named spell plus the linked MDX or M2 or texture or particle dependencies the current data can resolve

#### First proof target

- `wowviewer-inspect spell inspect --input <Spell.dbc|Spell.db2|archive-root>` returns named rows and linked asset paths for a fixed sample set across at least two builds
- this is a read-only proof slice, not yet a full real-time spell-cast simulator

### B. Character profile and paperdoll composition

#### Goal

- choose a character base model
- apply display or geoset or texture choices from character and item tables
- attach weapons and armor like a real paperdoll workflow

#### First anchor files

- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/`

#### First shared seams

- `CreatureDisplayInfoResolver`
- `CharacterDisplayResolver`
- `ItemDisplayResolver`
- `CharacterAppearanceProfile` contract that carries base model path, skin choices, equipment references, and resolved attachment intents

#### First proof target

- one shared composition service can build a deterministic profile for a fixed character plus equipment sample and emit the resolved model or texture dependency graph
- first runtime proof can still use a static preview model before full gameplay-authentic geoset toggling and animation parity exist

### C. `WorldSafeLocs` graveyard POIs

#### Goal

- render historical graveyard locations in world view and minimap
- keep them optional and lazy-loaded like existing area/taxi overlays

#### First anchor files

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/`
- `gillijimproject_refactor/src/MdxViewer/Terrain/`
- existing area/taxi overlay loaders in `MdxViewer`

#### First shared seams

- `WorldSafeLocsReader`
- `WorldSafeLocPoint` contract with map id, coordinates, optional name, and version/profile metadata
- `WorldPoiService` or equivalent small overlay-data service that can later serve area POIs, taxis, and graveyards through one cache policy

#### First proof target

- `wowviewer-inspect worldsafelocs inspect` or equivalent report lists locations for a fixed build
- first GUI overlay should be a thin sibling layer to existing area/taxi overlays

#### Sequencing guardrail

- shared `WorldSafeLocs` data ownership can start now in `wow-viewer`
- active `MdxViewer` overlay wiring should still wait until the current viewer render/input cleanup stops thrashing, because the overlay itself is cheap but more live marker surfaces will muddy feedback if the shell and runtime remain noisy

### D. Unified converter modernization

#### Goal

- replace the current split between `WoWRollback`, `WoWMapConverter`, one-off WMO converters, model converters, and ADT or AlphaWDT conversion executables with one explicit conversion surface
- ensure version knowledge lives in shared format services and detection, not in hand-maintained CLI folklore

#### First anchor files

- `wow-viewer/tools/converter/WowViewer.Tool.Converter`
- `wow-viewer/src/core/WowViewer.Core/Files`
- `wow-viewer/src/core/WowViewer.Core.IO/`
- old reference inputs from `gillijimproject_refactor/src/WoWMapConverter`, `gillijimproject_refactor/WoWRollback`, and related tools

#### Tool shape

- `wowviewer-convert detect`
- `wowviewer-convert plan`
- `wowviewer-convert model`
- `wowviewer-convert wmo`
- `wowviewer-convert terrain`

#### Supported user flow

1. inspect input and detect family plus source version
2. list target profiles that the shared writers can actually support
3. show required companion files and expected outputs
4. run the conversion with an explicit report of assumptions and failures

#### First proof target

- one real end-to-end converter family should prove the pattern before broader merger
- best first candidate is the currently open converter mess around WMO version conversion and terrain-family conversion planning, because that directly exercises current version detection and shared I/O ownership

## Tool Classification Direction

### Rebuild as first-class `wow-viewer` surfaces

- `WowViewer.Tool.Converter`
- `WowViewer.Tool.Inspect`
- future `WowViewer.App` spell browser, paperdoll viewer, and POI layers

### Fold into shared services, then consume from CLI and GUI

- DBC-backed display resolution currently scattered across `MdxViewer`, `WoWRollback`, and legacy DBCTool variants
- historical WMO and model conversion logic in `WoWMapConverter` and `WoWRollback`
- any future spell visual resolution or paperdoll composition work

### Keep as reference/archeology unless a narrow algorithm is needed

- `parpToolbox`
- `PM4Tool`
- old `DBCTool` executables
- one-off inspection/export helpers whose only value is format archaeology or previous experiments

## Recommended Phase Order

### Phase 0 - Shared DBC or DB2 service expansion

- extend shared table loading beyond `AreaTable` and `Map`
- establish one reusable pattern for archive-backed and extracted-table loading plus version-profile routing
- first new families to add:
  - `WorldSafeLocs`
  - `CreatureDisplayInfo`
  - `CreatureDisplayInfoExtra`
  - `ItemDisplayInfo`
  - `Spell`

Exit condition:

- `wow-viewer` can resolve these tables through shared services without depending on viewer-local parser ownership

### Phase 1 - Character display and POI read-only services

- extract the existing character-display logic seed from `ReplaceableTextureResolver`
- add `WorldSafeLocs` inspect/report capability
- keep this slice read-only and report-first

Exit condition:

- shared services can produce deterministic resolved character-display profiles and graveyard POI datasets for fixed sample assets/builds

### Phase 2 - Spell inspect and spell-asset bundle browsing

- add the first spell-table reader and spell-asset resolution surface
- ship it first as inspect/report plus optional JSON export

Exit condition:

- named spell browsing across at least two data versions is real, even if final runtime effect playback is still partial

### Phase 3 - Unified converter planning surface

- expand `detect` into `plan`
- start converging old WMO/model/terrain conversion entrypoints into one version-aware surface

Exit condition:

- one converter command can tell the user what the input is, what targets are valid, and what companion files are needed before any write happens

### Phase 4 - GUI consumers

- add spell browser panel
- add paperdoll/profile panel
- add graveyard overlay consumer
- keep GUI thin over the shared services landed above

Exit condition:

- the viewer can browse these systems without owning their parsing or version rules

### Phase 5 - Higher-fidelity runtime parity and UI theming

- spell-cast playback fidelity improvements
- richer character geoset or attachment parity
- optional old-2001-style UI presentation experiments

Exit condition:

- only start this phase after the underlying services and preview workflows are already proven

## First Three Vertical Slices

### Slice 1 - `WorldSafeLocs` shared reader plus inspect report

- smallest high-value proof because it is self-contained, low-risk, and aligns with an already-requested visible feature
- add:
  - shared `WorldSafeLocs` table reader in `wow-viewer`
  - inspect/report command
  - tests against fixed extracted/archive-backed table inputs
- do not claim GUI overlay signoff yet

### Slice 2 - character-display resolver extraction from `ReplaceableTextureResolver`

- move `CreatureDisplayInfo`, `CreatureDisplayInfoExtra`, and `ItemDisplayInfo` resolution into shared `wow-viewer` services
- first output is a resolved character or NPC display profile report, not a full live dress-up UI

### Slice 3 - spell inspect plus linked asset bundle report

- add the first `Spell` shared table seam and spell-to-asset bundle resolution
- expose it through `WowViewer.Tool.Inspect`
- keep runtime playback intentionally out of scope for the first slice

## What Should Deliberately Wait

- full native-authentic spell runtime playback
- exact retail paperdoll parity across every client branch
- UI skinning to imitate the 2001 unreleased interface
- porting every old executable name into `wow-viewer`
- claiming that model/WMO/terrain conversion is solved before one shared plan-or-convert surface proves correct detection and target validation

## Validation Rules

- every new shared DBC or DB2 seam needs real-data proof against fixed build data, not only synthetic rows
- every converter slice must expose explicit detection and target-profile reporting before it writes outputs
- GUI proof is not a substitute for shared-service proof
- `MdxViewer` compile validation is only compatibility validation when it consumes a new shared seam; it is not the proof that `wow-viewer` owns the logic correctly

## Failure Modes To Avoid

- building a spell browser panel before there is any stable spell-resolution service
- treating `ReplaceableTextureResolver` as the final architecture instead of the extraction seed
- adding a graveyard overlay directly in `MdxViewer` without first owning `WorldSafeLocs` in shared services
- merging old converters by copy-pasting executables instead of converging behavior into one detection-aware converter host
- spending time on historical UI imitation before the underlying systems can actually resolve spells, characters, or graveyard data

## Bottom Line

- the fastest credible path is not a giant viewer rewrite
- the right first moves are shared DBC or DB2 seams, then read-only inspect/report surfaces, then thin GUI consumers, then richer runtime parity
- `WorldSafeLocs`, character-display resolution, and spell-asset bundle inspection are the three clearest first vertical slices
- converter unification should proceed through one `WowViewer.Tool.Converter` planning and detection surface, not by preserving every historical conversion executable as a permanent product