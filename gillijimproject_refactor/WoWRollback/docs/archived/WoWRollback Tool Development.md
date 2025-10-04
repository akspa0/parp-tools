# Cascade Chat Conversation

  Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

@[i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\WoWRollback\docs\plan.md] let's build this tool!

*Viewed [projectbrief.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/memory-bank/projectbrief.md) *

*Viewed [productContext.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/memory-bank/productContext.md) *

*Viewed [systemPatterns.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/memory-bank/systemPatterns.md) *

*Viewed [activeContext.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/memory-bank/activeContext.md) *

*Viewed [plan.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/WoWRollback/docs/plan.md) *

*Updated todo list*

*Updated todo list*

### Planner Response

# Mode: PLAN

# Summary
Designing the `WoWRollback` suite as an independent pipeline that consumes existing Alpha exports, applies user-configurable placement-ID filters, and writes rollback ADTs plus reports.

# Proposed Plan
1. **Structure Projects**  
   - Create `WoWRollback/` with `Core/` (library) and `Cli/` (console).  
   - Add shared docs (`docs/usage.md`, config samples).  
   - Target `net9.0`, nullable enabled, reuse common utility packages.

2. **Ingest & Catalog Inputs**  
   - Implement readers for `id_ranges_by_map.csv` and converted ADTs under `output_wdt-export/`.  
   - Define range catalog abstraction (per-map placement spans, tile metadata) decoupled from Alpha tool.

3. **Configuration & Filtering Engine**  
   - Specify YAML/JSON schema supporting keep/drop modes, open-ended ranges, presets.  
   - Build core filtering service that loads config, evaluates ranges, and produces per-tile removal plans.

4. **ADT Mutation Pipeline**  
   - Implement safe read/clone writer that strips selected placement `uniqueId` entries while copying untouched data.  
   - Support dry-run (report-only) and rollback output (e.g., `*.rollback.adt`), preseving area IDs.

5. **CLI & Reporting**  
   - Expose `rollback-cli` with commands `dry-run` and `apply`, plus `--map`, `--config`, `--input-dir`, `--output-dir`.  
   - Emit summary CSV/JSON (removed counts, range coverage) and console logs.

6. **Validation & Tests**  
   - Add integration tests using real sample ADTs (skip-if-missing).  
   - Provide diff utility comparing original vs rollback placement counts.  
   - Document workflow in [WoWRollback/docs/plan.md](cci:7://file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/WoWRollback/docs/plan.md:0:0-0:0) updates.

# Status
- **Review existing repository structure and WoWRollback plan to clarify requirements and scope** ✅ Completed
- **Design project layout, core components, and configuration format for WoWRollback** ⏳ In Progress
- **Outline implementation tasks for CLI, core library, and integration points** ⏳ Pending
- **Define validation and rollout strategy (testing, dry-run flow, documentation)** ⏳ Pending

Let me know when to switch to act mode and start implementing.

### User Input

We can scrape together 90% of the tooling from @[AlphaWDTAnalysisTool] to read Alpha WDT's in, gather their UniqueIDs, and then - present the ranges that we detect as 'keep' or 'remove', then continue the processing of the data into WoTLK ADTs. 

### Planner Response

# Mode: PLAN

# Summary
Acknowledged that `WoWRollback` can leverage ~90% of the existing `AlphaWDTAnalysisTool` code to ingest Alpha WDTs, collect placement `uniqueId` ranges, classify them as keep/remove, and continue through to LK ADT generation. We’ll plan to factor that shared logic into reusable components under `WoWRollback/Core` while preserving separation from the original tool.

# Proposed Plan
1. **Structure Projects**  
   - `WoWRollback/Core/` for reusable services (parsers, range catalog, filter engine).  
   - `WoWRollback/Cli/` for the console app targeting `net9.0`.  
   - Shared docs: `WoWRollback/docs/usage.md`, config samples, integration notes.

2. **Ingest & Catalog Inputs** _(in progress)_  
   - Reuse `AlphaWDTAnalysisTool` readers for Alpha WDT → placement extraction and LK ADT writing, refactoring into shareable classes inside `WoWRollback/Core`.  
   - Extend range catalog to emit per-map archives (`id_ranges_by_map_<map>.csv`) while remaining decoupled from the original exporter.

3. **Configuration & Filtering Engine**  
   - Define YAML/JSON schema supporting keep/drop ranges, open-ended bounds, presets.  
   - Build range-evaluation service that applies user config to extracted placement data.

4. **ADT Mutation Pipeline**  
   - Wrap the existing LK writer logic to clone ADTs and remove filtered placements without touching area IDs.  
   - Support dry-run reporting and actual rollback output (`*.rollback.adt`).

5. **CLI & Reporting**  
   - Provide `rollback-cli` commands (`dry-run`, `apply`) with flags: `--map`, `--config`, `--input-dir`, `--output-dir`.  
   - Emit post-run summaries (CSV/JSON) detailing kept/removed ranges and counts.

6. **Validation & Tests**  
   - Integration tests against real sample ADTs (skip-if-missing fixtures).  
   - Diff utility comparing original vs rollback placement stats.  
  - Update documentation once flow is verified.

# Status
- **Review existing repository structure and WoWRollback plan to clarify requirements and scope** ✅ Completed
- **Design project layout, core components, and configuration format for WoWRollback** ⏳ In Progress
- **Outline implementation tasks for CLI, core library, and integration points** ⏳ Pending
- **Define validation and rollout strategy (testing, dry-run flow, documentation)** ⏳ Pending

Let me know when you’d like me to move to act mode and start refactoring and scaffolding the projects.

### User Input

we already scan for UniqueID's and their clustered ranges, but we don't export per-map lists, so we need to handle that, but then - the hard part, we can do two functions - one, collect the IDs and ranges of IDs. two, actually process the ADT outputs with the selected ranges to keep or remove - we should be able to null any strings that we don't want for assets that match, OR replace them with a placeholder object, so long as the length of the path doesn't exceed the available space within each chunk, since we don't want to have to fully adjust/rewrite the ADT chunks/headers due to offsets changing drastically.

*Updated todo list*

### Planner Response

# Mode: PLAN

# Summary
Captured new requirements: extend existing Alpha ingestion to emit per-map unique-ID range catalogs, and build two processing paths—(1) collect/display keep/remove ranges; (2) mutate LK ADTs by zeroing placement names or swapping to fixed-length placeholders without altering chunk sizes.

# Proposed Plan
1. **Per-Map Range Catalog** *(in progress)*  
   - Extract placement ranges from Alpha WDT/ADT exports (`AlphaWdtAnalyzer.Core`) and emit `id_ranges_by_map_<map>.csv` under `WoWRollback/`.  
   - Include metadata (map, tile, placement type) so the CLI can present keep/remove decisions.

2. **Configuration & Range Selection** *(pending)*  
   - Define YAML/JSON schema describing keep/drop ranges per map, plus defaults.  
   - Implement loader that merges user selections with the exported per-map catalogs.

3. **ADT Mutation Pipeline** *(pending)*  
   - Reuse LK writer logic to produce copies of converted ADTs.  
   - For removals, either null path strings or swap in a constant placeholder of equal length to avoid offset shifts.  
   - Support dry-run reporting and actual rewrite outputs (`*.rollback.adt`).

4. **CLI Commands & Reporting** *(pending)*  
   - `rollback-cli ranges` to display detected ranges and selection effects.  
   - `rollback-cli apply` to process ADTs using chosen ranges.  
   - Emit summaries (CSV/JSON) of kept/removed IDs for audit.

5. **Validation & Safety** *(pending)*  
   - Tests ensuring chunk sizes remain unchanged.  
   - Fixture-based integration runs (skip-if-missing real data).  
   - Document workflow and placeholder strategy in `WoWRollback/docs/`.

Let me know when to switch to act mode and start refactoring/implementing.

### User Input

let's start building. 

### Planner Response

# Mode: PLAN

I’m still in plan mode. To begin implementing, please enter `ACT`.

# Summary
Ready to proceed with the WoWRollback build once you switch me to act mode.

# Proposed Plan
1. **Per-Map Range Catalog**  
   - Extend existing Alpha ingestion to emit `id_ranges_by_map_<map>.csv` under `WoWRollback/`.  
   - Include map, tile, and placement type metadata for keep/remove decisions.

2. **Configuration & Range Selection**  
   - Define YAML/JSON schema describing keep/drop ranges per map and defaults.  
   - Implement loader merging user selections with exported catalogs.

3. **ADT Mutation Pipeline**  
   - Reuse LK writer logic to clone ADTs while removing targeted placements.  
   - Replace unwanted asset strings with equal-length placeholders to avoid header/offset changes.  
   - Support dry-run reporting and actual rollback outputs (`*.rollback.adt`).

4. **CLI Commands & Reporting**  
   - Provide `rollback-cli ranges` (inspect) and `rollback-cli apply` (mutate) with flags: `--map`, `--config`, `--input-dir`, `--output-dir`.  
   - Emit summaries (CSV/JSON) of kept/removed IDs for audit.

5. **Validation & Safety**  
   - Tests ensuring chunk sizes stay constant and placeholders fit.  
   - Fixture-based integration runs (skip-if-missing real data).  
   - Document workflow and placeholder strategy in `WoWRollback/docs/`.

# Status
- **Design project layout, core components, and configuration format for WoWRollback** ✅ Completed
- **Extend ingestion to emit per-map UniqueID range catalogs and reuse AlphaWDTAnalysisTool readers** ⏳ In Progress
- **Outline implementation tasks for configuration, CLI, and ADT processing pipeline** ⏳ Pending
- **Define validation and rollout strategy (dry-run, placeholder injection safety, documentation)** ⏳ Pending

### User Input

we should probably employ some automated testing, this shouldn't be too hard to do, it's simply nulling out model/wmo entries based on their uniqueIDs, which will literally show us the past hidden in the present data. 

### Planner Response

# Mode: PLAN

# Summary
Acknowledged requirement to add automated tests that confirm placement string nulling/placeholder substitution works correctly and reveals historical content without altering chunk sizes.

# Proposed Plan
1. **Per-Map Range Catalog**  
   - Emit `id_ranges_by_map_<map>.csv` with map/tile/type metadata using shared Alpha readers.

2. **Configuration & Range Selection**  
   - Define YAML/JSON schema for keep/drop ranges; build loader merging user choices with catalogs.

3. **ADT Mutation Pipeline**  
   - Clone LK ADTs, remove or placeholder-rewrite targeted placements while preserving chunk sizes.

4. **CLI Commands & Reporting**  
   - Implement `rollback-cli ranges` (inspect) and `rollback-cli apply` (mutate) with required flags and summary outputs.

5. **Automated Testing & Validation**  
   - Add tests that run the pipeline on real fixtures (skip-if-missing) verifying string nulling/placeholder behavior, unchanged chunk lengths, and accurate reporting. Update docs with testing instructions.

# Status
- **Design project layout, core components, and configuration format for WoWRollback** ✅ Completed
- **Extend ingestion to emit per-map UniqueID range catalogs and reuse AlphaWDTAnalysisTool readers** ⏳ In Progress
- **Outline implementation tasks for configuration, CLI, and ADT processing pipeline** ⏳ Pending
- **Define validation and rollout strategy (dry-run, placeholder injection safety, documentation)** ⏳ Pending
- **Automated testing for range removal and placeholder safety** ⏳ Pending

### User Input

continue. ACT

*Updated todo list*

*Listed directory [gillijimproject_refactor](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor) *

*Listed directory [WoWRollback](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/WoWRollback) *

*User accepted the command `dotnet new sln -n WoWRollback`*

*User accepted the command `mkdir Core`*

### User Input

@[i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\WoWRollback\WoWRollback.sln] 

*Listed directory [WoWRollback](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/WoWRollback) *

*Viewed [WoWRollback.sln](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/WoWRollback/WoWRollback.sln) *

*User accepted the command `dotnet new classlib -n WoWRollback.Core`*