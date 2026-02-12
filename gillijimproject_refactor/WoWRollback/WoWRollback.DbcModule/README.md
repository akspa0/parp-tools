# WoWRollback.DbcModule

## Overview
DBC stage utilities for the pipeline. Wraps DBC parsing and crosswalk preparation used to map Alpha→LK Area IDs.

- Extracts/reads Map/AreaTable DBCs
- Produces CSVs for crosswalks
- Provides helpers consumed by Orchestrator and CLI

## Quick Start
This is a library-only module. Add a `ProjectReference` and call via the orchestrator helpers (see `WoWRollback.Orchestrator`).

## See Also
- `../README.md` (Architecture)
- `../docs/Alpha-Conversion-Quick-Reference.md`
- `../docs/Alpha-WDT-Conversion-Spec.md`
