# Critical Data Paths - NEVER ASK WHERE DATA IS

## Canonical Dataset Root
```
datasets/
```

- New ML dataset exports should land here, not under `output/ml-corpus`.
- Root-level HF-style parsing files now live alongside each dataset root: `metadata.jsonl`, `dataset_info.json`, and `ml_dataset_manifest.json`.

## Development Roots

### V7 Terrain Export / Sampling Root
```
test_data/original_development/World/Maps/development
```

- Use this as the terrain and ADT source for current V7 terrain-model export proof work.
- Do not silently substitute `test_data/development/World/Maps/development` as the sampling root for that path.
- If minimaps are needed and they are not present under `original_development`, use an explicit minimap root instead of broadening the terrain source.

### V7 Development Minimap-Only Root
```
test_data/development
```

- Current bounded proof uses this only for minimap lookup while keeping `original_development` as the terrain source.
- Do not describe this as the terrain sampling root for the V7 proof path.

### Development Repair / Constituent Reconstruction Root
```
test_data/development/World/Maps/development
```

This folder contains:
- **Split Cata ADTs**: 466 root ADTs + _obj0 + _tex0 files
- **PM4 files**: 616 PM4 pathfinding files
- **PM4 output**: ck_instances.csv files from PM4FacesTool

## All Fixed Paths

| Data | Path |
|------|------|
| **Source ADTs + PM4** | `test_data/development/World/Maps/development` |
| **WoWMuseum 3.3.5 ADTs** | `test_data/WoWMuseum/335-dev/World/Maps/development` |
| **Minimap tiles** | `test_data/minimaps/development` |
| **Pre-release 3.0.1.8303 client** | `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft` |
| **Cata 4.0.0.11927 client** | `H:\CLIENTS\World of Warcraft Cata beta 11927` |
| **Wrath 3.3.5.12340 client** | `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft` |
| **WMO Library** | `pm4-adt-test12/wmo_library.json` |
| **MODF Reconstruction** | `pm4-adt-test12/modf_reconstruction/` |
| **Merged ADTs** | `PM4ADTs/clean/` |
| **WDL Generated** | `PM4ADTs/wdl_generated/` |
| **Listfile** | `test_data/community-listfile-withcapitals.csv` |

## MODF Reconstruction Data
- `pm4-adt-test12/modf_reconstruction/modf_entries.csv` — 1101 MODF entries
- `pm4-adt-test12/modf_reconstruction/mwmo_names.csv` — 352 WMO names

## NEVER ASK THE USER FOR PATHS
These paths are fixed. They never change. Use them directly.

## Fixed Local Client Roots
- Use `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft` as the machine-local 3.0.1.8303 client root for harvesting and viewer exploration.
- Use `H:\CLIENTS\World of Warcraft Cata beta 11927` as the machine-local 4.0.0.11927 client root for harvesting and viewer exploration.
- Use `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft` as the machine-local 3.3.5.12340 client root for harvesting and viewer exploration.
