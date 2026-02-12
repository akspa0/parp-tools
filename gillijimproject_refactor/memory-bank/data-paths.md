# Critical Data Paths - NEVER ASK WHERE DATA IS

## THE SOURCE PATH (ALWAYS)
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
