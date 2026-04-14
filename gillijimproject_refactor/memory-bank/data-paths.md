# Critical Data Paths - NEVER ASK WHERE DATA IS

## Canonical Dataset Root
```
datasets/
```

- New ML dataset exports should land here, not under `output/ml-corpus`.
- Root-level HF-style parsing files now live alongside each dataset root: `metadata.jsonl`, `dataset_info.json`, and `ml_dataset_manifest.json`.

## WoWArchive Source And Mount Workflow
```
G:\WoW\WoWArchive-0.X-3.X
```

- Archive readme: `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`
- Mount entrypoint: `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
- Current batch content mounts the archive read-only with `rman-mount` into `G:\WoW\WoWArchive-0.X-3.X\Mount`
- Treat the mounted archive as a discovery and source surface, not the preferred high-throughput working root for export, audit, or training-prep jobs.

## Default WoWArchive Staging Root
```
output/tmp/wowarchive-clients/
```

- When the required client only exists in WoWArchive or when wider client coverage is needed, copy the required client folder here first and process the staged copy instead of streaming directly from the mount.
- Earlier user-provided performance notes say staged copies were roughly `5x` faster than direct mounted reads even before moving onto SSD; treat local staging as the default for repeated or wide reads.
- Delete staged client copies that are no longer needed after the task.
- In validation notes, say whether a result used an `H:\CLIENTS\...` root, a direct mounted archive path, or a staged local copy.

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
- If a wider client set is needed beyond these fixed local roots, source it from WoWArchive and stage the needed build into `output/tmp/wowarchive-clients/` before heavy processing.
