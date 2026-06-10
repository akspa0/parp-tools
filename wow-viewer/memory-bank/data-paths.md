# Data Paths — wow-viewer

## Trusted Client Staging Root (ONLY THIS)

```
I:\parp\parp-tools\output\tmp\wowarchive-clients\
```

- Use this directory for ALL game client reads.
- If a needed client build is not staged there, copy it from WoWArchive first.
- NEVER use `H:\CLIENTS\...` — those paths are untrusted and stale.
- When reporting validation results, always say which staged client path was used.

## WoWArchive Source

```
G:\WoW\WoWArchive-0.X-3.X
```

- Archive readme: `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`
- Mount entrypoint: `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
- Mount uses `rman-mount` (WinFsp), mounts read-only into `G:\WoW\WoWArchive-0.X-3.X\Mount`
- Treat the mount as a discovery/source surface only. Stage a local copy for actual work.
- Staged copies are roughly 5x faster than mounted reads.

## Test Data

```
test_data/development/World/Maps/development
```

- Split Cata ADTs (466 root + _obj0 + _tex0 files)
- PM4 files (616)
- Used by `WowViewer.Core.Tests` for unit tests

```
test_data/0.5.3/
```

- Alpha 0.5.3 reference assets (MDX, BLP, DBC)

## Output Roots

| Data | Path |
|------|------|
| Cache (listfiles, PM4 overlays) | `wow-viewer/output/cache/` |
| Datasets (Zarr stores) | `wow-viewer/output/datasets/` |
| Temp staging | `output/tmp/wowarchive-clients/` |

## Development Map Minimaps

```
test_data/minimaps/development
```
