# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## Current Focus: Spec 046 — PM4 Asset Matching
C# matching library is complete. Next: build the Python/Zarr signal-store lane so the matcher can score against durable asset corpora without requiring a staged client or _obj0.adt files.

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
No Python Zarr signal-store exists. The C# side can export segment signals and match against JSON corpora, but there's no Python tooling to:
- build Zarr stores from C# exports
- train/evaluate matchers
- validate proposal quality against known-tile ground truth

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)