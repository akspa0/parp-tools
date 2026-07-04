# Active Context

Last updated: 2026-07-04
Reference repo only. Do not treat this file as active design owner. Older June detail moved to `memory-bank/archive/2026-07-04-pre-wow-viewer-reset.md`.

## Role

- `gillijimproject_refactor` = read-only reference for working behavior, data paths, and old plans.
- New code belongs in `wow-viewer`.
- If task is not explicit hotfix or archaeology, route out of this tree.

## What still matters here

- `data-paths.md` = staged client rules and fixed paths.
- `planning/` = old extraction plans, reference only.
- `src/MdxViewer/memory-bank/` = current legacy viewer compatibility truth.
- `wow-viewer/memory-bank/` = active implementation truth.

## Current repo truth

- Old V19/V20/V21 terrain-model notes are historical only.
- Old June PM4 experiments are historical only.
- Current ML, UI, and runtime ownership lives in `wow-viewer`.
- Keep this file small. Archive stale session detail instead of stacking more logs.

## Boundaries

- No new feature work here.
- No `H:\CLIENTS`.
- No parser rewrites.
