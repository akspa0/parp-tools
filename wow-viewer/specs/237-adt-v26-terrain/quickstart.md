# Quickstart: ADT v26 Terrain

PowerShell, run from `wow-viewer/`. `$CORPUS` is wherever the v26 files live on your machine; it is
never baked into code.

> These commands are the **planned** surface from `contracts/cli-contract.md`. Before this doc is
> trusted, re-verify it against the real argument parser once the commands are implemented.

```powershell
$CORPUS = 'test_data\v22_adts'   # git-ignored; drop the acquired files here

# Phase 0: what is actually in the files
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr inventory --root $CORPUS --recursive `
  --json specs/237-adt-v26-terrain/evidence/phase0-inventory.json

# Phase 1: settle the ambiguous layouts (vertex order, normals, height frame, placement frame)
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr layout-probe --root $CORPUS `
  --json specs/237-adt-v26-terrain/evidence/phase1-layout-probe.json

# Look at one tile
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr dump --file "$CORPUS\SomeMap_32_48.adt"

# Real-data tests (read test_data\v22_adts by default; skipped when it is empty)
dotnet test tests/WowViewer.Core.Tests --filter "FullyQualifiedName~AdtAhdr"
```

## Phase 2: viewer

Start the viewer, then use **Open ADT v26 folder…** and pick `$CORPUS`. With a local CASC install open
as the data source (Spec 238), textures and models resolve by name → FileDataID; without one, layers show as index colours
and placements as labelled markers.
