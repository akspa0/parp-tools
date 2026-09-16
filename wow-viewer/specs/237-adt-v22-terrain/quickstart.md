# Quickstart: ADT/v22 Terrain

PowerShell, run from `wow-viewer/`. `$CORPUS` is wherever the v22 files live on your machine; it is
never baked into code.

> These commands are the **planned** surface from `contracts/cli-contract.md`. Before this doc is
> trusted, re-verify it against the real argument parser once the commands are implemented.

```powershell
$CORPUS = 'D:\path\to\adt-v22-files'

# Phase 0: what is actually in the files
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr inventory --root $CORPUS --recursive `
  --json specs/237-adt-v22-terrain/evidence/phase0-inventory.json

# Phase 1: settle the ambiguous layouts (vertex order, normals, height frame, placement frame)
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr layout-probe --root $CORPUS `
  --json specs/237-adt-v22-terrain/evidence/phase1-layout-probe.json

# Look at one tile
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  adt-ahdr dump --file "$CORPUS\SomeMap_32_48.adt"

# Real-data tests (skipped when the variable is unset)
$env:WOWVIEWER_AHDR_CORPUS = $CORPUS
dotnet test tests/WowViewer.Core.Tests --filter "FullyQualifiedName~AdtAhdr"
```

## Phase 2: viewer

Start the viewer, then use **Open ADT/v22 folder…** and pick `$CORPUS`. Textures and models resolve
through the currently configured client data source; anything missing renders as a placeholder.
