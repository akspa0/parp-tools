# GillijimProject Next

Docs-first scaffolding for the Next refactor. This solution lives entirely under `next/` and does not modify the legacy port.

## Quickstart

- Build the solution:
  
  ```bash
  dotnet build next/gillijimproject-next.sln
  ```

- Run the CLI help:
  
  ```bash
  dotnet run --project next/src/GillijimProject.Next.Cli -- --help
  ```

- WDL → OBJ/GLB export:
  
  ```bash
  # OBJ (normalized to world XY by default)
  dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-obj --in C:/data/Azeroth.wdl --out-root ./out

  # GLB (local SharpGLTF, same mapping as OBJ)
  dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-glb --in C:/data/Azeroth.wdl --out-root ./out

  # Options
  #   --height-scale <double>   # scale heights only (Z up)
  #   --no-normalize-world      # disable XY normalization (default is ON)
  #   --no-skip-holes           # include faces in MAHO-masked cells
  ```

## Structure

- `Next.Core` — domain, IO readers, transform pipeline, services, and adapters
- `Next.Cli` — thin command-line interface (convert, analyze, fix-areaids)
- `Next.Tests` — unit/integration test scaffolding

## External Dependencies

- Warcraft.NET — ADT v18 model/writer
- wow.tools.local DBCD — AreaTable loading for Alpha/LK mapping
- SharpGLTF (local source, Core + Toolkit) — GLB writer

These are referenced as source projects from `next/libs/`.
