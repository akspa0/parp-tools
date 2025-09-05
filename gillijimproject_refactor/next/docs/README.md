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

## Structure

- `Next.Core` — domain, IO readers, transform pipeline, services, and adapters
- `Next.Cli` — thin command-line interface (convert, analyze, fix-areaids)
- `Next.Tests` — unit/integration test scaffolding

## External Dependencies

- Warcraft.NET — ADT v18 model/writer
- wow.tools.local DBCD — AreaTable loading for Alpha/LK mapping

These are referenced as source projects from `next/libs/`.
