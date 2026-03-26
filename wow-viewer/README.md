# wow-viewer

Initial repository skeleton for the planned production split from parp-tools.

Current first-pass project layout:

- `src/viewer/WowViewer.App`
- `src/core/WowViewer.Core`
- `src/core/WowViewer.Core.IO`
- `src/core/WowViewer.Core.Runtime`
- `src/core/WowViewer.Core.PM4`
- `src/tools-shared/WowViewer.Tools.Shared`
- `tools/converter/WowViewer.Tool.Converter`
- `tools/inspect/WowViewer.Tool.Inspect`

This scaffold is intentionally minimal. It exists to lock the repo shape, project identities, and reference graph before the real code-port work starts.

Current first real code-port slice:

- `src/core/WowViewer.Core.PM4` now contains a research-seeded PM4 model and reader layer.
- Ported first slice from `Pm4Research.Core`:
	- typed chunk models for the currently trusted PM4 chunk set
	- research document container
	- binary PM4 reader
	- exploration snapshot builder
- This is still a raw research-facing PM4 layer. It does not replace the current MdxViewer runtime reconstruction contract yet.

Current PM4 inspect slice:

- `src/core/WowViewer.Core.PM4` now also contains the first single-file PM4 analyzer and report layer.
- `tools/inspect/WowViewer.Tool.Inspect` now supports:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 audit --input <file.pm4>`
	- `pm4 audit-directory --input <directory>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>]`
- Smoke-test command that passed on Mar 25, 2026:
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`

Current PM4 runtime-contract slice:

- `src/core/WowViewer.Core.PM4` now contains the first MdxViewer-facing PM4 runtime placement contract slice.
- Landed pieces:
	- public `Pm4AxisConvention`, `Pm4CoordinateMode`, and `Pm4PlanarTransform` contracts
	- shared `Pm4CoordinateService`
	- shared planar candidate contract in `Pm4PlacementContract`
- The current single-file inspect output also records the working research note that CK24 low-16 object values may be plausible `UniqueID` candidates, but this remains unverified until correlated against real placement data.

Current validation:

- `dotnet build .\WowViewer.slnx -c Debug` passed on Mar 25, 2026 in this workspace scaffold.
