# Quickstart: 1.12.1 Era-Aware MD20 Reader

## Preconditions

- Staged 1.12.1 client exists at `I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\`
- wow-viewer is built: `dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug`
- listfile exists at `I:\parp\parp-tools\wow-viewer\libs\wowdev\wow-listfile\listfile.txt`

## 1. Run the unit tests

```powershell
dotnet test "I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj" -c Debug --filter "FullyQualifiedName~M2Era1121ModelReaderTests"
```

Expected: **7 tests pass**.

## 2. Inspect a 1.12.1 .mdx via CLI

```powershell
cd "I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\Data"
& "I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe" m2 inspect --archive-root "." --virtual-path "creature\bear\bear.mdx"
```

Expected first line:

```
ERA: 1.12.1 (MD20 v0x100)
```

(Caveat: the retail listfile does not include 1.12.1-era paths, so a listfile-based
catalog lookup will fail with "Could not read virtual archive file". For the 048
slice the unit tests on the real fixture are the canonical validation surface;
the CLI era-tag printing is exercised in the inspect tool's M2 dispatch path on
3.3.5 (and MDLX) fixtures instead.)

## 3. Inspect a 3.3.5 .m2 via CLI (regression)

```powershell
& "I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe" m2 inspect --input <path-to-3.3.5.m2>
```

Expected first line:

```
ERA: 3.3.5 (MD20 v0x108)
```

## 4. Inspect a chunked 0.5.3 .mdx via CLI (regression)

Expected first line:

```
ERA: MDLX (chunked)
```

## 5. Try a 2.x TBC MD20 (rejected)

```powershell
# A 2.x .m2 with version 0x104 returns:
# NotSupportedException: 2.x TBC MD20 (version 0x104) is not yet supported. See spec 049.
```

## Key files

- `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121ModelReader.cs` — the era-aware reader
- `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121Constants.cs` — 1.12.1 strides and offsets
- `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121Version.cs` — supported versions
- `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121EraTag.cs` — era tag enum + display
- `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` — dispatch + `M2DispatchResult`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` — `m2 inspect` CLI
- `wow-viewer/tests/WowViewer.Core.Tests/M2Era1121ModelReaderTests.cs` — 7 unit tests
