# Quickstart: CASC Data Source (planned surface)

PowerShell, from `wow-viewer/`. **This is the planned command surface.** Diff it against the real
argument parser once implemented, before trusting it.

```powershell
# After cloning or pulling: fetch the vendored CASC library
git submodule update --init libs/wowdev/TACTSharp

$INSTALL = 'D:\path\to\World of Warcraft'   # your install; never hardcoded
$CACHE   = 'D:\path\to\casc-cache'

# Local install
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- casc products --install $INSTALL
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- casc identity --install $INSTALL --product wow
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  casc read --install $INSTALL --product wow --id 1349477 --out output\casc\map.db2

# Remote
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- casc builds --product wow --region us
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  casc read --remote --product wow --region us --cache $CACHE --path "world\maps\azeroth\azeroth.wdt" --out output\casc\azeroth.wdt

# Verification against an independent extraction (SC-001)
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -- `
  casc verify --install $INSTALL --product wow --sample 1000 --reference D:\path\to\reference-extract `
  --json specs\238-casc-data-source\evidence\phase1-verify.json

# Real-build tests
$env:WOWVIEWER_CASC_LOCAL = $INSTALL
dotnet test tests/WowViewer.Core.Tests --filter "FullyQualifiedName~Casc"
```
