param(
  [string]$Config = "Release",
  [string]$RepoRoot,
  [string]$MpqRoot,
  [string]$MpqLocales = "enUS,enGB,deDE,frFR,koKR,zhCN,zhTW,ruRU,esES,esMX,ptBR,itIT",
  [switch]$SkipNative,
  [switch]$RestoreOnly,
  [switch]$NoPrompt,
  [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
$PSStyle.OutputRendering = 'PlainText'

function Write-Info([string]$msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn([string]$msg){ Write-Warning $msg }
function Write-Ok([string]$msg){ Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-Err([string]$msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red }

if (-not $RepoRoot) {
  $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
}

Write-Info "RepoRoot = $RepoRoot"

function Ensure-Exe([string]$exe) {
  $cmd = Get-Command $exe -ErrorAction SilentlyContinue
  if (-not $cmd) { throw "Required tool not found on PATH: $exe" }
  return $cmd.Path
}

$git = Ensure-Exe 'git'
$dotnet = Ensure-Exe 'dotnet'

# vswhere (for MSBuild)
function Get-MSBuildPath {
  $vswhereCandidates = @(
    "$Env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe",
    "$Env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  )
  $vswhere = $vswhereCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($vswhere) {
    $msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find 'MSBuild\Current\Bin\MSBuild.exe' 2>$null
    if ($msbuild) { return $msbuild }
  }
  # fallback to PATH
  $msbuildCmd = Get-Command msbuild.exe -ErrorAction SilentlyContinue
  if ($msbuildCmd) { return $msbuildCmd.Path }
  return $null
}

function Ensure-Repo([string]$url, [string]$path, [string]$branch = $null) {
  if (Test-Path $path) {
    Write-Info "Updating repo: $path"
    & $git -C $path fetch --all --prune
    if ($branch) {
      try { & $git -C $path checkout $branch } catch { Write-Warn "Could not checkout $branch; staying on current branch." }
    }
    & $git -C $path pull --ff-only
  } else {
    Write-Info "Cloning $url -> $path"
    New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
    & $git clone $url $path
    if ($branch) { try { & $git -C $path checkout $branch } catch { Write-Warn "Could not checkout $branch; using default." } }
  }
}

# Repos to ensure
$lib = Join-Path $RepoRoot 'lib'
$nextLibs = Join-Path $RepoRoot 'next' | Join-Path -ChildPath 'libs'

Ensure-Repo 'https://github.com/ladislav-zezula/StormLib' (Join-Path $lib 'StormLib')
Ensure-Repo 'https://github.com/ModernWoWTools/Warcraft.NET/' (Join-Path $lib 'Warcraft.NET')
Ensure-Repo 'https://github.com/wowdev/WoWDBDefs.git' (Join-Path $lib 'WoWDBDefs')
Ensure-Repo 'https://github.com/Marlamin/wow.tools.local' (Join-Path $nextLibs 'wow.tools.local')

# Build native StormLib
$stormlibSln = Join-Path $lib 'StormLib' | Join-Path -ChildPath 'StormLib.sln'
$nativeOut = Join-Path $RepoRoot 'WoWRollback' | Join-Path -ChildPath 'WoWRollback.Mpq' | Join-Path -ChildPath 'runtimes\win-x64\native'
New-Item -ItemType Directory -Force -Path $nativeOut | Out-Null

if (-not $SkipNative) {
  $msbuild = Get-MSBuildPath
  if (-not $msbuild) { throw 'MSBuild not found. Install Visual Studio Build Tools or ensure msbuild.exe is on PATH.' }
  if (-not (Test-Path $stormlibSln)) { throw "StormLib solution not found: $stormlibSln" }

  Write-Info "Building StormLib (x64|$Config) via $msbuild"
  & $msbuild $stormlibSln /m /p:Configuration=$Config /p:Platform=x64 | Write-Host

  # Candidate outputs
  $candidates = @(
    (Join-Path $lib 'StormLib\bin\StormLib_dll\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\bin\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\StormLib\bin\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\build\x64' | Join-Path -ChildPath "$Config\StormLib.dll")
  )

  $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $found) { Write-Warn "StormLib.dll not found after build. Candidates: `n  - $($candidates -join "`n  - ")" }
  else {
    Copy-Item -Force $found (Join-Path $nativeOut 'StormLib.dll')
    Write-Ok "StormLib.dll deployed -> $nativeOut"
  }
}

# dotnet restore/build
$solution = Join-Path $RepoRoot 'WoWRollback\WoWRollback.sln'
Write-Info "dotnet restore $solution"
& $dotnet restore $solution | Write-Host
if (-not $RestoreOnly) {
  Write-Info "dotnet build $solution -c $Config"
  & $dotnet build $solution -c $Config -m | Write-Host
}

# Optional smoke run
if ($MpqRoot) {
  $orchestrator = Join-Path $RepoRoot 'WoWRollback\WoWRollback.Orchestrator'
  $assets = Join-Path $RepoRoot 'WoWRollback\WoWRollback.Viewer\assets2d'
  Write-Info "Running smoke test with MPQ root: $MpqRoot"
  & $dotnet run --project $orchestrator -- --maps development --versions 3.3.5 --alpha-root ..\test_data --mpq-root "$MpqRoot" --mpq-locales "$MpqLocales" --viewer-label smoke --serve --viewer-assets "$assets" | Write-Host
}

Write-Ok "Setup/build completed."
