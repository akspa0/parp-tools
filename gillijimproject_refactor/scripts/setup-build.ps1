param(
  [string]$Config = "Release",
  [string]$RepoRoot,
  [string]$MpqRoot,
  [string]$MpqLocales = "enUS,enGB,deDE,frFR,koKR,zhCN,zhTW,ruRU,esES,esMX,ptBR,itIT",
  [switch]$SkipNative,
  [switch]$EnableNative,
  [string]$StormLibPrebuilt,
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

# Try to locate CMake (PATH or VS2022 bundled copy)
function Get-CMakePath {
  $cm = Get-Command cmake -ErrorAction SilentlyContinue
  if ($cm) { return $cm.Path }
  $vsCmakeCandidates = @(
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe",
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe",
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe"
  )
  $cand = $vsCmakeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($cand) { return $cand }
  return $null
}

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

function Ensure-Repo([string]$url, [string]$path, [string]$branch = $null, [switch]$Recurse) {
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
    if ($Recurse) { & $git clone --recurse-submodules $url $path } else { & $git clone $url $path }
    if ($branch) { try { & $git -C $path checkout $branch } catch { Write-Warn "Could not checkout $branch; using default." } }
  }
}

# Ensure git submodules are initialized and up-to-date (recursive)
function Ensure-Submodules([string]$path) {
  if (Test-Path $path) {
    Write-Info "Syncing submodules: $path"
    try {
      & $git -C $path submodule sync --recursive | Write-Host
      & $git -C $path submodule update --init --recursive | Write-Host
    } catch {
      Write-Warn ("Submodule sync/update failed for {0}: {1}" -f $path, $_.Exception.Message)
    }
  }
}

# Repos to ensure
$lib = Join-Path $RepoRoot 'lib'
$next = Join-Path $RepoRoot 'next'
$nextLibs = Join-Path $next 'libs'

Ensure-Repo 'https://github.com/ladislav-zezula/StormLib' (Join-Path $lib 'StormLib')
Ensure-Repo 'https://github.com/ModernWoWTools/Warcraft.NET/' (Join-Path $lib 'Warcraft.NET')
Ensure-Repo 'https://github.com/wowdev/WoWDBDefs.git' (Join-Path $lib 'WoWDBDefs')
${wowLocalLib} = Join-Path $lib 'wow.tools.local'
Write-Info "Ensuring wow.tools.local at: ${wowLocalLib}"
Ensure-Repo 'https://github.com/Marlamin/wow.tools.local' ${wowLocalLib} -Recurse
if (Test-Path ${wowLocalLib}) {
  Write-Ok "wow.tools.local present: ${wowLocalLib}"
  Ensure-Submodules ${wowLocalLib}
} else {
  Write-Err "wow.tools.local not found at expected path: ${wowLocalLib}. Retrying clone..."
  Ensure-Repo 'https://github.com/Marlamin/wow.tools.local' ${wowLocalLib} -Recurse
  if (Test-Path ${wowLocalLib}) {
    Write-Ok "wow.tools.local cloned on retry: ${wowLocalLib}"
    Ensure-Submodules ${wowLocalLib}
  } else {
    Write-Err "Failed to obtain wow.tools.local. Please check network or permissions."
  }
}

# Back-compat: if legacy location exists under next\libs, sync submodules too
${wowLocalLegacy} = Join-Path $nextLibs 'wow.tools.local'
if (Test-Path ${wowLocalLegacy}) {
  Write-Warn "Legacy wow.tools.local detected at: ${wowLocalLegacy} (preferring ${wowLocalLib})"
  Ensure-Submodules ${wowLocalLegacy}
}

# Build/deploy native StormLib via CMake + VS2022 when possible (dotnet-first: do not hard-require toolchains)
$stormlibSrc = Join-Path $lib 'StormLib'
$stormlibSln = Join-Path $stormlibSrc 'StormLib.sln'
$buildDir = Join-Path $stormlibSrc 'build\vs2022'
$nativeOut = Join-Path $RepoRoot 'WoWRollback' | Join-Path -ChildPath 'WoWRollback.Mpq' | Join-Path -ChildPath 'runtimes\win-x64\native'
New-Item -ItemType Directory -Force -Path $nativeOut | Out-Null

# 1) Prebuilt override
if ($StormLibPrebuilt) {
  if (Test-Path $StormLibPrebuilt) {
    Copy-Item -Force $StormLibPrebuilt (Join-Path $nativeOut 'StormLib.dll')
    Write-Ok "StormLib.dll deployed from prebuilt -> $nativeOut"
  } else {
    Write-Warn "StormLibPrebuilt not found: $StormLibPrebuilt"
  }
}

# 2) If already present, nothing else to do
if (Test-Path (Join-Path $nativeOut 'StormLib.dll')) {
  Write-Info "StormLib.dll already present in runtimes; skipping native steps."
}
elseif (-not $SkipNative) {
  # Prefer CMake + VS2022 when available
  $cmake = Get-CMakePath
  if ($cmake) {
    Write-Info "Configuring StormLib with CMake (VS2022, x64)"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    & $cmake -S $stormlibSrc -B $buildDir -G "Visual Studio 17 2022" -A x64 | Write-Host
    Write-Info "Building StormLib_dll via CMake"
    & $cmake --build $buildDir --config $Config --target StormLib_dll -- /m | Write-Host
  } else {
    Write-Warn "CMake not found; skipping CMake generation."
  }

  # Try copying from likely CMake/MSBuild output folders
  $candidates = @(
    (Join-Path $buildDir "bin\StormLib_dll\x64" | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $buildDir "$Config" | Join-Path -ChildPath 'StormLib.dll'),
    (Join-Path $lib 'StormLib\bin\StormLib_dll\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\bin\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\StormLib\bin\x64' | Join-Path -ChildPath "$Config\StormLib.dll"),
    (Join-Path $lib 'StormLib\build\x64' | Join-Path -ChildPath "$Config\StormLib.dll")
  )
  $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($found) {
    Copy-Item -Force $found (Join-Path $nativeOut 'StormLib.dll')
    Write-Ok "StormLib.dll deployed -> $nativeOut"
  } elseif ($EnableNative) {
    # As a last resort, allow MSBuild path if explicitly requested
    $msbuild = Get-MSBuildPath
    if (-not $msbuild) {
      Write-Warn "MSBuild not found; cannot run fallback native build. Provide -StormLibPrebuilt or install VS Build Tools."
    } elseif (-not (Test-Path $stormlibSln)) {
      Write-Warn "StormLib solution not found: $stormlibSln"
    } else {
      Write-Info "Building StormLib (x64|$Config) via MSBuild fallback"
      & $msbuild $stormlibSln /m /p:Configuration=$Config /p:Platform=x64 | Write-Host
      $foundAfter = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
      if ($foundAfter) {
        Copy-Item -Force $foundAfter (Join-Path $nativeOut 'StormLib.dll')
        Write-Ok "StormLib.dll deployed -> $nativeOut"
      } else {
        Write-Warn "StormLib.dll not found after fallback build. Candidates: `n  - $($candidates -join "`n  - ")"
      }
    }
  } else {
    Write-Info "Native build not enabled and no DLL found. Supply -StormLibPrebuilt or install CMake/VS2022 to build."
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
