param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataHarvesterRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $dataHarvesterRoot ".venv"
$pyvenvCfgPath = Join-Path $venvRoot "pyvenv.cfg"

if (-not (Test-Path -LiteralPath $pyvenvCfgPath)) {
    throw "Missing pyvenv.cfg at '$pyvenvCfgPath'. Expected wow-viewer/data-harvester/.venv."
}

$cfg = @{}
foreach ($line in Get-Content -LiteralPath $pyvenvCfgPath) {
    if ($line -match "^\s*([^=]+?)\s*=\s*(.*)\s*$") {
        $cfg[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$pythonHome = $cfg["home"]
if ([string]::IsNullOrWhiteSpace($pythonHome)) {
    throw "pyvenv.cfg does not define a Python home."
}

$basePython = if ($cfg.ContainsKey("executable") -and -not [string]::IsNullOrWhiteSpace($cfg["executable"])) {
    $cfg["executable"]
} else {
    Join-Path $pythonHome "python.exe"
}

if (-not (Test-Path -LiteralPath $basePython)) {
    throw "Base Python not found at '$basePython'."
}

$sitePackages = Join-Path $venvRoot "Lib\site-packages"
$scriptsPath = Join-Path $venvRoot "Scripts"
$srcPath = Join-Path $dataHarvesterRoot "src"

if (-not (Test-Path -LiteralPath $sitePackages)) {
    throw "Missing site-packages directory at '$sitePackages'."
}

if (-not (Test-Path -LiteralPath $srcPath)) {
    throw "Missing source directory at '$srcPath'."
}

$pythonPathParts = @($sitePackages, $srcPath)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pythonPathParts += $env:PYTHONPATH
}

$env:VIRTUAL_ENV = $venvRoot
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = ($pythonPathParts -join ";")
$env:PATH = @($scriptsPath, $pythonHome, $env:PATH) -join ";"

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    & $basePython
} else {
    & $basePython @PythonArgs
}

exit $LASTEXITCODE
