param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$AlphaRoot = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\test_data"),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "parp_out"),
    [string[]]$Maps = @("Shadowfang"),
    [string[]]$Versions = @("0.5.3"),
    [string]$DbdDir = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\lib\WoWDBDefs\definitions"),
    [string]$LkDbcDir = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\test_data\3.3.5\tree\DBFilesClient"),
    [switch]$Verbose
)

$projectPath = Join-Path $ProjectRoot "WoWRollback.Orchestrator\WoWRollback.Orchestrator.csproj"

if (-not (Test-Path $projectPath)) {
    Write-Error "WoWRollback.Orchestrator project not found at $projectPath"
    exit 1
}

if (-not (Test-Path $AlphaRoot)) {
    Write-Error "Alpha root not found: $AlphaRoot"
    exit 1
}

if (-not (Test-Path $DbdDir)) {
    Write-Warning "DBD directory not found at $DbdDir. The run may fail if definitions are required."
}

if (-not (Test-Path $LkDbcDir)) {
    Write-Warning "LK DBC directory not found at $LkDbcDir. The orchestrator will try to infer it."
}

$mapsArg = ($Maps -join ",")
$versionsArg = ($Versions -join ",")

$dotnetArgs = @(
    "--project", $projectPath,
    "--",
    "--maps", $mapsArg,
    "--versions", $versionsArg,
    "--alpha-root", $AlphaRoot,
    "--output", $OutputRoot,
    "--dbd-dir", $DbdDir
)

if ($LkDbcDir) {
    $dotnetArgs += @("--lk-dbc-dir", $LkDbcDir)
}

if ($Verbose) {
    $dotnetArgs += "--verbose"
}

Write-Host "dotnet run $($dotnetArgs -join ' ')"

$argumentList = @("run") + $dotnetArgs
$process = Start-Process -FilePath "dotnet" -ArgumentList $argumentList -NoNewWindow -PassThru -Wait

if ($process.ExitCode -ne 0) {
    Write-Error "WoWRollback orchestrator smoke run failed with exit code $($process.ExitCode)."
    exit $process.ExitCode
}

Write-Host "WoWRollback orchestrator smoke run completed successfully."
