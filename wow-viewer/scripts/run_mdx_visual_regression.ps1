param(
    [switch]$UpdateBaselines,
    [string]$Configuration = "Debug",
    [string]$ActualRoot = "I:\parp\parp-tools\output\build-validation\mdx-gpu-visual-regression\actual",
    [string]$DiffRoot = "I:\parp\parp-tools\output\build-validation\mdx-gpu-visual-regression\diff"
)

$manifestPath = "I:\parp\parp-tools\wow-viewer\testdata\visual\mdx-gpu-regression.manifest.json"
$dllPath = "I:\parp\parp-tools\wow-viewer\src\viewer\WowViewer.App\bin\$Configuration\net10.0\WowViewer.App.dll"

if (-not (Test-Path $dllPath)) {
    Write-Error "Built app not found at $dllPath. Run 'dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c $Configuration' first."
    exit 1
}

$arguments = @(
    $dllPath,
    "mdx-visual-regression",
    "--manifest", $manifestPath,
    "--write-actual-root", $ActualRoot,
    "--write-diff-root", $DiffRoot
)

if ($UpdateBaselines) {
    $arguments += "--update-baselines"
}

dotnet @arguments
exit $LASTEXITCODE
