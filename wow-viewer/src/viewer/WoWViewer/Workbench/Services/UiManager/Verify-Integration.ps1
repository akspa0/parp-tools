param([switch]$LayoutOnly)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path 'wow-viewer/src/viewer/WoWViewer').Path
Add-Type -Path "$PSHOME/Microsoft.CodeAnalysis.dll"
Add-Type -Path "$PSHOME/Microsoft.CodeAnalysis.CSharp.dll"
$parse = [Microsoft.CodeAnalysis.CSharp.CSharpParseOptions]::Default.WithLanguageVersion([Microsoft.CodeAnalysis.CSharp.LanguageVersion]::Preview)
if (!$LayoutOnly) {
    $inputs = Get-Content 'wow-viewer/specs/280-ui-audit-refactor/evidence/u-shell-compile-inputs.json' -Raw | ConvertFrom-Json
    $trees = [System.Collections.Generic.List[Microsoft.CodeAnalysis.SyntaxTree]]::new()
    foreach ($file in $inputs.Items.Compile) {
        if (!(Test-Path $file.FullPath)) { continue }
        $text = [IO.File]::ReadAllText($file.FullPath)
        # Diagnostic-only in-memory exclusion of frozen legacy minimap hosting. No stubs, aliases,
        # on-disk changes or runtime proof. Original full build remains the authoritative gate.
        if ($file.FullPath -like '*MinimapStatus*') {
            $text = $text -replace '(?m)^    private (?:ShellLayoutService|ref bool _useDockspaceUi)[^\r\n]*\r?\n', ''
            $text = $text -replace '(?ms)^    internal void DrawMinimapWindow\(\).*?^    }\r?\n', ''
        }
        $trees.Add([Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($text, $parse, $file.FullPath))
    }
    $refs = [System.Collections.Generic.List[Microsoft.CodeAnalysis.MetadataReference]]::new()
    foreach ($ref in $inputs.Items.ReferencePath) { $refs.Add([Microsoft.CodeAnalysis.MetadataReference]::CreateFromFile($ref.FullPath)) }
    $options = [Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions]::new([Microsoft.CodeAnalysis.OutputKind]::ConsoleApplication).WithAllowUnsafe($true)
    $compilation = [Microsoft.CodeAnalysis.CSharp.CSharpCompilation]::Create('UiIntegrationCheck', $trees, $refs, $options)
    # The SDK's regex generator cannot load in this host (missing Workspaces). Its only effect on
    # this source set is two generated partial-method bodies; exclude exactly those diagnostics.
    $errors = @($compilation.GetDiagnostics() | Where-Object {
        $_.Severity -eq 'Error' -and !($_.Id -eq 'CS8795' -and $_.GetMessage() -match 'TileFilePattern|RootTilePattern')
    })
    $errors | ForEach-Object ToString
    Write-Output "Focused semantic check (frozen legacy minimap host excluded in memory): $($errors.Count) errors; $($trees.Count) trees"
    if ($errors.Count) { exit 1 }
}

$code = @'
using System;
using System.Numerics;
using System.Collections.Generic;
'@ + [IO.File]::ReadAllText("$root/Workbench/Services/UiManager/UiLayout.cs") + @'

internal static class UiLayoutChecks
{
    private static int _checks;
    private static void Check(bool value, string label) { _checks++; if (!value) throw new Exception(label); }
    public static int Run()
    {
        foreach (float width in new float[] { 0, 12, 320, 1024, 1279, 1280, 1599, 1600, 1919, 1920, 3840 })
        foreach (float height in new float[] { 0, 12, 54, 600, 1080, 2160 })
        foreach (bool hidden in new[] { false, true })
        foreach (bool expanded in new[] { false, true })
        {
            var a = new WoWViewer.UiLayout(new(width,height), hidden, expanded, expanded, 280, 360, true);
            var b = new WoWViewer.UiLayout(new(width,height), hidden, !expanded, !expanded, float.NaN, float.PositiveInfinity);
            Check(a[WoWViewer.UiRegion.Scene].Rect == b[WoWViewer.UiRegion.Scene].Rect, "sidebars must not inset scene");
            Check(a[WoWViewer.UiRegion.Scene].Rect.Width == width, "scene edge to edge");
            Check(a[WoWViewer.UiRegion.StatusBar].Visible, "status always visible");
            foreach (var region in a.Regions.Values)
            {
                var r = region.Rect;
                Check(float.IsFinite(r.Width) && float.IsFinite(r.Height) && r.Width >= 0 && r.Height >= 0, "finite nonnegative geometry");
                if (region.Visible) Check(r.X >= 0 && r.Y >= 0 && r.X+r.Width <= width && r.Y+r.Height <= height, "visible region contained");
            }
        }
        Check(WoWViewer.UiLayout.Defaults(1279) is (false,false,_,_), "compact defaults");
        Check(WoWViewer.UiLayout.Defaults(1280) is (false,true,_,_), "medium defaults");
        Check(WoWViewer.UiLayout.Defaults(1600) is (true,true,_,_), "wide defaults");
        Check(new WoWViewer.UiRect(0,0,10,10).Contains(0,0), "inclusive leading edge");
        Check(!new WoWViewer.UiRect(0,0,10,10).Contains(10,10), "exclusive trailing edge");
        return _checks;
    }
}
'@
# File-scoped namespace must follow all using directives, and the appended checks share it.
$code = $code.Replace('WoWViewer.UiLayout', 'UiLayout').Replace('WoWViewer.UiRegion', 'UiRegion').Replace('WoWViewer.UiRect', 'UiRect')
$tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($code, $parse)
$runtimeRefs = [System.Collections.Generic.List[Microsoft.CodeAnalysis.MetadataReference]]::new()
([AppContext]::GetData('TRUSTED_PLATFORM_ASSEMBLIES') -split [IO.Path]::PathSeparator) | ForEach-Object { $runtimeRefs.Add([Microsoft.CodeAnalysis.MetadataReference]::CreateFromFile($_)) }
$opts = [Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions]::new([Microsoft.CodeAnalysis.OutputKind]::DynamicallyLinkedLibrary)
$test = [Microsoft.CodeAnalysis.CSharp.CSharpCompilation]::Create('UiLayoutChecks', [Microsoft.CodeAnalysis.SyntaxTree[]]@($tree), $runtimeRefs, $opts)
$stream = [IO.MemoryStream]::new()
$result = $test.Emit($stream)
if (!$result.Success) { $result.Diagnostics | ForEach-Object ToString; exit 1 }
$assembly = [Reflection.Assembly]::Load($stream.ToArray())
$count = $assembly.GetType('WoWViewer.UiLayoutChecks').GetMethod('Run').Invoke($null, @())
Write-Output "Pure layout checks passed: $count assertions"
