# Lightweight source/reporting checks only. Does not load the viewer, renderer, or client data.
param()
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot ('../' * 7))).Path
Push-Location $repo
try {
    $services = 'wow-viewer/src/viewer/WoWViewer/Workbench/Services'
    $evidence = 'wow-viewer/specs/280-ui-audit-refactor/evidence'
    Add-Type -Path (Join-Path $PSHOME 'Microsoft.CodeAnalysis.dll')
    Add-Type -Path (Join-Path $PSHOME 'Microsoft.CodeAnalysis.CSharp.dll')
    $menuPath = "$services/MainMenu/MainMenuBarService.cs"
    $menu = Get-Content $menuPath -Raw
    $syntax = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($menu).GetRoot()
    $method = @($syntax.DescendantNodes() | Where-Object {
        $_.GetType().Name -eq 'MethodDeclarationSyntax' -and $_.Identifier.ValueText -eq 'DrawContent'
    })[0]
    $roots = @($method.Body.Statements | Where-Object {
        $_.GetType().Name -eq 'IfStatementSyntax' -and $_.Condition.Expression.ToString() -eq 'ImGui.BeginMenu'
    } | ForEach-Object { $_.Condition.ArgumentList.Arguments[0].Expression.Token.ValueText })
    if (($roots -join ',') -ne 'File,View,Help') { throw "Wrong top menus: $roots" }
    $menuSources = (Get-ChildItem "$services/MainMenu/*.cs" | ForEach-Object { Get-Content $_ -Raw }) -join "`n"
    if ($menuSources -match 'ImGui\.(Begin|BeginMainMenuBar|BeginPopupModal|OpenPopup|SetNextWindow\w+)\(') {
        throw 'Menu content owns chrome or popup lifecycle'
    }
    if ($menu -match 'OpenFilePopup|Enter file path|Tab System \(069\)|Dockable Shell Panels|top_mode_') {
        throw 'Deleted UI remains'
    }
    $all = @($method.DescendantNodes() | Where-Object {
        $_.GetType().Name -eq 'InvocationExpressionSyntax' -and $_.Expression.ToString() -eq 'ImGui.MenuItem'
    } | ForEach-Object { $_.ArgumentList.Arguments[0].Expression.ToString() })
    if ($all -match '^"(Export|Import)') { throw 'Top menu still owns file export/import actions' }
    $terrain = Get-Content "$services/FileOperations/TerrainFileOperationsService.cs" -Raw
    if ([regex]::Matches($terrain, 'ExportButton\("').Count -ne 10) { throw 'Missing terrain export action' }
    if ([regex]::Matches($terrain, 'ImportButton\("').Count -ne 3) { throw 'Missing terrain import action' }
    foreach ($api in 'ReplaceTileAlphaShadowArray', 'ApplyTileImageToChunks', 'ApplyHeightmap257ToChunks', 'ReplaceTileChunksAndRebuild') {
        if (-not $terrain.Contains($api)) { throw "Missing proven apply path: $api" }
    }
    'STRUCTURE_PASS roots=File,View,Help chrome_owners=0 menu_exports_imports=0 terrain_exports=10 terrain_imports=3'

    $reporter = "using System;`nusing System.IO;`n" + (Get-Content "$services/FileOperations/FileOperationReporter.cs" -Raw)
    $harness = @'
public static class FileOperationReporterChecks
{
    public static string Run(string path)
    {
        int checks = 0;
        string status = "";
        Action<string> report = s => status = s;
        Action<bool> assert = b => { if (!b) throw new Exception("Reporter check " + checks + " failed: " + status); checks++; };
        File.WriteAllText(path, "# Reporter-only file-write check\n\nNot terrain/model export or client proof.\n");
        assert(FileOperationReporter.Run("Export", path, () => { FileOperationReporter.RequireWrittenFile(path); return 1; }, report));
        assert(status.Contains(Path.GetFullPath(path)) && !status.Contains("failed"));
        assert(!FileOperationReporter.Run("Export", path, () => 0, report) && status.Contains("No artifacts"));
        assert(!FileOperationReporter.Run("Import", path, () => -1, report) && status.Contains("failed"));
        assert(!FileOperationReporter.Run("Export", path, () => throw new IOException("write denied"), report) && status.Contains("write denied"));
        assert(!FileOperationReporter.Run("Export", path, () => { FileOperationReporter.RequireWrittenFile(path + ".missing"); return 1; }, report));
        File.WriteAllText(path, "");
        assert(!FileOperationReporter.Run("Export", path, () => { FileOperationReporter.RequireWrittenFile(path); return 1; }, report));
        File.WriteAllText(path, "# Reporter-only file-write check\n\nNot terrain/model export or client proof.\n");
        return "REPORTER_PASS checks=" + checks + " real_file=" + Path.GetFullPath(path);
    }
}
'@
    Add-Type -TypeDefinition ($reporter + "`n" + $harness)
    [WoWViewer.FileOperationReporterChecks]::Run((Join-Path $repo "$evidence/p1-reporter-artifact.md"))
}
finally { Pop-Location }
