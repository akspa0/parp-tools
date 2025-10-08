using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWRollback.Core.Logging;
using WoWRollback.DbcModule;

namespace WoWRollback.Orchestrator;

internal sealed class DbcStageRunner
{
    public DbcStageResult Run(SessionContext session)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        var options = session.Options;
        var dbdDir = options.DbdDirectory ?? throw new InvalidOperationException("DBD directory is required for DBC stage.");
        var locale = "enUS";

        var lkDbcDir = ResolveLkDbcDirectory(options);
        if (lkDbcDir is null)
        {
            throw new InvalidOperationException("Lich King DBC directory could not be resolved. Provide --lk-dbc-dir or ensure default path exists.");
        }

        // Create DbcOrchestrator for library API calls
        var orchestrator = new DbcOrchestrator(dbdDir, locale);

        var perVersion = new List<DbcVersionResult>();
        var overallSuccess = true;

        foreach (var version in options.Versions)
        {
            var alias = DeriveAlias(version);
            var build = ResolveBuildIdentifier(alias);
            var sourceDir = ResolveSourceDbcDirectory(options.AlphaRoot, version);

            var dbcVersionDir = Path.Combine(session.Paths.DbcDir, version);
            var crosswalkVersionDir = Path.Combine(session.Paths.CrosswalkDir, version);
            
            // Match DBCTool expected structure
            var sharedDbcDir = Path.Combine(dbcVersionDir, "raw");
            var sharedCrosswalkAliasRoot = crosswalkVersionDir;
            var sharedCrosswalkVersionRoot = crosswalkVersionDir;
            var sharedCrosswalkCompareRoot = Path.Combine(sharedCrosswalkVersionRoot, "compare");
            var sharedCrosswalkV2Dir = Path.Combine(sharedCrosswalkCompareRoot, "v2");

            if (string.IsNullOrWhiteSpace(sourceDir) || !Directory.Exists(sourceDir))
            {
                overallSuccess = false;
                perVersion.Add(new DbcVersionResult(
                    version,
                    alias,
                    sourceDir ?? string.Empty,
                    DbcOutputDirectory: sharedDbcDir,
                    CrosswalkOutputDirectory: sharedCrosswalkV2Dir,
                    DumpExitCode: -1,
                    CompareExitCode: -1,
                    Error: $"Missing source DBC directory for version '{version}' (expected {sourceDir})."));
                continue;
            }

            var inputs = new List<(string build, string dir)>
            {
                (alias, sourceDir),
                ("3.3.5", lkDbcDir)
            };

            Directory.CreateDirectory(sharedDbcDir);
            Directory.CreateDirectory(sharedCrosswalkAliasRoot);
            Directory.CreateDirectory(sharedCrosswalkVersionRoot);

            var hasCachedDbc = Directory.Exists(sharedDbcDir) && Directory.EnumerateFileSystemEntries(sharedDbcDir).GetEnumerator().MoveNext();
            var mapsJsonPath = Path.Combine(sharedCrosswalkAliasRoot, "maps.json");
            var hasCachedCrosswalk = Directory.Exists(sharedCrosswalkV2Dir) && File.Exists(mapsJsonPath);

            bool success = true;
            string? error = null;

            if (hasCachedDbc && hasCachedCrosswalk)
            {
                // Use cached results
            }
            else
            {
                // Step 1: Dump ALL DBCs to JSON (comprehensive exploratory dump)
                var jsonDumpDir = Path.Combine(dbcVersionDir, "json");
                var dumpAllResult = orchestrator.DumpAllDbcs(alias, sourceDir, jsonDumpDir);

                if (!dumpAllResult.Success)
                {
                    success = false;
                    error = dumpAllResult.ErrorMessage ?? "Comprehensive DBC dump failed";
                }
                else
                {
                    ConsoleLogger.Success($"  ✓ Dumped {dumpAllResult.DumpedFiles.Count} DBCs to JSON");
                    if (dumpAllResult.ErrorMessage != null)
                    {
                        ConsoleLogger.Warn($"    {dumpAllResult.ErrorMessage}");
                    }
                    
                    // Step 2: Legacy - Also dump AreaTable to CSV for crosswalk compatibility
                    var dumpResult = orchestrator.DumpAreaTables(
                        srcAlias: alias,
                        srcDbcDir: sourceDir,
                        tgtDbcDir: lkDbcDir,
                        outDir: dbcVersionDir);

                    if (!dumpResult.Success)
                    {
                        success = false;
                        error = dumpResult.ErrorMessage ?? "AreaTable CSV dump failed";
                    }
                    else
                    {
                        // DumpAreaTables writes to dbcVersionDir/alias/raw/
                        // We want files in dbcVersionDir/raw/ directly
                        var rawSource = Path.Combine(dbcVersionDir, alias, "raw");
                        if (Directory.Exists(rawSource))
                        {
                            foreach (var file in Directory.EnumerateFiles(rawSource, "*.csv"))
                            {
                                var fileName = Path.GetFileName(file);
                                File.Copy(file, Path.Combine(sharedDbcDir, fileName), overwrite: true);
                            }
                        }
                    }
                }

                if (success)
                {
                    var crosswalkResult = orchestrator.GenerateCrosswalks(
                        srcAlias: alias,
                        srcDbcDir: sourceDir,
                        tgtDbcDir: lkDbcDir,
                        outDir: crosswalkVersionDir,
                        chainVia060: false);

                    if (!crosswalkResult.Success)
                    {
                        success = false;
                        error = crosswalkResult.ErrorMessage ?? "Crosswalk generation failed";
                    }
                }
            }

            if (error is not null)
            {
                overallSuccess = false;
            }

            perVersion.Add(new DbcVersionResult(
                SourceVersion: version,
                SourceAlias: alias,
                SourceDirectory: sourceDir,
                DbcOutputDirectory: sharedDbcDir,
                CrosswalkOutputDirectory: sharedCrosswalkV2Dir,
                DumpExitCode: success ? 0 : 1,
                CompareExitCode: success ? 0 : 1,
                Error: error));
        }

        return new DbcStageResult(overallSuccess, perVersion);
    }

    internal static string DeriveAlias(string version)
    {
        if (string.IsNullOrWhiteSpace(version))
        {
            return string.Empty;
        }

        var parts = version.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length >= 2)
        {
            if (parts.Length >= 3)
            {
                return string.Create(CultureInfo.InvariantCulture, $"{parts[0]}.{parts[1]}.{parts[2]}");
            }

            return string.Create(CultureInfo.InvariantCulture, $"{parts[0]}.{parts[1]}");
        }

        return version;
    }

    private static string? ResolveSourceDbcDirectory(string alphaRoot, string version)
    {
        if (string.IsNullOrWhiteSpace(alphaRoot) || string.IsNullOrWhiteSpace(version))
        {
            return null;
        }

        var candidate = Path.Combine(alphaRoot, version, "tree", "DBFilesClient");
        return Directory.Exists(candidate) ? candidate : candidate;
    }

    private static string? ResolveLkDbcDirectory(PipelineOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.LkDbcDirectory))
        {
            return options.LkDbcDirectory;
        }

        var candidate = Path.Combine(options.AlphaRoot, "3.3.5", "tree", "DBFilesClient");
        return Directory.Exists(candidate) ? candidate : null;
    }

    internal static string ResolveBuildIdentifier(string alias)
    {
        return alias switch
        {
            "0.5.3" => "3368",
            "0.5.5" => "3494",
            "0.6.0" => "3592",
            "3.3.5" => "12340",
            _ => alias.Replace('.', '_')
        };
    }

}

internal sealed record DbcStageResult(
    bool Success,
    IReadOnlyList<DbcVersionResult> Versions);

internal sealed record DbcVersionResult(
    string SourceVersion,
    string SourceAlias,
    string SourceDirectory,
    string DbcOutputDirectory,
    string CrosswalkOutputDirectory,
    int DumpExitCode,
    int CompareExitCode,
    string? Error);
