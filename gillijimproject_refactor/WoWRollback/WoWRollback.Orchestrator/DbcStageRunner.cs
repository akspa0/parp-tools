using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using DBCTool.V2.Cli;

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

        var workRoot = Path.Combine(session.Root, "work");
        Directory.CreateDirectory(workRoot);

        var dumpCommand = new DumpAreaCommand();
        var compareCommand = new CompareAreaV2Command();

        var perVersion = new List<DbcVersionResult>();
        var overallSuccess = true;

        foreach (var version in options.Versions)
        {
            var alias = DeriveAlias(version);
            var build = ResolveBuildIdentifier(alias);
            var sourceDir = ResolveSourceDbcDirectory(options.AlphaRoot, version);

            var sharedDbcDir = Path.Combine(session.SharedDbcRoot, alias, build, "DBC");
            var sharedCrosswalkAliasRoot = Path.Combine(session.SharedCrosswalkRoot, alias);
            var sharedCrosswalkVersionRoot = Path.Combine(sharedCrosswalkAliasRoot, build);
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

            int dumpExit;
            int compareExit;
            string? error = null;

            if (hasCachedDbc && hasCachedCrosswalk)
            {
                dumpExit = 0;
                compareExit = 0;
            }
            else
            {
                var tempRoot = Path.Combine(workRoot, $"{alias}_{build}_{Guid.NewGuid():N}");
                var tempDbcRoot = Path.Combine(tempRoot, "dbc");
                var tempCrosswalkRoot = Path.Combine(tempRoot, "crosswalk");
                Directory.CreateDirectory(tempDbcRoot);
                Directory.CreateDirectory(tempCrosswalkRoot);

                dumpExit = dumpCommand.Run(
                    dbdDir: dbdDir,
                    outBase: tempDbcRoot,
                    localeStr: locale,
                    inputs: inputs);

                compareExit = compareCommand.Run(
                    dbdDir: dbdDir,
                    outBase: tempCrosswalkRoot,
                    localeStr: locale,
                    inputs: inputs,
                    chainVia060: false);

                if (dumpExit == 0)
                {
                    var rawSource = Path.Combine(tempDbcRoot, alias, "raw");
                    CopyDirectory(rawSource, sharedDbcDir, overwrite: true);
                }

                if (compareExit == 0)
                {
                    var tempAliasRoot = Path.Combine(tempCrosswalkRoot, alias);
                    var compareSource = Path.Combine(tempAliasRoot, "compare");
                    var auditSource = Path.Combine(tempAliasRoot, "audit");
                    CopyDirectory(compareSource, sharedCrosswalkCompareRoot, overwrite: true);
                    CopyDirectory(auditSource, Path.Combine(sharedCrosswalkVersionRoot, "audit"), overwrite: true);

                    var mapsSource = Path.Combine(tempAliasRoot, "maps.json");
                    if (File.Exists(mapsSource))
                    {
                        File.Copy(mapsSource, mapsJsonPath, overwrite: true);
                        File.Copy(mapsSource, Path.Combine(sharedCrosswalkVersionRoot, "maps.json"), overwrite: true);
                    }
                }

                try
                {
                    if (Directory.Exists(tempRoot))
                    {
                        Directory.Delete(tempRoot, recursive: true);
                    }
                }
                catch
                {
                    // Best-effort cleanup; ignore failures.
                }

                if (dumpExit != 0 || compareExit != 0)
                {
                    error = "DBCTool execution reported a non-zero exit code.";
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
                DumpExitCode: hasCachedDbc && hasCachedCrosswalk ? 0 : dumpExit,
                CompareExitCode: hasCachedDbc && hasCachedCrosswalk ? 0 : compareExit,
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

    private static void CopyDirectory(string sourceDir, string destinationDir, bool overwrite)
    {
        if (!Directory.Exists(sourceDir))
        {
            return;
        }

        if (Directory.Exists(destinationDir) && overwrite)
        {
            Directory.Delete(destinationDir, recursive: true);
        }

        Directory.CreateDirectory(destinationDir);

        foreach (var directory in Directory.EnumerateDirectories(sourceDir, "*", SearchOption.AllDirectories))
        {
            var relativeDir = Path.GetRelativePath(sourceDir, directory);
            var targetDir = Path.Combine(destinationDir, relativeDir);
            Directory.CreateDirectory(targetDir);
        }

        foreach (var file in Directory.EnumerateFiles(sourceDir, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(sourceDir, file);
            var targetPath = Path.Combine(destinationDir, relative);
            var targetDirectory = Path.GetDirectoryName(targetPath);
            if (!string.IsNullOrEmpty(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            File.Copy(file, targetPath, overwrite: true);
        }
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
