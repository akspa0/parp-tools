using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Batch processes multiple WoW versions from the 10.5TB archive.
/// Extracts placements, UniqueIDs, and DBCs from all versions systematically.
/// </summary>
public sealed class VersionBatchProcessor
{
    private readonly string _archiveRoot;
    private readonly string _globalOutputRoot;

    public VersionBatchProcessor(string archiveRoot, string globalOutputRoot)
    {
        _archiveRoot = archiveRoot;
        _globalOutputRoot = globalOutputRoot;
    }

    /// <summary>
    /// Scans archive root for all WoW version directories.
    /// Expected format: "0.X_Pre-Release_OSX_enUS_0.9.1.3810\World of Warcraft\Data"
    /// </summary>
    public List<VersionInfo> ScanVersions()
    {
        var versions = new List<VersionInfo>();

        if (!Directory.Exists(_archiveRoot))
        {
            return versions;
        }

        var versionDirs = Directory.EnumerateDirectories(_archiveRoot, "*", SearchOption.TopDirectoryOnly);

        foreach (var dir in versionDirs)
        {
            var dirName = Path.GetFileName(dir);
            var versionInfo = ParseVersionFolder(dirName, dir);
            
            if (versionInfo != null)
            {
                versions.Add(versionInfo);
            }
        }

        return versions.OrderBy(v => v.MajorVersion)
                      .ThenBy(v => v.MinorVersion)
                      .ThenBy(v => v.PatchVersion)
                      .ThenBy(v => v.BuildNumber)
                      .ToList();
    }

    /// <summary>
    /// Parses version folder name.
    /// Examples:
    /// - "0.X_Pre-Release_OSX_enUS_0.9.1.3810"
    /// - "1.12.1.5875_enUS"
    /// - "3.3.5.12340_enGB"
    /// </summary>
    private VersionInfo? ParseVersionFolder(string folderName, string fullPath)
    {
        // Try various patterns
        var patterns = new[]
        {
            // 0.X_Pre-Release_OSX_enUS_0.9.1.3810
            @"(?<major>\d+)\.(?<minor>[\dX]+)[^_]*_[^_]*_[^_]*_(?<locale>\w+)_(?<major2>\d+)\.(?<minor2>\d+)\.(?<patch>\d+)\.(?<build>\d+)",
            
            // 1.12.1.5875_enUS
            @"(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)\.(?<build>\d+)_(?<locale>\w+)",
            
            // 3.3.5a_enUS (build might be missing)
            @"(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)[a-z]*_(?<locale>\w+)",
        };

        foreach (var pattern in patterns)
        {
            var match = Regex.Match(folderName, pattern);
            if (match.Success)
            {
                // Use the second set of version numbers if present (more accurate)
                var major = match.Groups["major2"].Success 
                    ? ParseInt(match.Groups["major2"].Value) 
                    : ParseInt(match.Groups["major"].Value);
                    
                var minor = match.Groups["minor2"].Success 
                    ? ParseInt(match.Groups["minor2"].Value) 
                    : ParseInt(match.Groups["minor"].Value);

                var patch = ParseInt(match.Groups["patch"].Value);
                var build = ParseInt(match.Groups["build"].Value);
                var locale = match.Groups["locale"].Value;

                // Find Data directory
                var dataDir = FindDataDirectory(fullPath);
                if (dataDir == null)
                    continue;

                return new VersionInfo
                {
                    FolderName = folderName,
                    FullPath = fullPath,
                    DataDirectory = dataDir,
                    MajorVersion = major,
                    MinorVersion = minor,
                    PatchVersion = patch,
                    BuildNumber = build,
                    Locale = locale,
                    VersionString = $"{major}.{minor}.{patch}.{build}"
                };
            }
        }

        return null;
    }

    private string? FindDataDirectory(string versionRoot)
    {
        // Common locations
        var candidates = new[]
        {
            Path.Combine(versionRoot, "World of Warcraft", "Data"),
            Path.Combine(versionRoot, "Data"),
            Path.Combine(versionRoot, "data")
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
    }

    /// <summary>
    /// Processes a single version: extracts placements, analyzes UniqueIDs.
    /// </summary>
    public VersionProcessResult ProcessVersion(VersionInfo version, string mapName)
    {
        try
        {
            var versionOutputDir = Path.Combine(_globalOutputRoot, version.VersionString);
            Directory.CreateDirectory(versionOutputDir);

            // Write version metadata
            var metadataPath = Path.Combine(versionOutputDir, "version_info.json");
            File.WriteAllText(metadataPath, JsonSerializer.Serialize(version, new JsonSerializerOptions { WriteIndented = true }));

            // TODO: Process MPQ files or loose ADTs from this version
            // For now, just log the discovery
            
            return new VersionProcessResult
            {
                VersionString = version.VersionString,
                Success = true,
                OutputDirectory = versionOutputDir,
                PlacementsExtracted = 0,
                DbcsExported = 0
            };
        }
        catch (Exception ex)
        {
            return new VersionProcessResult
            {
                VersionString = version.VersionString,
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Batch processes all versions in the archive.
    /// </summary>
    public BatchProcessSummary ProcessAllVersions(string mapName, Action<string>? progressCallback = null)
    {
        var versions = ScanVersions();
        var results = new List<VersionProcessResult>();

        progressCallback?.Invoke($"Found {versions.Count} versions in archive");

        foreach (var version in versions)
        {
            progressCallback?.Invoke($"Processing {version.VersionString}...");
            
            var result = ProcessVersion(version, mapName);
            results.Add(result);

            if (result.Success)
            {
                progressCallback?.Invoke($"  ✓ {version.VersionString} complete");
            }
            else
            {
                progressCallback?.Invoke($"  ✗ {version.VersionString} failed: {result.ErrorMessage}");
            }
        }

        return new BatchProcessSummary
        {
            TotalVersions = versions.Count,
            SuccessfulVersions = results.Count(r => r.Success),
            FailedVersions = results.Count(r => !r.Success),
            Results = results
        };
    }

    private static int ParseInt(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value == "X")
            return 0;
            
        return int.TryParse(value, out var result) ? result : 0;
    }
}

/// <summary>
/// Information about a discovered WoW version.
/// </summary>
public record VersionInfo
{
    public required string FolderName { get; init; }
    public required string FullPath { get; init; }
    public required string DataDirectory { get; init; }
    public required int MajorVersion { get; init; }
    public required int MinorVersion { get; init; }
    public required int PatchVersion { get; init; }
    public required int BuildNumber { get; init; }
    public required string Locale { get; init; }
    public required string VersionString { get; init; }
}

/// <summary>
/// Result of processing a single version.
/// </summary>
public record VersionProcessResult
{
    public required string VersionString { get; init; }
    public required bool Success { get; init; }
    public string? OutputDirectory { get; init; }
    public int PlacementsExtracted { get; init; }
    public int DbcsExported { get; init; }
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Summary of batch processing all versions.
/// </summary>
public record BatchProcessSummary
{
    public required int TotalVersions { get; init; }
    public required int SuccessfulVersions { get; init; }
    public required int FailedVersions { get; init; }
    public required List<VersionProcessResult> Results { get; init; }
}
