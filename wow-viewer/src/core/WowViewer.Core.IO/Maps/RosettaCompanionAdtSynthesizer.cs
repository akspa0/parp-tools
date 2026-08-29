using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Represents an identified PM4 tile and its companion ADT status.
/// </summary>
public sealed record RosettaOrphanPm4Tile(
    string Pm4FilePath,
    string MapName,
    int TileX,
    int TileY,
    string ExpectedCompanionFileName,
    bool HasExistingCompanion);

/// <summary>
/// Options controlling companion ADT synthesis.
/// </summary>
public sealed record RosettaCompanionSynthesisOptions(
    string GroundTexture = "tileset\\ocean\\westfallseafloor.blp",
    bool EmitSplitObj0 = false,
    bool Overwrite = false,
    string? DefaultMapName = null);

/// <summary>
/// Record of a synthesized companion ADT file with cryptographic provenance.
/// </summary>
public sealed record RosettaSynthesizedCompanionRecord(
    string Pm4FilePath,
    string MapName,
    int TileX,
    int TileY,
    string CompanionFilePath,
    string ContentSha256,
    string Status,
    string CreatedAtUtc,
    string? ErrorMessage = null);

/// <summary>
/// Comprehensive provenance report distinguishing synthetic data from authentic game data (FR-010).
/// </summary>
public sealed record RosettaCompanionProvenanceReport(
    string ReportVersion,
    string GeneratedUtc,
    string Pm4SourceDirectory,
    string OutputDirectory,
    int TotalPm4FilesScanned,
    int OrphanCount,
    int SynthesizedCount,
    int SkippedCount,
    int FailedCount,
    RosettaCompanionSynthesisOptions Options,
    IReadOnlyList<RosettaSynthesizedCompanionRecord> Records);

/// <summary>
/// Scans orphan PM4 files lacking companion ADTs and synthesizes minimal compliant companion
/// ADT files with cryptographic provenance tracking (Spec 190 US4 / FR-009 / FR-010).
/// </summary>
public static partial class RosettaCompanionAdtSynthesizer
{
    public const string CurrentProvenanceVersion = "rosetta-companion-provenance-v1";

    [GeneratedRegex(@"^(?<map>.+?)_(?<first>\d{1,2})_(?<second>\d{1,2})(?:_.*)?\.pm4$", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex Pm4FilenameRegex();

    /// <summary>
    /// Scans a directory for PM4 tiles and identifies which tiles lack companion ADT files.
    /// </summary>
    public static IReadOnlyList<RosettaOrphanPm4Tile> ScanPm4Tiles(
        string pm4Directory,
        string? adtDirectory = null,
        string? defaultMapName = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pm4Directory);
        if (!Directory.Exists(pm4Directory))
            return [];

        string targetAdtDir = !string.IsNullOrWhiteSpace(adtDirectory) && Directory.Exists(adtDirectory)
            ? adtDirectory
            : pm4Directory;

        var results = new List<RosettaOrphanPm4Tile>();
        var pm4Files = Directory.EnumerateFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);

        foreach (string pm4File in pm4Files)
        {
            string fileName = Path.GetFileName(pm4File);
            Match match = Pm4FilenameRegex().Match(fileName);
            if (!match.Success)
                continue;

            string mapName = match.Groups["map"].Value;
            if (string.IsNullOrWhiteSpace(mapName) && !string.IsNullOrWhiteSpace(defaultMapName))
                mapName = defaultMapName;

            if (!int.TryParse(match.Groups["first"].Value, out int first) ||
                !int.TryParse(match.Groups["second"].Value, out int second))
            {
                continue;
            }

            // Standard coordinate mapping: first is tileY, second is tileX in standard ADT naming,
            // or (first=X, second=Y). We check both standard naming forms for existing companions.
            int tileX = second;
            int tileY = first;

            string stdAdtName = $"{mapName}_{tileY}_{tileX}.adt";
            string altAdtName = $"{mapName}_{tileX}_{tileY}.adt";
            string stdObj0Name = $"{mapName}_{tileY}_{tileX}_obj0.adt";
            string altObj0Name = $"{mapName}_{tileX}_{tileY}_obj0.adt";

            bool hasCompanion = File.Exists(Path.Combine(targetAdtDir, stdAdtName))
                || File.Exists(Path.Combine(targetAdtDir, altAdtName))
                || File.Exists(Path.Combine(targetAdtDir, stdObj0Name))
                || File.Exists(Path.Combine(targetAdtDir, altObj0Name));

            results.Add(new RosettaOrphanPm4Tile(
                pm4File,
                mapName,
                tileX,
                tileY,
                stdAdtName,
                hasCompanion));
        }

        return results;
    }

    /// <summary>
    /// Synthesizes a minimal compliant companion ADT for an orphan PM4 tile.
    /// </summary>
    public static RosettaSynthesizedCompanionRecord SynthesizeCompanion(
        RosettaOrphanPm4Tile orphan,
        string outputDirectory,
        RosettaCompanionSynthesisOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(orphan);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);

        options ??= new RosettaCompanionSynthesisOptions();
        Directory.CreateDirectory(outputDirectory);

        string fileName = options.EmitSplitObj0
            ? $"{orphan.MapName}_{orphan.TileY}_{orphan.TileX}_obj0.adt"
            : $"{orphan.MapName}_{orphan.TileY}_{orphan.TileX}.adt";

        string targetPath = Path.Combine(outputDirectory, fileName);
        string utcNow = DateTime.UtcNow.ToString("o");

        if (File.Exists(targetPath) && !options.Overwrite)
        {
            byte[] existingBytes = File.ReadAllBytes(targetPath);
            string existingHash = Convert.ToHexString(SHA256.HashData(existingBytes)).ToLowerInvariant();
            return new RosettaSynthesizedCompanionRecord(
                orphan.Pm4FilePath,
                orphan.MapName,
                orphan.TileX,
                orphan.TileY,
                targetPath,
                existingHash,
                "SkippedExisting",
                utcNow);
        }

        try
        {
            LkAdtData blankAdt = BlankAdtFactory.CreateBlank(
                orphan.MapName,
                orphan.TileX,
                orphan.TileY,
                options.GroundTexture);

            byte[] adtBytes = LkAdtWriter.Build(blankAdt);
            File.WriteAllBytes(targetPath, adtBytes);

            string hash = Convert.ToHexString(SHA256.HashData(adtBytes)).ToLowerInvariant();
            return new RosettaSynthesizedCompanionRecord(
                orphan.Pm4FilePath,
                orphan.MapName,
                orphan.TileX,
                orphan.TileY,
                targetPath,
                hash,
                "Synthesized",
                utcNow);
        }
        catch (Exception ex)
        {
            return new RosettaSynthesizedCompanionRecord(
                orphan.Pm4FilePath,
                orphan.MapName,
                orphan.TileX,
                orphan.TileY,
                targetPath,
                string.Empty,
                "Failed",
                utcNow,
                ex.Message);
        }
    }

    /// <summary>
    /// Scans for orphan PM4 tiles and synthesizes companions for all of them, producing a provenance report.
    /// </summary>
    public static RosettaCompanionProvenanceReport SynthesizeAllCompanions(
        string pm4Directory,
        string outputDirectory,
        string? adtDirectory = null,
        RosettaCompanionSynthesisOptions? options = null,
        string? provenanceReportPath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pm4Directory);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);

        options ??= new RosettaCompanionSynthesisOptions();
        var allTiles = ScanPm4Tiles(pm4Directory, adtDirectory, options.DefaultMapName);
        var orphans = allTiles.Where(static t => !t.HasExistingCompanion).ToList();

        var records = new List<RosettaSynthesizedCompanionRecord>(allTiles.Count);

        foreach (RosettaOrphanPm4Tile tile in allTiles)
        {
            if (tile.HasExistingCompanion && !options.Overwrite)
            {
                records.Add(new RosettaSynthesizedCompanionRecord(
                    tile.Pm4FilePath,
                    tile.MapName,
                    tile.TileX,
                    tile.TileY,
                    Path.Combine(outputDirectory, tile.ExpectedCompanionFileName),
                    string.Empty,
                    "SkippedExisting",
                    DateTime.UtcNow.ToString("o")));
                continue;
            }

            var record = SynthesizeCompanion(tile, outputDirectory, options);
            records.Add(record);
        }

        int synthesized = records.Count(static r => r.Status == "Synthesized");
        int skipped = records.Count(static r => r.Status == "SkippedExisting");
        int failed = records.Count(static r => r.Status == "Failed");

        var report = new RosettaCompanionProvenanceReport(
            CurrentProvenanceVersion,
            DateTime.UtcNow.ToString("o"),
            pm4Directory,
            outputDirectory,
            allTiles.Count,
            orphans.Count,
            synthesized,
            skipped,
            failed,
            options,
            records);

        if (!string.IsNullOrWhiteSpace(provenanceReportPath))
        {
            string? reportDir = Path.GetDirectoryName(provenanceReportPath);
            if (!string.IsNullOrEmpty(reportDir))
                Directory.CreateDirectory(reportDir);

            string json = JsonSerializer.Serialize(report, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            File.WriteAllText(provenanceReportPath, json);
        }

        return report;
    }
}
