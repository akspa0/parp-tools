using System.Text.RegularExpressions;

namespace WowViewer.Core.IO.Wmo;

/// <summary>
/// A single WMO minimap BLP entry discovered from MPQ archives.
/// Based on Ghidra RE of wowclient.exe build 3368: the client resolves
/// WMO minimap BLPs via the pattern &lt;WMOName&gt;_&lt;groupIdx&gt;_&lt;quadY&gt;_&lt;quadX&gt;.blp
/// under Textures\Minimap\.
/// </summary>
public sealed record WmoMinimapBlpEntry
{
    /// <summary>Full MPQ virtual path to the BLP file (e.g. Textures\Minimap\deadmines_000_00_00.blp).</summary>
    public required string BlpPath { get; init; }

    /// <summary>WMO stem name extracted from the filename (e.g. "deadmines").</summary>
    public required string WmoStem { get; init; }

    /// <summary>Group index within the WMO (3-digit zero-padded in some builds).</summary>
    public required int GroupIndex { get; init; }

    /// <summary>Quad Y coordinate (2-digit zero-padded).</summary>
    public required int QuadY { get; init; }

    /// <summary>Quad X coordinate (2-digit zero-padded).</summary>
    public required int QuadX { get; init; }

    /// <summary>Client build this entry was discovered from.</summary>
    public required string Build { get; init; }
}

/// <summary>
/// Discovers and parses WMO minimap BLP filenames from staged client MPQ archives.
/// </summary>
public static class WmoMinimapBlpReader
{
    /// <summary>
    /// Regex matching the WMO minimap BLP pattern: &lt;stem&gt;_&lt;digits&gt;_&lt;digits&gt;_&lt;digits&gt;.blp
    /// The stem contains alphanumeric characters and hyphens (no underscores, since underscores
    /// are the delimiter between stem/group/quadY/quadX).
    /// Group index is 2-3 digits, quad coordinates are 2 digits each.
    /// </summary>
    private static readonly Regex WmoMinimapPattern = new(
        @"^(?<stem>[A-Za-z0-9-]+?)_(?<group>\d{2,3})_(?<quady>\d{2})_(?<quadx>\d{2})\.blp$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Discover all WMO minimap BLPs from the given archive catalog's known file list.
    /// Filters by the Textures\Minimap\ prefix and the WMO naming pattern.
    /// </summary>
    /// <param name="catalog">The MPQ archive catalog with loaded archives and listfile.</param>
    /// <param name="build">Build version string to tag each entry (e.g. "3_3_5_12340").</param>
    /// <returns>List of discovered WMO minimap BLP entries.</returns>
    public static List<WmoMinimapBlpEntry> DiscoverMinimapBlps(
        Files.IArchiveCatalog catalog,
        string build)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentException.ThrowIfNullOrWhiteSpace(build);

        var entries = new List<WmoMinimapBlpEntry>();
        var knownFiles = catalog.GetAllKnownFiles();

        foreach (string path in knownFiles)
        {
            if (!path.StartsWith("Textures\\Minimap\\", StringComparison.OrdinalIgnoreCase) &&
                !path.StartsWith("Textures/Minimap/", StringComparison.OrdinalIgnoreCase))
                continue;

            string filename = Path.GetFileName(path);
            var match = WmoMinimapPattern.Match(filename);
            if (!match.Success)
                continue;

            entries.Add(new WmoMinimapBlpEntry
            {
                BlpPath = path,
                WmoStem = match.Groups["stem"].Value,
                GroupIndex = ParseGroupIndex(match.Groups["group"].Value),
                QuadY = int.Parse(match.Groups["quady"].Value),
                QuadX = int.Parse(match.Groups["quadx"].Value),
                Build = build,
            });
        }

        return entries;
    }

    /// <summary>
    /// Parse the group index from its string representation.
    /// Handles both 2-digit and 3-digit zero-padded formats.
    /// </summary>
    public static int ParseGroupIndex(string groupStr)
    {
        return int.Parse(groupStr);
    }

    /// <summary>
    /// Check if a filename matches the WMO minimap BLP pattern without constructing a full entry.
    /// Useful for fast filtering.
    /// </summary>
    public static bool IsWmoMinimapBlp(string filename)
    {
        return WmoMinimapPattern.IsMatch(filename);
    }

    /// <summary>
    /// Try to parse a WMO minimap BLP filename into its components.
    /// Returns false if the filename does not match the expected pattern.
    /// </summary>
    public static bool TryParseFilename(
        string filename,
        out string? wmoStem,
        out int groupIndex,
        out int quadY,
        out int quadX)
    {
        var match = WmoMinimapPattern.Match(filename);
        if (!match.Success)
        {
            wmoStem = null;
            groupIndex = 0;
            quadY = 0;
            quadX = 0;
            return false;
        }

        wmoStem = match.Groups["stem"].Value;
        groupIndex = ParseGroupIndex(match.Groups["group"].Value);
        quadY = int.Parse(match.Groups["quady"].Value);
        quadX = int.Parse(match.Groups["quadx"].Value);
        return true;
    }
}
