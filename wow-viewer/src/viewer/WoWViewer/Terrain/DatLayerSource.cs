using System.Diagnostics.CodeAnalysis;

namespace WoWViewer.Terrain;

/// <summary>
/// Spec 247 US5: lets a folder of AHDR-family DAT files act as a Cartography layer source.
/// <para>
/// A <see cref="WowViewer.Core.Maps.PhaseLayerSettings"/> names its donor with a single string, and every
/// resolution site in the base adapters takes that string. DAT files have no map name and no WDT, so this
/// encodes a folder into the same slot as <c>dat:&lt;absolute folder&gt;</c>. Keeping the locator in the existing
/// field means layer cloning and Cartography project persistence keep working untouched, and a base adapter that
/// never sees the prefix behaves exactly as before.
/// </para>
/// <para>
/// DAT files are the terrain project files the client's ADTs were built from, so a DAT layer over the shipped
/// map is a direct before/after of the same ground. The layer's own tile offset, cell offset, quarter-turn
/// rotation and mirrors are also the tool for settling the unresolved DAT axis question (the DAT grid's row axis
/// runs along ALOC tile Y while the renderer's runs along its tile X): align it by eye once, and the answer
/// feeds back into <c>DatToLkAdtConverter</c>'s <c>--transpose</c>.
/// </para>
/// </summary>
public static class DatLayerSource
{
    public const string Prefix = "dat:";

    private static readonly Dictionary<string, AhdrTerrainAdapter> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object Gate = new();

    /// <summary>Builds the layer locator for a folder of DAT files.</summary>
    public static string ForFolder(string folder)
        => Prefix + Path.GetFullPath(folder);

    public static bool IsDatSource([NotNullWhen(true)] string? mapName)
        => mapName is not null && mapName.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase);

    /// <summary>Extracts the folder from a locator. False when this is an ordinary map name.</summary>
    public static bool TryGetFolder([NotNullWhen(true)] string? mapName, [NotNullWhen(true)] out string? folder)
    {
        folder = null;
        if (!IsDatSource(mapName))
            return false;

        string candidate = mapName[Prefix.Length..].Trim();
        if (candidate.Length == 0)
            return false;

        folder = candidate;
        return true;
    }

    /// <summary>Folder leaf name for the layer list, e.g. "dat:...\Expansion01" -> "DAT: Expansion01".</summary>
    public static string DisplayName(string mapName)
        => TryGetFolder(mapName, out string? folder)
            ? "DAT: " + Path.GetFileName(Path.TrimEndingDirectorySeparator(folder))
            : mapName;

    /// <summary>
    /// The adapter for a locator, scanned once and reused. Returns null when the locator is not a DAT source, the
    /// folder is gone, or nothing in it is AHDR-family. Never throws: an unreadable donor is a displayable state.
    /// </summary>
    public static AhdrTerrainAdapter? Resolve(string? mapName, float heightDivisor)
    {
        if (!TryGetFolder(mapName, out string? folder))
            return null;

        lock (Gate)
        {
            if (Cache.TryGetValue(folder, out AhdrTerrainAdapter? cached))
                return cached.ExistingTiles.Count > 0 ? cached : null;

            AhdrTerrainAdapter? adapter = null;
            try
            {
                if (Directory.Exists(folder))
                    adapter = new AhdrTerrainAdapter(folder, heightDivisor);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                adapter = null;
            }

            if (adapter is null)
                return null;

            Cache[folder] = adapter;
            return adapter.ExistingTiles.Count > 0 ? adapter : null;
        }
    }

    /// <summary>Tiles a DAT donor occupies in its own grid, for the Cartography footprint overlay.</summary>
    public static IReadOnlyList<(int TileX, int TileY)> OccupiedTiles(string? mapName, float heightDivisor)
    {
        AhdrTerrainAdapter? adapter = Resolve(mapName, heightDivisor);
        if (adapter is null)
            return [];

        var tiles = new List<(int, int)>(adapter.ExistingTiles.Count);
        foreach (int key in adapter.ExistingTiles)
            tiles.Add((key / 64, key % 64));

        return tiles;
    }

    /// <summary>Drops a cached donor so the next resolve rescans, for when the folder changed on disk.</summary>
    public static void Forget(string? mapName)
    {
        if (!TryGetFolder(mapName, out string? folder))
            return;

        lock (Gate)
        {
            Cache.Remove(folder);
        }
    }
}
