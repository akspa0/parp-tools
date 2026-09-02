namespace WoWViewer.Terrain;

/// <summary>How a list of discovered maps is ordered for display.</summary>
public enum MapListSortMode
{
    /// <summary>Ascending Map.dbc row id. Loose maps without a DBC entry sort after all of them.</summary>
    MapId = 0,

    /// <summary>Ascending display name.</summary>
    Name = 1,

    /// <summary>Ascending map directory, which is what the phase stack and file paths actually key on.</summary>
    Directory = 2,
}

/// <summary>
/// Shared ordering for every list of discovered maps.
/// </summary>
/// <remarks>
/// <para>
/// Every map row is rendered as <c>[id] Name</c>, but the lists were ordered by <em>name</em>, so the
/// visible leading number ran in no order at all. That makes finding a map by its id -- which is how
/// the DBC, the phase relationships and most external tooling refer to maps -- a linear scan.
/// </para>
/// <para>
/// Ordering therefore belongs at the display layer rather than in <c>MapDiscoveryService</c>, so it
/// can be switched without re-running discovery, and so every list that shows maps agrees.
/// </para>
/// </remarks>
public static class MapListSorting
{
    /// <summary>Loose maps carry no DBC id; keep them together after the numbered ones.</summary>
    private const int NoDbcIdSortKey = int.MaxValue;

    public static IEnumerable<MapDefinition> Sort(IEnumerable<MapDefinition> maps, MapListSortMode mode)
    {
        ArgumentNullException.ThrowIfNull(maps);

        return mode switch
        {
            MapListSortMode.Name => maps
                .OrderBy(static m => m.Name, StringComparer.OrdinalIgnoreCase)
                .ThenBy(static m => m.Directory, StringComparer.OrdinalIgnoreCase),

            MapListSortMode.Directory => maps
                .OrderBy(static m => m.Directory, StringComparer.OrdinalIgnoreCase),

            _ => maps
                .OrderBy(static m => m.HasDbcEntry ? m.Id : NoDbcIdSortKey)
                .ThenBy(static m => m.Directory, StringComparer.OrdinalIgnoreCase),
        };
    }

    public static string Describe(MapListSortMode mode) => mode switch
    {
        MapListSortMode.Name => "Name",
        MapListSortMode.Directory => "Directory",
        _ => "Map ID",
    };
}
