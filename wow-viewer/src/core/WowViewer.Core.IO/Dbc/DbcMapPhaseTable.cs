using DBCD;
using DBCD.Providers;

namespace WowViewer.Core.IO.Dbc;

/// <summary>One <c>Map.dbc</c> row, reduced to what phase composition needs.</summary>
public sealed record MapPhaseRecord(int MapId, string Directory, string DisplayName, int ParentMapId)
{
    /// <summary>True when this map is a child (phase / terrain swap) of another map.</summary>
    public bool HasParent => ParentMapId >= 0;
}

/// <summary>One <c>Phase.dbc</c> row.</summary>
/// <remarks>
/// <b>Phase.dbc carries no map reference in 5.0.1.</b> Its 5.0.1-5.4.8 layout is exactly
/// <c>ID, Name, Flags</c>. The <c>MapID</c>/<c>ParentMapID</c> columns some documentation shows
/// belong to the <b>4.0.0.11927-4.0.0.12539</b> layout and were gone by 4.0.0.12911. So this table
/// is useful for <em>naming</em> and for identifying terrain swaps, and cannot by itself tell you
/// which map a phase belongs to -- that is <c>Map.dbc.ParentMapID</c>.
/// </remarks>
public sealed record PhaseRecord(int PhaseId, string Name, int Flags)
{
    /// <summary>
    /// Whether the row's name marks it as a terrain swap.
    /// </summary>
    /// <remarks>
    /// MEASURED on 5.0.1.15464: of 814 rows, the ones named "... Terrain Swap" are the ones that
    /// correspond to a swapped ADT set. <c>Flags</c> is frequently 4 on those rows, but that is an
    /// observed correlation and is <b>not</b> established as the flag's meaning, so the name is what
    /// is matched here and the flag is exposed raw for a caller that wants to test it.
    /// </remarks>
    public bool LooksLikeTerrainSwap =>
        Name.Contains("Terrain Swap", StringComparison.OrdinalIgnoreCase);
}

/// <summary>
/// Reads the client tables that describe phase / terrain-swap map relationships.
/// </summary>
/// <remarks>
/// <para>
/// The relationship that actually matters for composition is <c>Map.dbc.ParentMapID</c>: a child map
/// names its parent, which is what lets the viewer offer "these are the phases of the map you have
/// open" instead of asking the user to know the directory names.
/// </para>
/// <para>
/// MEASURED on 5.0.1.15464 (239 map rows): 18 carry a parent, including
/// <c>JadeForestAllianceHubPhase</c> and <c>JadeForestBattlefieldPhase</c> -&gt; <c>HawaiiMainLand</c>
/// (Pandaria), <c>GilneasPhase1</c>/<c>GilneasPhase2</c>/<c>Gilneas</c> -&gt; <c>Gilneas2</c>, and
/// <c>LostIslesPhase1</c>/<c>LostIslesPhase2</c> -&gt; <c>LostIsles</c>. Maps with no parent store
/// <c>-1</c>.
/// </para>
/// </remarks>
public sealed class DbcMapPhaseTable
{
    private static readonly string[] DirectoryColumns = ["Directory", "InternalName"];
    private static readonly string[] ParentColumns = ["ParentMapID", "ParentMapId", "ParentMap"];
    private static readonly string[] DisplayNameColumns = ["MapName_lang", "MapName", "Name_lang", "Name"];

    private readonly Dictionary<int, MapPhaseRecord> _byMapId = [];
    private readonly Dictionary<string, MapPhaseRecord> _byDirectory = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, List<MapPhaseRecord>> _childrenByParent = [];

    private DbcMapPhaseTable()
    {
    }

    public int RowCount => _byMapId.Count;

    /// <summary>Rows that name a parent map.</summary>
    public IEnumerable<MapPhaseRecord> ChildMaps => _byMapId.Values.Where(static row => row.HasParent);

    /// <summary>An empty table. Every lookup misses and callers fall back to manual configuration.</summary>
    public static DbcMapPhaseTable Empty { get; } = new();

    public static DbcMapPhaseTable FromRows(IEnumerable<MapPhaseRecord> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        DbcMapPhaseTable table = new();
        foreach (MapPhaseRecord row in rows)
            table.Add(row);

        return table;
    }

    public static DbcMapPhaseTable Load(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        IDBCDStorage storage = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "Map");
        return FromStorage(storage);
    }

    public static DbcMapPhaseTable FromStorage(IDBCDStorage storage)
    {
        ArgumentNullException.ThrowIfNull(storage);

        string? idColumn = DbcTableLoader.DetectColumn(storage, DbcTableLoader.IdColumns);
        string? directoryColumn = DbcTableLoader.DetectColumn(storage, DirectoryColumns);
        string? parentColumn = DbcTableLoader.DetectColumn(storage, ParentColumns);
        string? displayColumn = DbcTableLoader.DetectColumn(storage, DisplayNameColumns);

        DbcMapPhaseTable table = new();
        if (directoryColumn is null)
            return table;

        foreach (DBCDRow row in storage.Values)
        {
            int mapId = DbcTableLoader.ResolveRowId(row, idColumn);
            string directory = ReadString(row, directoryColumn);
            if (string.IsNullOrWhiteSpace(directory))
                continue;

            // -1 is the no-parent sentinel; a self-reference is not a parent relationship either.
            int parentId = DbcTableLoader.TryGetInt(row, parentColumn) ?? -1;
            if (parentId == mapId)
                parentId = -1;

            string display = ReadString(row, displayColumn);
            table.Add(new MapPhaseRecord(
                mapId,
                directory,
                string.IsNullOrWhiteSpace(display) ? directory : display,
                parentId));
        }

        return table;
    }

    private void Add(MapPhaseRecord row)
    {
        _byMapId[row.MapId] = row;
        _byDirectory[row.Directory] = row;

        if (!row.HasParent)
            return;

        if (!_childrenByParent.TryGetValue(row.ParentMapId, out List<MapPhaseRecord>? children))
        {
            children = [];
            _childrenByParent[row.ParentMapId] = children;
        }

        children.Add(row);
    }

    public bool TryGetByDirectory(string mapDirectory, out MapPhaseRecord record)
        => _byDirectory.TryGetValue(mapDirectory ?? string.Empty, out record!);

    public bool TryGetByMapId(int mapId, out MapPhaseRecord record)
        => _byMapId.TryGetValue(mapId, out record!);

    /// <summary>
    /// The maps that name <paramref name="mapDirectory"/> as their parent, ordered by directory.
    /// </summary>
    /// <remarks>
    /// A child map may itself be the parent's own base terrain rather than a phase -- on 5.0.1
    /// <c>Gilneas</c> names <c>Gilneas2</c> as its parent -- so the caller decides what to do with
    /// each candidate rather than this method filtering on a guess.
    /// </remarks>
    public IReadOnlyList<MapPhaseRecord> GetChildMapsOf(string mapDirectory)
    {
        if (!TryGetByDirectory(mapDirectory, out MapPhaseRecord parent))
            return [];

        return GetChildMapsOf(parent.MapId);
    }

    public IReadOnlyList<MapPhaseRecord> GetChildMapsOf(int parentMapId)
        => _childrenByParent.TryGetValue(parentMapId, out List<MapPhaseRecord>? children)
            ? children.OrderBy(static child => child.Directory, StringComparer.OrdinalIgnoreCase).ToList()
            : [];

    private static string ReadString(DBCDRow row, string? column)
    {
        if (string.IsNullOrWhiteSpace(column))
            return string.Empty;

        try
        {
            return row[column]?.ToString() ?? string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }
}

/// <summary>
/// Reads <c>Phase.dbc</c> and <c>PhaseXPhaseGroup.dbc</c>.
/// </summary>
/// <remarks>
/// These name and group phases; they do <b>not</b> associate a phase with a map in 5.0.1. Use
/// <see cref="DbcMapPhaseTable"/> for the map relationship.
/// </remarks>
public sealed class DbcPhaseTable
{
    private static readonly string[] NameColumns = ["Name", "Name_lang"];
    private static readonly string[] FlagsColumns = ["Flags"];
    private static readonly string[] PhaseIdColumns = ["PhaseID", "PhaseId"];
    private static readonly string[] PhaseGroupIdColumns = ["PhaseGroupID", "PhaseGroupId"];

    private readonly Dictionary<int, PhaseRecord> _byPhaseId = [];
    private readonly Dictionary<int, List<int>> _phaseIdsByGroup = [];

    private DbcPhaseTable()
    {
    }

    public int RowCount => _byPhaseId.Count;

    public int GroupCount => _phaseIdsByGroup.Count;

    public IEnumerable<PhaseRecord> Phases => _byPhaseId.Values.OrderBy(static phase => phase.PhaseId);

    public static DbcPhaseTable Empty { get; } = new();

    public static DbcPhaseTable FromRows(IEnumerable<PhaseRecord> phases, IEnumerable<(int PhaseId, int GroupId)>? groups = null)
    {
        ArgumentNullException.ThrowIfNull(phases);

        DbcPhaseTable table = new();
        foreach (PhaseRecord phase in phases)
            table._byPhaseId[phase.PhaseId] = phase;

        foreach ((int phaseId, int groupId) in groups ?? [])
            table.AddGroupMember(phaseId, groupId);

        return table;
    }

    public static DbcPhaseTable Load(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        DbcPhaseTable table = new();

        IDBCDStorage phases = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "Phase");
        string? idColumn = DbcTableLoader.DetectColumn(phases, DbcTableLoader.IdColumns);
        string? nameColumn = DbcTableLoader.DetectColumn(phases, NameColumns);
        string? flagsColumn = DbcTableLoader.DetectColumn(phases, FlagsColumns);

        foreach (DBCDRow row in phases.Values)
        {
            int phaseId = DbcTableLoader.ResolveRowId(row, idColumn);
            string name = string.Empty;
            if (nameColumn is not null)
            {
                try { name = row[nameColumn]?.ToString() ?? string.Empty; }
                catch { name = string.Empty; }
            }

            table._byPhaseId[phaseId] = new PhaseRecord(phaseId, name, DbcTableLoader.TryGetInt(row, flagsColumn) ?? 0);
        }

        // The group table is optional: a client without it still yields usable phase names.
        try
        {
            IDBCDStorage groups = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "PhaseXPhaseGroup");
            string? groupPhaseColumn = DbcTableLoader.DetectColumn(groups, PhaseIdColumns);
            string? groupIdColumn = DbcTableLoader.DetectColumn(groups, PhaseGroupIdColumns);
            if (groupPhaseColumn is not null && groupIdColumn is not null)
            {
                foreach (DBCDRow row in groups.Values)
                {
                    int phaseId = DbcTableLoader.TryGetInt(row, groupPhaseColumn) ?? 0;
                    int groupId = DbcTableLoader.TryGetInt(row, groupIdColumn) ?? 0;
                    if (phaseId > 0 && groupId > 0)
                        table.AddGroupMember(phaseId, groupId);
                }
            }
        }
        catch
        {
            // Absent or unreadable group table: leave GroupCount at zero rather than failing the load.
        }

        return table;
    }

    private void AddGroupMember(int phaseId, int groupId)
    {
        if (!_phaseIdsByGroup.TryGetValue(groupId, out List<int>? members))
        {
            members = [];
            _phaseIdsByGroup[groupId] = members;
        }

        if (!members.Contains(phaseId))
            members.Add(phaseId);
    }

    public bool TryGetPhase(int phaseId, out PhaseRecord record) => _byPhaseId.TryGetValue(phaseId, out record!);

    public IReadOnlyList<int> GetPhaseIdsInGroup(int groupId)
        => _phaseIdsByGroup.TryGetValue(groupId, out List<int>? members) ? members : [];

    /// <summary>Phases whose name marks them as a terrain swap.</summary>
    public IEnumerable<PhaseRecord> TerrainSwapPhases => Phases.Where(static phase => phase.LooksLikeTerrainSwap);
}
