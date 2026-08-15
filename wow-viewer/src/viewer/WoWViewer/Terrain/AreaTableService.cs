using DBCD;
using DBCD.Providers;
using WoWViewer.Logging;
using WowViewer.Core.World;

namespace WoWViewer.Terrain;

/// <summary>
/// Loads AreaTable.dbc via DBCD and provides AreaID → area name lookups.
/// Used to display the current area name in the status bar based on the camera's chunk position.
/// </summary>
public class AreaTableService
{
    private readonly Dictionary<int, AreaEntry> _areasById = new();
    private readonly Dictionary<int, AreaEntry> _areas = new();
    private readonly Dictionary<(int MapId, AreaNumberParts AreaNumber), AreaEntry> _areasByMapAndNumber = new();
    private readonly Dictionary<AreaNumberParts, List<AreaEntry>> _areasByNumber = new();
    private int _rowCount;
    private int _primaryKeyCount;
    private int _fallbackAliasCount;
    private int _fallbackAliasCollisions;

    public record AreaEntry(
        int Id,
        string Name,
        int ParentAreaId,
        int MapId,
        int Flags,
        int AreaNumber = 0,
        int ParentAreaNumber = 0);

    public int Count => _primaryKeyCount;
    public string? LoadedBuild { get; private set; }
    public AreaIdentityLayout IdentityLayout { get; private set; } = AreaIdentityLayout.DirectAreaId;
    public string? LoadedLocale { get; private set; }
    public string? NameColumn { get; private set; }
    public string? IdColumn { get; private set; }
    public string? ParentColumn { get; private set; }
    public string? MapColumn { get; private set; }
    public string? FlagsColumn { get; private set; }

    /// <summary>
    /// Load AreaTable.dbc from the given DBC provider.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build)
    {
        _areasById.Clear();
        _areas.Clear();
        _areasByMapAndNumber.Clear();
        _areasByNumber.Clear();
        _rowCount = 0;
        _primaryKeyCount = 0;
        _fallbackAliasCount = 0;
        _fallbackAliasCollisions = 0;
        LoadedBuild = build;
        IdentityLayout = AreaIdentityLayoutResolver.FromBuild(build);

        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        IDBCDStorage storage;
        Locale localeUsed;
        try
        {
            try
            {
                storage = dbcd.Load("AreaTable", build, Locale.EnUS);
                localeUsed = Locale.EnUS;
            }
            catch (Exception enUsEx)
            {
                ViewerLog.Important(ViewerLog.Category.General,
                    $"[AreaTable] Locale.EnUS load failed for build {build}: {enUsEx.Message}. Retrying Locale.None.");
                storage = dbcd.Load("AreaTable", build, Locale.None);
                localeUsed = Locale.None;
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Dbc,
                $"[AreaTable] Failed to load AreaTable.dbc for build {build}: {ex.Message}");
            return;
        }

        LoadedLocale = localeUsed.ToString();

        var availableColumns = new HashSet<string>(storage.AvailableColumns, StringComparer.OrdinalIgnoreCase);

        // Detect column names from the active DBD-backed layout rather than probing a row.
        string? nameCol = DetectColumn(availableColumns, "AreaName_lang", "AreaName", "Name");
        bool alphaAreaNumberLayout = build.StartsWith("0.5.", StringComparison.OrdinalIgnoreCase);
        string? idCol = DetectColumn(availableColumns, "ID", "AreaID", "AreaNumber");
        string? areaNumberCol = DetectColumn(availableColumns, "AreaNumber");
        string? parentCol = alphaAreaNumberLayout
            ? DetectColumn(availableColumns, "ParentAreaNum", "ParentAreaID")
            : DetectColumn(availableColumns, "ParentAreaID", "ParentAreaNum");
        string? parentNumberCol = DetectColumn(availableColumns, "ParentAreaNum");
        string? mapCol = DetectColumn(availableColumns, "ContinentID", "MapID", "Continent");
        string? flagsCol = DetectColumn(availableColumns, "Flags", "AreaFlags");
        NameColumn = nameCol;
        IdColumn = idCol;
        ParentColumn = parentCol;
        MapColumn = mapCol;
        FlagsColumn = flagsCol;

        // Standard MCNK values are direct AreaTable IDs. Alpha MCNK values use
        // AreaNumber with high16=zone and low16=subzone, so AreaNumber is the
        // primary key for the Alpha layout and is indexed by both halves.
        foreach (var key in storage.Keys)
        {
            _rowCount++;
            var row = storage[key];
            int areaId = SafeIntField(row, idCol, key);
            if (areaId == 0 && key != 0)
                areaId = key;

            string name = Sanitize(SafeField<string>(row, nameCol, string.Empty) ?? string.Empty);
            int parentId = SafeIntField(row, parentCol, 0);
            int mapId = SafeIntField(row, mapCol, 0);
            int flags = SafeIntField(row, flagsCol, 0);
            int areaNumber = SafeIntField(row, areaNumberCol, 0);
            int parentAreaNumber = SafeIntField(row, parentNumberCol, 0);

            var entry = new AreaEntry(areaId, name, parentId, mapId, flags, areaNumber, parentAreaNumber);
            RegisterPrimary(areaId, entry);
            RegisterAlias(key, entry);
            RegisterAlias(areaNumber, entry);
            RegisterAreaNumber(entry);
        }

        ViewerLog.Important(ViewerLog.Category.General,
            $"[AreaTable] Loaded build={LoadedBuild} locale={LoadedLocale} rows={_rowCount} indexed={_areas.Count} primaryKeys={_primaryKeyCount} fallbackAliases={_fallbackAliasCount} aliasCollisions={_fallbackAliasCollisions} nameCol='{FormatColumn(nameCol)}' idCol='{FormatColumn(idCol)}' parentCol='{FormatColumn(parentCol)}' mapCol='{FormatColumn(mapCol)}' flagsCol='{FormatColumn(flagsCol)}'");
    }

    /// <summary>
    /// Look up an area name by AreaID. Returns null if not found.
    /// </summary>
    public string? GetAreaName(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry.Name : null;
    }

    /// <summary>
    /// Get full area entry by ID.
    /// </summary>
    public AreaEntry? GetArea(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry : null;
    }

    /// <summary>
    /// Get area name with parent context (e.g. "Durotar > Razor Hill").
    /// </summary>
    public string GetAreaDisplayName(int areaId)
    {
        AreaLookupResult result = ResolveArea(areaId, mapId: -1);
        if (result.PrimaryText is null)
            return $"Unknown ({areaId})";

        return result.ZoneText == result.SubzoneText || result.ZoneText is null
            ? result.SubzoneText!
            : $"{result.ZoneText} > {result.SubzoneText}";
    }

    /// <summary>
    /// Get area name with parent context, but only if the area belongs to the given MapID.
    /// Returns null if the area belongs to a different map or AreaID is not found.
    /// Alpha MCNK values are packed AreaNumbers: high16 is the zone and low16
    /// is the subzone. Standard MCNK values remain direct AreaTable IDs.
    /// </summary>
    public string? GetAreaDisplayNameForMap(int areaId, int mapId)
    {
        AreaLookupResult result = ResolveArea(areaId, mapId);
        if (result.PrimaryText is null)
            return null;

        return result.ZoneText == result.SubzoneText || result.ZoneText is null
            ? result.SubzoneText
            : $"{result.ZoneText} > {result.SubzoneText}";
    }

    /// <summary>
    /// Resolve a raw MCNK area value into native-style ZoneText/SubzoneText roles.
    /// The active build layout is authoritative: standard-era values resolve only by direct
    /// AreaTable ID, while Alpha packed AreaNumber values resolve by map-aware AreaNumber first.
    /// Map mismatch remains visible in the result instead of erasing an otherwise valid table row.
    /// </summary>
    public AreaLookupResult ResolveArea(int rawAreaId, int mapId)
    {
        if (rawAreaId == 0)
            return AreaLookupResult.Unresolved(rawAreaId, mapId, AreaResolutionReason.MissingAreaId);

        AreaEntry? entry = null;
        AreaContextSource source = AreaContextSource.DirectAreaId;

        if (IdentityLayout == AreaIdentityLayout.PackedAreaNumber)
        {
            AreaNumberParts rawAreaNumber = AreaNumberParts.FromRaw(rawAreaId);
            if (_areasByMapAndNumber.TryGetValue((mapId, rawAreaNumber), out AreaEntry? packedEntry))
            {
                entry = packedEntry;
                source = AreaContextSource.PackedAreaNumber;
            }
            else if (mapId < 0 && TryGetUniqueAreaNumber(rawAreaNumber, out packedEntry))
            {
                entry = packedEntry;
                source = AreaContextSource.PackedAreaNumber;
            }
            else if (_areasById.TryGetValue(rawAreaId, out AreaEntry? alphaFallbackEntry))
            {
                entry = alphaFallbackEntry;
                source = AreaContextSource.PackedAreaNumber;
            }
        }
        else if (_areasById.TryGetValue(rawAreaId, out AreaEntry? directEntry))
        {
            entry = directEntry;
        }

        if (entry is null)
            return AreaLookupResult.Unresolved(rawAreaId, mapId, AreaResolutionReason.AreaRowMissing);

        bool mapMatched = mapId < 0 || entry.MapId == mapId;
        AreaResolutionReason reason = mapMatched
            ? AreaResolutionReason.Resolved
            : AreaResolutionReason.MapMismatch;

        AreaEntry? parent = TryGetParent(entry);
        AreaContextEntry contextEntry = ToContextEntry(entry);
        AreaContextEntry? contextParent = parent is null ? null : ToContextEntry(parent);
        AreaDisplayText display = AreaDisplayTextResolver.Resolve(contextEntry, contextParent, source, reason);

        return new AreaLookupResult(
            rawAreaId,
            mapId,
            entry.Id,
            parent?.Id,
            entry.AreaNumber == 0 ? null : entry.AreaNumber,
            string.IsNullOrWhiteSpace(entry.Name) ? null : entry.Name,
            display.ZoneText,
            display.SubzoneText,
            source,
            display.Reason,
            mapMatched);
    }

    public string DescribeLoadContext()
    {
        return $"build={LoadedBuild ?? "unknown"}, locale={LoadedLocale ?? "unknown"}, rows={_rowCount}, indexed={_areas.Count}, primaryKeys={_primaryKeyCount}, idCol={IdColumn ?? "?"}, mapCol={MapColumn ?? "?"}";
    }

    public string DescribeLookup(int areaId, int mapId)
    {
        AreaLookupResult result = ResolveArea(areaId, mapId);
        return $"[AreaTable] Lookup RawArea={areaId} MapId={mapId} layout={IdentityLayout} source={result.Source} reason={result.Reason} zone='{result.ZoneText ?? ""}' subzone='{result.SubzoneText ?? ""}' canonicalId={result.CanonicalAreaId?.ToString() ?? "n/a"} {DescribeLoadContext()}";
    }

    private void RegisterAreaNumber(AreaEntry entry)
    {
        if (entry.AreaNumber == 0)
            return;

        AreaNumberParts areaNumber = AreaNumberParts.FromRaw(entry.AreaNumber);
        _areasByMapAndNumber.TryAdd((entry.MapId, areaNumber), entry);
        if (!_areasByNumber.TryGetValue(areaNumber, out var entries))
        {
            entries = new List<AreaEntry>();
            _areasByNumber[areaNumber] = entries;
        }

        if (entries.All(existing => existing.Id != entry.Id))
            entries.Add(entry);
    }

    private bool TryGetUniqueAreaNumber(AreaNumberParts areaNumber, out AreaEntry entry)
    {
        if (_areasByNumber.TryGetValue(areaNumber, out var entries) && entries.Count == 1)
        {
            entry = entries[0];
            return true;
        }

        entry = null!;
        return false;
    }

    private AreaEntry? TryGetParent(AreaEntry entry)
    {
        if (IdentityLayout == AreaIdentityLayout.PackedAreaNumber
            && entry.ParentAreaNumber != 0
            && _areasByMapAndNumber.TryGetValue(
                (entry.MapId, AreaNumberParts.FromRaw(entry.ParentAreaNumber)), out var packedParent))
            return packedParent;

        if (entry.ParentAreaId != 0 && _areasById.TryGetValue(entry.ParentAreaId, out var directParent))
            return directParent;

        return null;
    }

    private static AreaContextEntry ToContextEntry(AreaEntry entry)
    {
        return new AreaContextEntry(
            entry.Id,
            entry.Name,
            entry.ParentAreaId,
            entry.ParentAreaNumber,
            entry.MapId,
            entry.Flags,
            entry.AreaNumber);
    }

    private void RegisterPrimary(int areaId, AreaEntry entry)
    {
        _areasById[areaId] = entry;
        if (_areas.TryAdd(areaId, entry))
        {
            _primaryKeyCount++;
            return;
        }

        _areas[areaId] = entry;
    }

    private void RegisterAlias(int aliasId, AreaEntry entry)
    {
        if (aliasId == 0 || aliasId == entry.Id)
            return;

        if (_areas.TryGetValue(aliasId, out var existing))
        {
            if (existing.Id != entry.Id)
                _fallbackAliasCollisions++;
            return;
        }

        _areas[aliasId] = entry;
        _fallbackAliasCount++;
    }

    private static string? DetectColumn(ISet<string> availableColumns, params string[] candidates)
    {
        foreach (var col in candidates)
        {
            if (availableColumns.Contains(col))
                return col;
        }

        return null;
    }

    private static T SafeField<T>(dynamic row, string? col, T fallback)
    {
        if (string.IsNullOrWhiteSpace(col))
            return fallback;

        try { return (T)row[col]; }
        catch { return fallback; }
    }

    private static int SafeIntField(dynamic row, string? col, int fallback)
    {
        if (string.IsNullOrWhiteSpace(col))
            return fallback;

        try
        {
            object value = row[col];
            return value switch
            {
                int intValue => intValue,
                uint uintValue => unchecked((int)uintValue),
                long longValue => unchecked((int)longValue),
                ulong ulongValue => unchecked((int)ulongValue),
                short shortValue => shortValue,
                ushort ushortValue => ushortValue,
                byte byteValue => byteValue,
                sbyte sbyteValue => sbyteValue,
                _ => Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture)
            };
        }
        catch
        {
            return fallback;
        }
    }

    private static string FormatColumn(string? col)
    {
        return string.IsNullOrWhiteSpace(col) ? "n/a" : col;
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }
}
