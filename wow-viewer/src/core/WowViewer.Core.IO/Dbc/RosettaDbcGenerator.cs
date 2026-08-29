using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using DBCD;
using DBCD.Providers;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// A map entry to be encoded into <c>Map.dbc</c>.
/// </summary>
public sealed record RosettaMapDbcEntry(
    uint Id,
    string Directory,
    uint InstanceType = 0,
    uint Pvp = 0,
    string MapName = "",
    uint AreaTableId = 0);

/// <summary>
/// An area entry to be encoded into <c>AreaTable.dbc</c>.
/// </summary>
public sealed record RosettaAreaTableDbcEntry(
    uint Id,
    uint ContinentId,
    uint ParentAreaId = 0,
    uint AreaBit = 0,
    uint Flags = 0,
    uint SoundAmbience = 0,
    uint ZoneMusic = 0,
    uint ZoneIntroMusic = 0,
    uint Level = 0,
    string AreaName = "");

/// <summary>
/// Generates and patches authentic client DBC files (<c>Map.dbc</c> and <c>AreaTable.dbc</c>)
/// using DBCD and WoWDBDefs definitions for multi-version client support (Alpha 0.5.3, Vanilla 1.12.1, WotLK 3.3.5, etc.).
/// </summary>
public static class RosettaDbcGenerator
{
    private const uint WdbcMagic = 0x43424457; // "WDBC"

    /// <summary>
    /// Locates the bundled or workspace WoWDBDefs definitions directory.
    /// </summary>
    public static string? TryFindDefinitionsDirectory()
    {
        List<string> startDirectories = [AppContext.BaseDirectory, Directory.GetCurrentDirectory()];
        string? assemblyDir = Path.GetDirectoryName(typeof(RosettaDbcGenerator).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDir))
            startDirectories.Add(assemblyDir);

        foreach (string startDir in startDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            DirectoryInfo? current = new(startDir);
            for (int i = 0; i < 8 && current != null; i++)
            {
                string[] candidates =
                [
                    Path.Combine(current.FullName, "definitions"),
                    Path.Combine(current.FullName, "wow-viewer", "libs", "wowdev", "WoWDBDefs", "definitions"),
                    Path.Combine(current.FullName, "libs", "wowdev", "WoWDBDefs", "definitions"),
                    Path.Combine(current.FullName, "lib", "WoWDBDefs", "definitions"),
                ];

                foreach (string candidate in candidates)
                {
                    if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "Map.dbd")))
                        return candidate;
                }

                current = current.Parent;
            }
        }

        return null;
    }

    /// <summary>
    /// Infers the client build version string from client root path or target format.
    /// </summary>
    public static string InferClientBuild(string? clientRoot, string? format = null, string? dbdDirectory = null)
    {
        string? resolvedDbd = dbdDirectory ?? TryFindDefinitionsDirectory();
        HashSet<string> knownBuilds = new(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrEmpty(resolvedDbd) && Directory.Exists(resolvedDbd))
        {
            string mapDbd = Path.Combine(resolvedDbd, "Map.dbd");
            if (File.Exists(mapDbd))
            {
                foreach (string line in File.ReadLines(mapDbd))
                {
                    string trimmed = line.Trim();
                    if (!trimmed.StartsWith("BUILD ", StringComparison.OrdinalIgnoreCase)) continue;
                    string[] parts = trimmed[6..].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                    foreach (string part in parts)
                    {
                        string[] rangeParts = part.Split('-', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                        foreach (string rp in rangeParts)
                        {
                            if (Regex.IsMatch(rp, @"^\d+\.\d+\.\d+\.\d+$"))
                                knownBuilds.Add(rp);
                        }
                    }
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(clientRoot))
        {
            // 1. Look for full 4-part build in path
            var fullMatches = Regex.Matches(clientRoot, @"(\d+\.\d+\.\d+\.\d+)");
            foreach (Match m in fullMatches)
            {
                string candidate = m.Groups[1].Value;
                if (knownBuilds.Contains(candidate))
                    return candidate;
            }

            // 2. Look for 3-part short version (e.g. 3.3.5 or 0.5.3 or 1.12.1)
            var shortMatches = Regex.Matches(clientRoot, @"(\d+\.\d+\.\d+)");
            foreach (Match m in shortMatches)
            {
                string shortVer = m.Groups[1].Value;
                string? match = knownBuilds.FirstOrDefault(b => b.StartsWith(shortVer + "."));
                if (!string.IsNullOrEmpty(match))
                    return match;
            }

            if (fullMatches.Count > 0)
                return fullMatches[0].Groups[1].Value;
        }

        bool isAlpha = string.Equals(format, "alpha", StringComparison.OrdinalIgnoreCase);
        return isAlpha ? "0.5.3.3368" : "3.3.5.12340";
    }

    /// <summary>
    /// Patches the real existing client DBC files (<c>Map.dbc</c> &amp; <c>AreaTable.dbc</c>) using DBCD and WoWDBDefs,
    /// appending Rosetta maps and areas without losing existing client records or mixing schemas across versions.
    /// </summary>
    public static (int PatchedMapsCount, int PatchedAreasCount) PatchAndSaveClientDbcs(
        IDBCProvider dbcProvider,
        string? buildVersion,
        IReadOnlyList<RosettaMapDbcEntry> mapEntries,
        IReadOnlyList<RosettaAreaTableDbcEntry> areaEntries,
        string outputDirectory,
        string? dbdDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        ArgumentNullException.ThrowIfNull(mapEntries);
        ArgumentNullException.ThrowIfNull(areaEntries);
        ArgumentNullException.ThrowIfNull(outputDirectory);

        string? resolvedDbd = dbdDirectory ?? TryFindDefinitionsDirectory();
        if (string.IsNullOrEmpty(resolvedDbd) || !Directory.Exists(resolvedDbd))
            throw new DirectoryNotFoundException("WoWDBDefs definitions directory was not found. Required for DBC patching.");

        string resolvedBuild = string.IsNullOrWhiteSpace(buildVersion) ? "3.3.5.12340" : buildVersion;

        FilesystemDBDProvider dbdProvider = new(resolvedDbd);
        DBCD.DBCD dbcd = new(dbcProvider, dbdProvider);

        // Load existing client Map.dbc
        IDBCDStorage mapStorage = LoadTableWithFallback(dbcd, "Map", resolvedBuild);

        // Load existing client AreaTable.dbc
        IDBCDStorage areaStorage = LoadTableWithFallback(dbcd, "AreaTable", resolvedBuild);

        // Patch Map table
        HashSet<string> mapCols = new(mapStorage.AvailableColumns, StringComparer.OrdinalIgnoreCase);
        int highestMapId = mapStorage.Keys.DefaultIfEmpty(0).Max();

        foreach (RosettaMapDbcEntry entry in mapEntries)
        {
            int targetId = (int)entry.Id;
            if (mapStorage.ContainsKey(targetId))
            {
                targetId = ++highestMapId;
            }
            else if (targetId > highestMapId)
            {
                highestMapId = targetId;
            }

            DBCDRow row = mapStorage.ConstructRow(targetId);
            SetFieldIfPresent(row, mapCols, "ID", targetId);
            SetFieldIfPresent(row, mapCols, "Directory", entry.Directory);
            SetFieldIfPresent(row, mapCols, "MapName_lang", string.IsNullOrWhiteSpace(entry.MapName) ? entry.Directory : entry.MapName);
            SetFieldIfPresent(row, mapCols, "MapName", string.IsNullOrWhiteSpace(entry.MapName) ? entry.Directory : entry.MapName);
            SetFieldIfPresent(row, mapCols, "InstanceType", (int)entry.InstanceType);
            SetFieldIfPresent(row, mapCols, "PVP", (int)entry.Pvp);
            SetFieldIfPresent(row, mapCols, "IsInMap", 0);
            SetFieldIfPresent(row, mapCols, "AreaTableID", (int)entry.AreaTableId);
            SetFieldIfPresent(row, mapCols, "MinimapIconScale", 0f);
            SetFieldIfPresent(row, mapCols, "TimeOfDayOverride", -1);
            SetFieldIfPresent(row, mapCols, "ExpansionID", 0);
            SetFieldIfPresent(row, mapCols, "LoadingScreenID", 0);

            mapStorage[targetId] = row;
        }

        // Patch AreaTable
        HashSet<string> areaCols = new(areaStorage.AvailableColumns, StringComparer.OrdinalIgnoreCase);
        int highestAreaId = areaStorage.Keys.DefaultIfEmpty(0).Max();

        foreach (RosettaAreaTableDbcEntry entry in areaEntries)
        {
            int targetAreaId = (int)entry.Id;
            if (areaStorage.ContainsKey(targetAreaId))
            {
                targetAreaId = ++highestAreaId;
            }
            else if (targetAreaId > highestAreaId)
            {
                highestAreaId = targetAreaId;
            }

            DBCDRow row = areaStorage.ConstructRow(targetAreaId);
            SetFieldIfPresent(row, areaCols, "ID", targetAreaId);
            SetFieldIfPresent(row, areaCols, "ContinentID", (int)entry.ContinentId);
            SetFieldIfPresent(row, areaCols, "AreaName_lang", entry.AreaName);
            SetFieldIfPresent(row, areaCols, "AreaName", entry.AreaName);
            SetFieldIfPresent(row, areaCols, "ZoneName", entry.AreaName);
            SetFieldIfPresent(row, areaCols, "ParentAreaID", (int)entry.ParentAreaId);
            SetFieldIfPresent(row, areaCols, "ParentAreaNum", (int)entry.ParentAreaId);
            SetFieldIfPresent(row, areaCols, "AreaBit", (int)entry.AreaBit);
            SetFieldIfPresent(row, areaCols, "Flags", (int)entry.Flags);
            SetFieldIfPresent(row, areaCols, "ExplorationLevel", (int)entry.Level);
            SetFieldIfPresent(row, areaCols, "AreaNumber", targetAreaId);

            areaStorage[targetAreaId] = row;
        }

        // Save patched DBC files
        string dbFilesDir = outputDirectory.EndsWith("DBFilesClient", StringComparison.OrdinalIgnoreCase)
            ? outputDirectory
            : Path.Combine(outputDirectory, "DBFilesClient");

        Directory.CreateDirectory(dbFilesDir);
        string mapDbcPath = Path.Combine(dbFilesDir, "Map.dbc");
        string areaTableDbcPath = Path.Combine(dbFilesDir, "AreaTable.dbc");

        mapStorage.Save(mapDbcPath);
        areaStorage.Save(areaTableDbcPath);

        return (mapStorage.Count, areaStorage.Count);
    }

    private static IDBCDStorage LoadTableWithFallback(DBCD.DBCD dbcd, string tableName, string build)
    {
        try
        {
            return dbcd.Load(tableName, build, Locale.None);
        }
        catch
        {
            return dbcd.Load(tableName, build, Locale.EnUS);
        }
    }

    private static void SetFieldIfPresent(DBCDRow row, HashSet<string> availableCols, string fieldName, object value)
    {
        if (!availableCols.Contains(fieldName))
            return;

        try
        {
            object current = row[fieldName];
            if (current is null)
            {
                row[fieldName] = value;
                return;
            }

            if (current is string[] strArr)
            {
                string strVal = value?.ToString() ?? string.Empty;
                for (int i = 0; i < strArr.Length; i++)
                    strArr[i] = i == 0 ? strVal : string.Empty;
                row[fieldName] = strArr;

                string maskField = fieldName + "_mask";
                if (availableCols.Contains(maskField))
                {
                    try { row[maskField] = 0xFFu; } catch { try { row[maskField] = 1u; } catch { } }
                }
                return;
            }

            Type targetType = current.GetType();
            if (targetType == typeof(string))
            {
                row[fieldName] = value?.ToString() ?? string.Empty;
            }
            else if (targetType == typeof(int))
            {
                row[fieldName] = Convert.ToInt32(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(uint))
            {
                row[fieldName] = Convert.ToUInt32(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(short))
            {
                row[fieldName] = Convert.ToInt16(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(ushort))
            {
                row[fieldName] = Convert.ToUInt16(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(byte))
            {
                row[fieldName] = Convert.ToByte(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(sbyte))
            {
                row[fieldName] = Convert.ToSByte(value, CultureInfo.InvariantCulture);
            }
            else if (targetType == typeof(float))
            {
                row[fieldName] = Convert.ToSingle(value, CultureInfo.InvariantCulture);
            }
            else
            {
                row[fieldName] = Convert.ChangeType(value, targetType, CultureInfo.InvariantCulture);
            }
        }
        catch
        {
            // Ignore non-assignable field
        }
    }

    /// <summary>
    /// Builds a binary <c>Map.dbc</c> byte array in Alpha 0.5.3 format (5 uint32 fields per record).
    /// </summary>
    public static byte[] BuildAlphaMapDbc(IReadOnlyList<RosettaMapDbcEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        const uint fieldCount = 5;
        const uint recordSize = fieldCount * 4;

        using MemoryStream stringStream = new();
        stringStream.WriteByte(0); // leading null byte

        var rows = new List<uint[]>(entries.Count);
        foreach (RosettaMapDbcEntry entry in entries)
        {
            uint dirOffset = WriteDbcString(stringStream, entry.Directory);
            uint nameOffset = WriteDbcString(stringStream, string.IsNullOrWhiteSpace(entry.MapName) ? entry.Directory : entry.MapName);

            rows.Add([
                entry.Id,
                dirOffset,
                entry.InstanceType,
                entry.Pvp,
                nameOffset
            ]);
        }

        return AssembleDbc(fieldCount, recordSize, rows, stringStream);
    }

    /// <summary>
    /// Builds a binary <c>AreaTable.dbc</c> byte array in Alpha 0.5.3 format (14 uint32 fields per record).
    /// </summary>
    public static byte[] BuildAlphaAreaTableDbc(IReadOnlyList<RosettaAreaTableDbcEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        const uint fieldCount = 14;
        const uint recordSize = fieldCount * 4;

        using MemoryStream stringStream = new();
        stringStream.WriteByte(0); // leading null byte

        var rows = new List<uint[]>(entries.Count);
        foreach (RosettaAreaTableDbcEntry entry in entries)
        {
            uint nameOffset = WriteDbcString(stringStream, entry.AreaName);

            rows.Add([
                entry.Id,
                entry.ContinentId,
                entry.ParentAreaId,
                entry.AreaBit,
                entry.Flags,
                entry.SoundAmbience,
                entry.ZoneMusic,
                entry.ZoneIntroMusic,
                entry.Level,
                nameOffset,
                0u, 0u, 0u, 0u // string localization / padding fields in 0.5.3
            ]);
        }

        return AssembleDbc(fieldCount, recordSize, rows, stringStream);
    }

    /// <summary>
    /// Writes standalone <c>Map.dbc</c> to the specified file path.
    /// </summary>
    public static void WriteMapDbc(string filePath, IReadOnlyList<RosettaMapDbcEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        byte[] bytes = BuildAlphaMapDbc(entries);
        File.WriteAllBytes(filePath, bytes);
    }

    /// <summary>
    /// Writes standalone <c>AreaTable.dbc</c> to the specified file path.
    /// </summary>
    public static void WriteAreaTableDbc(string filePath, IReadOnlyList<RosettaAreaTableDbcEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        byte[] bytes = BuildAlphaAreaTableDbc(entries);
        File.WriteAllBytes(filePath, bytes);
    }

    private static uint WriteDbcString(MemoryStream stringStream, string value)
    {
        if (string.IsNullOrEmpty(value))
            return 0;

        uint offset = checked((uint)stringStream.Position);
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        stringStream.Write(bytes, 0, bytes.Length);
        stringStream.WriteByte(0);
        return offset;
    }

    private static byte[] AssembleDbc(
        uint fieldCount, uint recordSize, List<uint[]> rows, MemoryStream stringStream)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(WdbcMagic);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(recordSize);
        writer.Write(checked((uint)stringStream.Length));

        foreach (uint[] row in rows)
        {
            foreach (uint val in row)
            {
                writer.Write(val);
            }
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();

        return stream.ToArray();
    }
}
