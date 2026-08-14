using DBCD;
using DBCD.Providers;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

public sealed class AlphaAreaAudioCatalogReader
{
    public const string DefaultBuildVersion = "0.5.3.3368";

    public AlphaAreaAudioCatalog? Load(
        IEnumerable<string> searchPaths,
        IArchiveReader? archiveReader = null,
        string buildVersion = DefaultBuildVersion)
    {
        ArgumentNullException.ThrowIfNull(searchPaths);

        byte[]? areaTableData = TryReadFromDisk(searchPaths, "AreaTable")
            ?? TryReadFromArchive(archiveReader, "AreaTable");
        byte[]? areaMidiData = TryReadFromDisk(searchPaths, "AreaMIDIAmbiences")
            ?? TryReadFromArchive(archiveReader, "AreaMIDIAmbiences");

        if (areaTableData is null)
        {
            return null;
        }

        string? definitionsDirectory = TryFindDefinitionsDirectory(searchPaths.ToArray());
        if (string.IsNullOrWhiteSpace(definitionsDirectory))
        {
            throw new DirectoryNotFoundException("WoWDBDefs definitions directory was not found.");
        }

        IDBCDStorage areaTable = LoadTableWithDbcd(
            new InMemoryDbcProvider(("AreaTable", areaTableData)),
            definitionsDirectory,
            "AreaTable",
            buildVersion);
        Dictionary<int, AlphaAreaMidiAmbience> midiAmbiences = areaMidiData is null
            ? []
            : ParseMidiAmbiences(LoadTableWithDbcd(
                new InMemoryDbcProvider(("AreaMIDIAmbiences", areaMidiData)),
                definitionsDirectory,
                "AreaMIDIAmbiences",
                buildVersion));

        return new AlphaAreaAudioCatalog(ParseAreas(areaTable, buildVersion), midiAmbiences);
    }

    /// <summary>
    /// Load the area-audio tables directly from the active DBC provider. The
    /// viewer uses this overload so area music follows the selected client
    /// build rather than the reader's historical archive-probe default.
    /// </summary>
    public AlphaAreaAudioCatalog Load(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string buildVersion)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        if (string.IsNullOrWhiteSpace(definitionsDirectory))
            throw new ArgumentException("A WoWDBDefs definitions directory is required.", nameof(definitionsDirectory));
        if (string.IsNullOrWhiteSpace(buildVersion))
            throw new ArgumentException("An exact client build is required.", nameof(buildVersion));

        IDBCDStorage areaTable = LoadTableWithDbcd(dbcProvider, definitionsDirectory, "AreaTable", buildVersion);
        IDBCDStorage? areaMidi = TryLoadOptionalTable(dbcProvider, definitionsDirectory, "AreaMIDIAmbiences", buildVersion);
        return new AlphaAreaAudioCatalog(
            ParseAreas(areaTable, buildVersion),
            areaMidi is null ? new Dictionary<int, AlphaAreaMidiAmbience>() : ParseMidiAmbiences(areaMidi));
    }

    private static IDBCDStorage? TryLoadOptionalTable(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string tableName,
        string buildVersion)
    {
        try
        {
            return LoadTableWithDbcd(dbcProvider, definitionsDirectory, tableName, buildVersion);
        }
        catch
        {
            return null;
        }
    }

    private static byte[]? TryReadFromArchive(IArchiveReader? archiveReader, string tableName)
    {
        return archiveReader is null ? null : DbClientFileReader.TryReadTable(archiveReader, tableName);
    }

    private static byte[]? TryReadFromDisk(IEnumerable<string> searchPaths, string tableName)
    {
        foreach (string basePath in searchPaths.Where(static path => !string.IsNullOrWhiteSpace(path)))
        {
            foreach (string relativePath in DbClientFileReader.EnumerateTablePaths(tableName))
            {
                string candidate = Path.Combine(
                    basePath,
                    relativePath
                        .Replace('\\', Path.DirectorySeparatorChar)
                        .Replace('/', Path.DirectorySeparatorChar));

                if (File.Exists(candidate))
                {
                    return File.ReadAllBytes(candidate);
                }
            }
        }

        return null;
    }

    private static IDBCDStorage LoadTableWithDbcd(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string tableName,
        string buildVersion)
    {
        FilesystemDBDProvider dbdProvider = new(definitionsDirectory);
        DBCD.DBCD dbcd = new(dbcProvider, dbdProvider);

        try
        {
            return dbcd.Load(tableName, buildVersion, Locale.EnUS);
        }
        catch
        {
            return dbcd.Load(tableName, buildVersion, Locale.None);
        }
    }

    private static Dictionary<int, AlphaAreaRecord> ParseAreas(IDBCDStorage storage, string buildVersion)
    {
        Dictionary<int, AlphaAreaRecord> areas = [];
        bool alphaAreaNumberLayout = buildVersion.StartsWith("0.5.", StringComparison.OrdinalIgnoreCase);

        foreach (DBCDRow row in storage.Values)
        {
            int areaNumber = GetPackedIntField(row, "AreaNumber");
            int id = GetIntField(row, "ID") ?? 0;
            if (id == 0 && areaNumber != 0)
                id = areaNumber;
            if (id == 0)
            {
                continue;
            }

            AlphaAreaRecord entry = new(
                id,
                GetIntField(row, "ContinentID") ?? 0,
                alphaAreaNumberLayout
                    ? GetPackedIntField(row, "ParentAreaNum")
                    : GetIntField(row, "ParentAreaID", "ParentAreaNum") ?? 0,
                GetStringField(row, "AreaName_lang", "AreaName", "ZoneName") ?? string.Empty,
                GetIntField(row, "MIDIAmbience") ?? 0,
                GetIntField(row, "MIDIAmbienceUnderwater") ?? 0,
                GetIntField(row, "ZoneMusic") ?? 0,
                GetIntField(row, "IntroSound") ?? 0,
                GetIntField(row, "IntroPriority") ?? 0,
                areaNumber,
                GetPackedIntField(row, "ParentAreaNum"));

            areas[id] = entry;
        }

        return areas;
    }

    private static Dictionary<int, AlphaAreaMidiAmbience> ParseMidiAmbiences(IDBCDStorage storage)
    {
        Dictionary<int, AlphaAreaMidiAmbience> midiAmbiences = [];

        foreach (DBCDRow row in storage.Values)
        {
            int id = GetIntField(row, "ID") ?? 0;
            if (id <= 0)
            {
                continue;
            }

            AlphaAreaMidiAmbience entry = new(
                id,
                GetStringField(row, "DaySequence") ?? string.Empty,
                GetStringField(row, "NightSequence") ?? string.Empty,
                GetStringField(row, "DLSFile") ?? string.Empty,
                GetFloatField(row, "Volume") ?? 0f);

            midiAmbiences[id] = entry;
        }

        return midiAmbiences;
    }

    private static int? GetIntField(DBCDRow row, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            try
            {
                object value = row[fieldName];
                if (value is int intValue)
                {
                    return intValue;
                }

                if (value is uint uintValue)
                {
                    return unchecked((int)uintValue);
                }

                if (value is long longValue)
                {
                    return unchecked((int)longValue);
                }

                if (value is short shortValue)
                {
                    return shortValue;
                }

                if (value is ushort ushortValue)
                {
                    return ushortValue;
                }

                if (value is byte byteValue)
                {
                    return byteValue;
                }

                if (value is sbyte sbyteValue)
                {
                    return sbyteValue;
                }

                if (value != null)
                {
                    return Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
                }
            }
            catch
            {
            }
        }

        return null;
    }

    private static int GetPackedIntField(DBCDRow row, params string[] fieldNames)
    {
        return GetIntField(row, fieldNames) ?? 0;
    }

    private static float? GetFloatField(DBCDRow row, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            try
            {
                object value = row[fieldName];
                if (value is float floatValue)
                {
                    return floatValue;
                }

                if (value is double doubleValue)
                {
                    return (float)doubleValue;
                }

                if (value != null)
                {
                    return Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture);
                }
            }
            catch
            {
            }
        }

        return null;
    }

    private static string? GetStringField(DBCDRow row, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            try
            {
                object value = row[fieldName];
                if (value is string stringValue)
                {
                    return stringValue;
                }

                if (value != null)
                {
                    return value.ToString();
                }
            }
            catch
            {
            }
        }

        return null;
    }

    private static string? TryFindDefinitionsDirectory(params string?[] paths)
    {
        List<string> startDirectories = [];
        foreach (string? path in paths)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            string? directory = Directory.Exists(path) ? path : Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                startDirectories.Add(directory);
            }
        }

        startDirectories.Add(Directory.GetCurrentDirectory());

        string? assemblyDirectory = Path.GetDirectoryName(typeof(AlphaAreaAudioCatalogReader).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDirectory))
        {
            startDirectories.Add(assemblyDirectory);
        }

        foreach (string startDirectory in startDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            DirectoryInfo? current = new(startDirectory);
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
                    if (Directory.Exists(candidate))
                    {
                        return candidate;
                    }
                }

                current = current.Parent;
            }
        }

        return null;
    }

    private sealed class InMemoryDbcProvider(params (string TableName, byte[] Data)[] tables) : IDBCProvider
    {
        private readonly Dictionary<string, byte[]> _tables = tables.ToDictionary(
            static entry => entry.TableName,
            static entry => entry.Data,
            StringComparer.OrdinalIgnoreCase);

        public Stream StreamForTableName(string tableName, string build)
        {
            if (!_tables.TryGetValue(tableName, out byte[]? data))
            {
                throw new FileNotFoundException($"Table {tableName} was not supplied to the in-memory DBC provider.");
            }

            return new MemoryStream(data, writable: false);
        }
    }
}
