using System.Collections.Immutable;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using DBCD;
using DBCD.Providers;

namespace WowViewer.Core.IO.Lighting;

/// <summary>
/// Loads the build-scoped Light, LightParams, LightIntBand, LightFloatBand, and
/// LightSkybox database chain through DBCD and WoWDBDefs. This is the native banded
/// Classic contract; it deliberately does not substitute the later flattened LightData table.
/// Database rows are records. Only the Time/Data arrays inside band records are timed samples.
/// </summary>
public sealed class BuildScopedLightDbcProfileResolver
{
    public const int DayCycleUnits = 2880;
    public const int ClearWeatherParamsIndex = 0;
    public const int ColorBandCount = 18;
    public const int FloatBandCount = 6;

    private static readonly string[] RequiredTableNames =
        ["Light", "LightParams", "LightIntBand", "LightFloatBand", "LightSkybox"];

    private static readonly string[] RequiredLightColumns =
    [
        "ContinentID",
        "GameCoords",
        "GameFalloffStart",
        "GameFalloffEnd",
        "LightParamsID",
    ];

    private static readonly string[] RequiredParamsColumns =
    [
        "HighlightSky",
        "LightSkyboxID",
        "Glow",
        "WaterShallowAlpha",
        "WaterDeepAlpha",
        "OceanShallowAlpha",
        "OceanDeepAlpha",
        "Flags",
    ];

    private static readonly string[] RequiredBandColumns = ["Num", "Time", "Data"];

    public LightDbcCatalog Load(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string exactBuild)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        ArgumentException.ThrowIfNullOrWhiteSpace(definitionsDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(exactBuild);

        if (!Directory.Exists(definitionsDirectory))
        {
            throw new DirectoryNotFoundException(
                $"WoWDBDefs definitions directory was not found: {definitionsDirectory}");
        }

        HashingDbcProvider hashingProvider = new(dbcProvider);
        DBCD.DBCD dbcd = new(hashingProvider, new FilesystemDBDProvider(definitionsDirectory));
        IDBCDStorage light = LoadRequiredTable(dbcd, "Light", exactBuild);
        IDBCDStorage lightParams = LoadRequiredTable(dbcd, "LightParams", exactBuild);
        IDBCDStorage intBands = LoadRequiredTable(dbcd, "LightIntBand", exactBuild);
        IDBCDStorage floatBands = LoadRequiredTable(dbcd, "LightFloatBand", exactBuild);
        IDBCDStorage skyboxes = LoadRequiredTable(dbcd, "LightSkybox", exactBuild);

        RequireColumns(light, "Light", exactBuild, RequiredLightColumns);
        RequireColumns(lightParams, "LightParams", exactBuild, RequiredParamsColumns);
        RequireColumns(intBands, "LightIntBand", exactBuild, RequiredBandColumns);
        RequireColumns(floatBands, "LightFloatBand", exactBuild, RequiredBandColumns);
        RequireColumns(skyboxes, "LightSkybox", exactBuild, "Name");

        LightDbcSourceHashes sources = new(
            hashingProvider.Snapshot(RequiredTableNames),
            HashDefinitionFiles(definitionsDirectory, RequiredTableNames));

        return LightDbcCatalog.Create(
            exactBuild,
            ParseZones(light, exactBuild),
            ParseParams(lightParams, exactBuild),
            ParseIntBands(intBands, exactBuild),
            ParseFloatBands(floatBands, exactBuild),
            ParseSkyboxes(skyboxes, exactBuild),
            sources);
    }

    /// <summary>
    /// LightIntBand uses a dense eighteen-record block for each LightParams record.
    /// The exact record must exist in the loaded table; callers never receive an ordinal fallback.
    /// </summary>
    public static int GetIntBandRecordId(int lightParamsRecordId, LightDbcColorBand band)
    {
        if (lightParamsRecordId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lightParamsRecordId));
        }

        int index = (int)band;
        if ((uint)index >= ColorBandCount)
        {
            throw new ArgumentOutOfRangeException(nameof(band));
        }

        return checked(((lightParamsRecordId - 1) * ColorBandCount) + index + 1);
    }

    /// <summary>
    /// LightFloatBand uses a dense six-record block for each LightParams record.
    /// The exact record must exist in the loaded table; callers never receive an ordinal fallback.
    /// </summary>
    public static int GetFloatBandRecordId(int lightParamsRecordId, LightDbcFloatBand band)
    {
        if (lightParamsRecordId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lightParamsRecordId));
        }

        int index = (int)band;
        if ((uint)index >= FloatBandCount)
        {
            throw new ArgumentOutOfRangeException(nameof(band));
        }

        return checked(((lightParamsRecordId - 1) * FloatBandCount) + index + 1);
    }

    /// <summary>
    /// The integer is 0xXXRRGGBB. On little-endian disk the bytes are B, G, R, X,
    /// which is the BGRX layout documented for LightIntBand/LIT color samples.
    /// </summary>
    public static Vector3 UnpackBgrx(int packedBgrx)
    {
        uint packed = unchecked((uint)packedBgrx);
        return new Vector3(
            ((packed >> 16) & 0xffu) / 255f,
            ((packed >> 8) & 0xffu) / 255f,
            (packed & 0xffu) / 255f);
    }

    public static Vector3 EvaluateColorBand(LightDbcIntBandRecord band, int time)
    {
        ArgumentNullException.ThrowIfNull(band);
        if (band.Samples.IsDefaultOrEmpty)
        {
            return Vector3.Zero;
        }

        (LightDbcColorSample a, LightDbcColorSample b, float blend) =
            FindTimedPair(band.Samples, time, static sample => sample.Time);
        return Vector3.Lerp(UnpackBgrx(a.PackedBgrx), UnpackBgrx(b.PackedBgrx), blend);
    }

    public static float EvaluateFloatBand(LightDbcFloatBandRecord band, int time)
    {
        ArgumentNullException.ThrowIfNull(band);
        if (band.Samples.IsDefaultOrEmpty)
        {
            return 0f;
        }

        (LightDbcFloatSample a, LightDbcFloatSample b, float blend) =
            FindTimedPair(band.Samples, time, static sample => sample.Time);
        return a.Value + ((b.Value - a.Value) * blend);
    }

    public static int NormalizeTime(int time)
    {
        int normalized = time % DayCycleUnits;
        return normalized < 0 ? normalized + DayCycleUnits : normalized;
    }

    private static (T A, T B, float Blend) FindTimedPair<T>(
        ImmutableArray<T> samples,
        int requestedTime,
        Func<T, int> getTime)
    {
        if (samples.Length == 1)
        {
            return (samples[0], samples[0], 0f);
        }

        T[] ordered = samples
            .OrderBy(getTime)
            .ToArray();
        int time = NormalizeTime(requestedTime);

        for (int i = 0; i < ordered.Length; i++)
        {
            if (getTime(ordered[i]) == time)
            {
                return (ordered[i], ordered[i], 0f);
            }
        }

        int upper = Array.FindIndex(ordered, sample => getTime(sample) > time);
        T a;
        T b;
        int aTime;
        int bTime;
        int evaluationTime;

        if (upper < 0)
        {
            a = ordered[^1];
            b = ordered[0];
            aTime = getTime(a);
            bTime = getTime(b) + DayCycleUnits;
            evaluationTime = time;
        }
        else if (upper == 0)
        {
            a = ordered[^1];
            b = ordered[0];
            aTime = getTime(a);
            bTime = getTime(b) + DayCycleUnits;
            evaluationTime = time + DayCycleUnits;
        }
        else
        {
            a = ordered[upper - 1];
            b = ordered[upper];
            aTime = getTime(a);
            bTime = getTime(b);
            evaluationTime = time;
        }

        int range = bTime - aTime;
        float blend = range <= 0 ? 0f : Math.Clamp((float)(evaluationTime - aTime) / range, 0f, 1f);
        return (a, b, blend);
    }

    private static IDBCDStorage LoadRequiredTable(DBCD.DBCD dbcd, string tableName, string build)
    {
        Exception? localeFailure = null;
        try
        {
            return dbcd.Load(tableName, build, Locale.EnUS);
        }
        catch (Exception ex)
        {
            localeFailure = ex;
        }

        try
        {
            return dbcd.Load(tableName, build, Locale.None);
        }
        catch (Exception ex)
        {
            throw new LightDbcLoadException(
                $"DBCD could not load required table '{tableName}' for exact build '{build}' " +
                "using the supplied WoWDBDefs definitions.",
                new AggregateException(localeFailure!, ex));
        }
    }

    private static void RequireColumns(
        IDBCDStorage storage,
        string tableName,
        string build,
        params string[] requiredColumns)
    {
        HashSet<string> available = storage.AvailableColumns.ToHashSet(StringComparer.Ordinal);
        string[] missing = requiredColumns.Where(column => !available.Contains(column)).ToArray();
        if (missing.Length > 0)
        {
            throw new LightDbcLoadException(
                $"WoWDBDefs schema for table '{tableName}' and exact build '{build}' is missing " +
                $"required column(s): {string.Join(", ", missing)}. Available: {string.Join(", ", available)}.");
        }
    }

    private static IEnumerable<LightDbcZoneRecord> ParseZones(IDBCDStorage storage, string build)
    {
        foreach (DBCDRow row in storage.Values)
        {
            float[] coords = GetRequiredArray<float>(row, "GameCoords", "Light", build);
            if (coords.Length != 3)
            {
                throw InvalidRecord("Light", row.ID, build, "GameCoords must contain exactly three values.");
            }

            int[] paramsIds = GetRequiredArray<int>(row, "LightParamsID", "Light", build);
            if (paramsIds.Length <= ClearWeatherParamsIndex)
            {
                throw InvalidRecord("Light", row.ID, build, "LightParamsID has no clear-weather slot.");
            }

            yield return new LightDbcZoneRecord(
                row.ID,
                GetRequiredInt(row, "ContinentID", "Light", build),
                new Vector3(coords[0], coords[1], coords[2]),
                GetRequiredFloat(row, "GameFalloffStart", "Light", build),
                GetRequiredFloat(row, "GameFalloffEnd", "Light", build),
                paramsIds.ToImmutableArray());
        }
    }

    private static IEnumerable<LightDbcParamsRecord> ParseParams(IDBCDStorage storage, string build)
    {
        foreach (DBCDRow row in storage.Values)
        {
            yield return new LightDbcParamsRecord(
                row.ID,
                GetRequiredInt(row, "HighlightSky", "LightParams", build),
                GetRequiredInt(row, "LightSkyboxID", "LightParams", build),
                GetRequiredFloat(row, "Glow", "LightParams", build),
                GetRequiredFloat(row, "WaterShallowAlpha", "LightParams", build),
                GetRequiredFloat(row, "WaterDeepAlpha", "LightParams", build),
                GetRequiredFloat(row, "OceanShallowAlpha", "LightParams", build),
                GetRequiredFloat(row, "OceanDeepAlpha", "LightParams", build),
                GetRequiredInt(row, "Flags", "LightParams", build));
        }
    }

    private static IEnumerable<LightDbcIntBandRecord> ParseIntBands(IDBCDStorage storage, string build)
    {
        foreach (DBCDRow row in storage.Values)
        {
            int count = GetRequiredInt(row, "Num", "LightIntBand", build);
            int[] times = GetRequiredArray<int>(row, "Time", "LightIntBand", build);
            int[] data = GetRequiredArray<int>(row, "Data", "LightIntBand", build);
            ValidateBandCount("LightIntBand", row.ID, build, count, times.Length, data.Length);

            ImmutableArray<LightDbcColorSample>.Builder samples =
                ImmutableArray.CreateBuilder<LightDbcColorSample>(count);
            for (int i = 0; i < count; i++)
            {
                samples.Add(new LightDbcColorSample(times[i], data[i]));
            }

            yield return new LightDbcIntBandRecord(row.ID, samples.MoveToImmutable());
        }
    }

    private static IEnumerable<LightDbcFloatBandRecord> ParseFloatBands(IDBCDStorage storage, string build)
    {
        foreach (DBCDRow row in storage.Values)
        {
            int count = GetRequiredInt(row, "Num", "LightFloatBand", build);
            int[] times = GetRequiredArray<int>(row, "Time", "LightFloatBand", build);
            float[] data = GetRequiredArray<float>(row, "Data", "LightFloatBand", build);
            ValidateBandCount("LightFloatBand", row.ID, build, count, times.Length, data.Length);

            ImmutableArray<LightDbcFloatSample>.Builder samples =
                ImmutableArray.CreateBuilder<LightDbcFloatSample>(count);
            for (int i = 0; i < count; i++)
            {
                samples.Add(new LightDbcFloatSample(times[i], data[i]));
            }

            yield return new LightDbcFloatBandRecord(row.ID, samples.MoveToImmutable());
        }
    }

    private static IEnumerable<LightDbcSkyboxRecord> ParseSkyboxes(IDBCDStorage storage, string build)
    {
        bool hasFlags = storage.AvailableColumns.Contains("Flags", StringComparer.Ordinal);
        foreach (DBCDRow row in storage.Values)
        {
            yield return new LightDbcSkyboxRecord(
                row.ID,
                GetRequiredString(row, "Name", "LightSkybox", build),
                hasFlags ? GetRequiredInt(row, "Flags", "LightSkybox", build) : 0);
        }
    }

    private static void ValidateBandCount(
        string table,
        int recordId,
        string build,
        int count,
        int timeLength,
        int dataLength)
    {
        if (count < 0 || count > timeLength || count > dataLength)
        {
            throw InvalidRecord(
                table,
                recordId,
                build,
                $"Num={count} exceeds Time/Data capacity ({timeLength}/{dataLength}).");
        }
    }

    private static int GetRequiredInt(DBCDRow row, string column, string table, string build)
    {
        object value = GetRequiredValue(row, column, table, build);
        try
        {
            return value switch
            {
                uint typed => unchecked((int)typed),
                ulong typed => unchecked((int)typed),
                _ => Convert.ToInt32(value, CultureInfo.InvariantCulture),
            };
        }
        catch (Exception ex)
        {
            throw InvalidRecord(table, row.ID, build, $"Column '{column}' is not an integer.", ex);
        }
    }

    private static float GetRequiredFloat(DBCDRow row, string column, string table, string build)
    {
        object value = GetRequiredValue(row, column, table, build);
        try
        {
            return Convert.ToSingle(value, CultureInfo.InvariantCulture);
        }
        catch (Exception ex)
        {
            throw InvalidRecord(table, row.ID, build, $"Column '{column}' is not a float.", ex);
        }
    }

    private static string GetRequiredString(DBCDRow row, string column, string table, string build)
    {
        object value = GetRequiredValue(row, column, table, build);
        if (value is string text)
        {
            return text;
        }

        throw InvalidRecord(table, row.ID, build, $"Column '{column}' is not a string.");
    }

    private static T[] GetRequiredArray<T>(DBCDRow row, string column, string table, string build)
    {
        object value = GetRequiredValue(row, column, table, build);
        if (value is not Array array)
        {
            throw InvalidRecord(table, row.ID, build, $"Column '{column}' is not an array.");
        }

        T[] result = new T[array.Length];
        try
        {
            for (int i = 0; i < array.Length; i++)
            {
                object? item = array.GetValue(i);
                if (item is null)
                {
                    throw new InvalidCastException("Array element is null.");
                }

                result[i] = typeof(T) == typeof(int)
                    ? (T)(object)(item switch
                    {
                        uint typed => unchecked((int)typed),
                        ulong typed => unchecked((int)typed),
                        _ => Convert.ToInt32(item, CultureInfo.InvariantCulture),
                    })
                    : (T)Convert.ChangeType(item, typeof(T), CultureInfo.InvariantCulture);
            }

            return result;
        }
        catch (Exception ex)
        {
            throw InvalidRecord(table, row.ID, build, $"Column '{column}' has incompatible array data.", ex);
        }
    }

    private static object GetRequiredValue(DBCDRow row, string column, string table, string build)
    {
        try
        {
            return row[column];
        }
        catch (Exception ex)
        {
            throw InvalidRecord(table, row.ID, build, $"Column '{column}' could not be read.", ex);
        }
    }

    private static LightDbcLoadException InvalidRecord(
        string table,
        int recordId,
        string build,
        string detail,
        Exception? inner = null)
    {
        string message = $"Invalid {table} record {recordId} for exact build '{build}': {detail}";
        return inner is null ? new LightDbcLoadException(message) : new LightDbcLoadException(message, inner);
    }

    private static ImmutableDictionary<string, string> HashDefinitionFiles(
        string definitionsDirectory,
        IEnumerable<string> tableNames)
    {
        ImmutableDictionary<string, string>.Builder hashes =
            ImmutableDictionary.CreateBuilder<string, string>(StringComparer.Ordinal);
        foreach (string tableName in tableNames)
        {
            string path = Path.Combine(definitionsDirectory, $"{tableName}.dbd");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Required WoWDBDefs definition was not found for table '{tableName}'.",
                    path);
            }

            hashes.Add(tableName, Sha256(File.ReadAllBytes(path)));
        }

        return hashes.ToImmutable();
    }

    private static string Sha256(ReadOnlySpan<byte> bytes) =>
        Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class HashingDbcProvider(IDBCProvider inner) : IDBCProvider
    {
        private readonly Dictionary<string, string> _hashes = new(StringComparer.Ordinal);

        public Stream StreamForTableName(string tableName, string build)
        {
            using Stream source = inner.StreamForTableName(tableName, build);
            using MemoryStream buffer = new();
            source.CopyTo(buffer);
            byte[] bytes = buffer.ToArray();
            _hashes[tableName] = Sha256(bytes);
            return new MemoryStream(bytes, writable: false);
        }

        public ImmutableDictionary<string, string> Snapshot(IEnumerable<string> requiredTables)
        {
            ImmutableDictionary<string, string>.Builder result =
                ImmutableDictionary.CreateBuilder<string, string>(StringComparer.Ordinal);
            foreach (string table in requiredTables)
            {
                if (!_hashes.TryGetValue(table, out string? hash))
                {
                    throw new LightDbcLoadException(
                        $"Required table '{table}' was parsed without captured source-byte evidence.");
                }

                result.Add(table, hash);
            }

            return result.ToImmutable();
        }
    }
}
