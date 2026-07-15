using System.Globalization;
using System.Numerics;
using System.Text.Json;
using DBCD.Providers;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Lighting;

internal static class LightDbcProfileCommand
{
    public const string Schema = "wowviewer.light-dbc-profile.v1";
    public const float DefaultMapOrigin = 17066.666f;

    private static readonly string[] TableNames =
        ["Light", "LightParams", "LightIntBand", "LightFloatBand", "LightSkybox"];

    private static readonly HashSet<string> KnownOptions = new(StringComparer.OrdinalIgnoreCase)
    {
        "--archive-root", "-r",
        "--build", "-b",
        "--map-id", "-m",
        "--world-position",
        "--renderer-position",
        "--map-origin",
        "--game-time", "-t",
        "--dbd-dir",
        "--output", "-o",
        "--listfile", "-l",
        "--cache-key", "-k",
        "--cache-dir", "-d",
    };

    public static void Execute(string[] args, ArchiveCatalogBootstrapOptions archiveOptions)
    {
        ArgumentNullException.ThrowIfNull(args);
        ValidateOptionShape(args);

        string archiveRoot = RequireDirectory(
            RequireSingleOption(args, "--archive-root", "-r"),
            "archive root");
        string exactBuild = RequireSingleOption(args, "--build", "-b");
        int mapId = ParseMapId(RequireSingleOption(args, "--map-id", "-m"));
        float mapOrigin = ParseFiniteFloat(
            GetSingleOption(args, "--map-origin") ?? DefaultMapOrigin.ToString("R", CultureInfo.InvariantCulture),
            "--map-origin");
        LightDbcCoordinateQuery coordinate = ParseCoordinateQuery(args, mapOrigin);
        IReadOnlyList<LightDbcRequestedTime> times = ParseGameTimes(args);
        string definitionsDirectory = ResolveDefinitionsDirectory(GetSingleOption(args, "--dbd-dir"));
        string? output = GetSingleOption(args, "--output", "-o");

        ArchiveDbcProvider provider = new(archiveRoot, archiveOptions);
        LightDbcCatalog catalog = new BuildScopedLightDbcProfileResolver().Load(
            provider,
            definitionsDirectory,
            exactBuild);

        LightDbcProfileArtifact artifact = BuildArtifact(
            catalog,
            archiveRoot,
            definitionsDirectory,
            mapId,
            coordinate,
            times);
        string json = JsonSerializer.Serialize(
            artifact,
            new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            });

        if (string.IsNullOrWhiteSpace(output) || output == "-")
        {
            Console.WriteLine(json);
            return;
        }

        string outputPath = Path.GetFullPath(output);
        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllText(outputPath, json);
        Console.WriteLine($"Wrote {outputPath}");
    }

    internal static LightDbcProfileArtifact BuildArtifact(
        LightDbcCatalog catalog,
        string archiveRoot,
        string definitionsDirectory,
        int mapId,
        LightDbcCoordinateQuery coordinate,
        IReadOnlyList<LightDbcRequestedTime> times)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(times);
        if (times.Count == 0)
            throw new ArgumentException("At least one game time is required.", nameof(times));

        LightDbcTableSourceEvidence[] sourceTables = TableNames
            .Select(tableName => new LightDbcTableSourceEvidence(
                tableName,
                $"DBFilesClient/{tableName}.dbc",
                RequireHash(catalog.SourceHashes.DatabaseTableSha256, tableName, "DBC table"),
                Path.GetFullPath(Path.Combine(definitionsDirectory, $"{tableName}.dbd")),
                RequireHash(catalog.SourceHashes.WowDbDefsDefinitionSha256, tableName, "WoWDBDefs definition")))
            .ToArray();

        LightDbcSampleArtifact[] samples = times
            .Select(time => BuildSample(catalog, mapId, coordinate, time))
            .ToArray();

        return new LightDbcProfileArtifact(
            Schema,
            new LightDbcSourceArtifact(
                catalog.Build,
                Path.GetFullPath(archiveRoot),
                Path.GetFullPath(definitionsDirectory),
                sourceTables),
            new LightDbcCatalogArtifact(
                catalog.Zones.Length,
                catalog.LightParamsRecordCount,
                catalog.LightIntBandRecordCount,
                catalog.LightFloatBandRecordCount,
                catalog.LightSkyboxRecordCount,
                catalog.TimedSampleCount),
            new LightDbcQueryArtifact(mapId, coordinate, times),
            samples);
    }

    private static LightDbcSampleArtifact BuildSample(
        LightDbcCatalog catalog,
        int mapId,
        LightDbcCoordinateQuery coordinate,
        LightDbcRequestedTime requestedTime)
    {
        LightDbcEvaluation evaluation = catalog.EvaluateClearWeather(
            mapId,
            ToNumerics(coordinate.WorldPosition),
            requestedTime.ResolvedRawTime);
        LightDbcEvaluationEvidence evidence = evaluation.Evidence;

        LightDbcZoneBlendArtifact? global = BuildZoneBlend(
            catalog,
            evidence.GlobalProfile,
            evidence.GlobalProfile is null ? 0f : 1f - evidence.LocalWeight,
            coordinate.MapOrigin);
        LightDbcZoneBlendArtifact? local = BuildZoneBlend(
            catalog,
            evidence.LocalProfile,
            evidence.LocalProfile is null ? 0f : evidence.LocalWeight,
            coordinate.MapOrigin);

        LightDbcColorBandArtifact[] colors = Enum.GetValues<LightDbcColorBand>()
            .Select(band =>
            {
                Vector3 value = evaluation[band];
                return new LightDbcColorBandArtifact(
                    (int)band,
                    band.ToString(),
                    new LightDbcRgbArtifact(value.X, value.Y, value.Z));
            })
            .ToArray();
        LightDbcFloatBandArtifact[] floats = Enum.GetValues<LightDbcFloatBand>()
            .Select(band => new LightDbcFloatBandArtifact((int)band, band.ToString(), evaluation[band]))
            .ToArray();

        return new LightDbcSampleArtifact(
            requestedTime,
            evidence.NormalizedTime,
            evidence.NormalizedTime / (float)BuildScopedLightDbcProfileResolver.DayCycleUnits,
            new LightDbcSpatialBlendArtifact(global, local),
            colors,
            floats,
            new LightDbcParamsArtifact(
                evaluation.PrimaryParams.RecordId,
                evaluation.PrimaryParams.HighlightSky,
                evaluation.PrimaryParams.LightSkyboxId,
                evaluation.PrimaryParams.Glow,
                evaluation.PrimaryParams.WaterShallowAlpha,
                evaluation.PrimaryParams.WaterDeepAlpha,
                evaluation.PrimaryParams.OceanShallowAlpha,
                evaluation.PrimaryParams.OceanDeepAlpha,
                evaluation.PrimaryParams.Flags),
            evaluation.PrimarySkybox is null
                ? null
                : new LightDbcSkyboxArtifact(
                    evaluation.PrimarySkybox.RecordId,
                    evaluation.PrimarySkybox.Name,
                    evaluation.PrimarySkybox.Flags));
    }

    private static LightDbcZoneBlendArtifact? BuildZoneBlend(
        LightDbcCatalog catalog,
        LightDbcProfileEvidence? profile,
        float weight,
        float mapOrigin)
    {
        if (profile is null)
            return null;

        LightDbcZoneRecord zone = catalog.Zones.Single(candidate => candidate.RecordId == profile.LightRecordId);
        return new LightDbcZoneBlendArtifact(
            weight,
            new LightDbcZoneArtifact(
                zone.RecordId,
                zone.ContinentId,
                ToVector(zone.RawGameCoordsXzy),
                ToVector(zone.WorldPosition),
                ToVector(zone.ToRendererPosition(mapOrigin)),
                zone.RawFalloffStart,
                zone.RawFalloffEnd,
                zone.FalloffStart,
                zone.FalloffEnd,
                zone.LightParamsIds,
                zone.ClearWeatherLightParamsId),
            new LightDbcProfileJoinArtifact(
                profile.ClearWeatherParamsIndex,
                profile.LightParamsRecordId,
                profile.LightSkyboxRecordId,
                profile.LightIntBandRecordIds,
                profile.LightFloatBandRecordIds));
    }

    private static string RequireHash(
        IReadOnlyDictionary<string, string> hashes,
        string tableName,
        string sourceKind)
    {
        if (!hashes.TryGetValue(tableName, out string? hash) || string.IsNullOrWhiteSpace(hash))
            throw new InvalidDataException($"Missing {sourceKind} SHA-256 evidence for '{tableName}'.");

        return hash;
    }

    private static LightDbcCoordinateQuery ParseCoordinateQuery(string[] args, float mapOrigin)
    {
        string? worldText = GetSingleOption(args, "--world-position");
        string? rendererText = GetSingleOption(args, "--renderer-position");
        if (string.IsNullOrWhiteSpace(worldText) == string.IsNullOrWhiteSpace(rendererText))
        {
            throw new ArgumentException(
                "Provide exactly one of --world-position <x,y,z> or --renderer-position <x,y,z>.");
        }

        if (!string.IsNullOrWhiteSpace(worldText))
        {
            Vector3 world = ParseVector3(worldText, "--world-position");
            Vector3 renderer = WorldToRenderer(world, mapOrigin);
            return new LightDbcCoordinateQuery(
                "world",
                ToVector(world),
                mapOrigin,
                ToVector(world),
                ToVector(renderer));
        }

        Vector3 rendererInput = ParseVector3(rendererText!, "--renderer-position");
        Vector3 convertedWorld = RendererToWorld(rendererInput, mapOrigin);
        return new LightDbcCoordinateQuery(
            "renderer",
            ToVector(rendererInput),
            mapOrigin,
            ToVector(convertedWorld),
            ToVector(rendererInput));
    }

    private static IReadOnlyList<LightDbcRequestedTime> ParseGameTimes(string[] args)
    {
        string[] tokens = GetOptionValues(args, "--game-time", "-t")
            .SelectMany(value => value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .ToArray();
        if (tokens.Length == 0)
            tokens = ["normalized:0.35"];

        return tokens.Select(ParseGameTime).ToArray();
    }

    private static LightDbcRequestedTime ParseGameTime(string token)
    {
        string input = token.Trim();
        string unit;
        string numberText;
        int separator = input.IndexOf(':');
        if (separator >= 0)
        {
            unit = input[..separator].Trim().ToLowerInvariant();
            numberText = input[(separator + 1)..].Trim();
        }
        else
        {
            numberText = input;
            float inferred = ParseFiniteFloat(numberText, "--game-time");
            unit = inferred is >= 0f and <= 1f ? "normalized" : "raw";
        }

        float value = ParseFiniteFloat(numberText, "--game-time");
        bool normalized = unit is "normalized" or "norm" or "n";
        bool raw = unit is "raw" or "units" or "u";
        if (!normalized && !raw)
        {
            throw new ArgumentException(
                $"Unknown --game-time unit '{unit}'. Use normalized:<0..1> or raw:<0..2880>.");
        }

        if (normalized && value is < 0f or > 1f)
            throw new ArgumentOutOfRangeException("--game-time", value, "Normalized game time must be within 0..1.");
        if (raw && value is < 0f or > BuildScopedLightDbcProfileResolver.DayCycleUnits)
            throw new ArgumentOutOfRangeException("--game-time", value, "Raw game time must be within 0..2880.");
        if (raw && value != MathF.Truncate(value))
            throw new ArgumentException("Raw --game-time values must be integer units within 0..2880.");

        float requestedRaw = normalized
            ? value * BuildScopedLightDbcProfileResolver.DayCycleUnits
            : value;
        int resolvedRaw = normalized
            ? checked((int)MathF.Round(requestedRaw, MidpointRounding.AwayFromZero))
            : checked((int)requestedRaw);
        float normalizedValue = normalized
            ? value
            : value / BuildScopedLightDbcProfileResolver.DayCycleUnits;
        return new LightDbcRequestedTime(input, normalized ? "normalized_0_to_1" : "raw_0_to_2880", value, normalizedValue, requestedRaw, resolvedRaw);
    }

    private static int ParseMapId(string text)
    {
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int mapId) || mapId < 0)
            throw new ArgumentException("--map-id must be a non-negative integer.");
        return mapId;
    }

    private static Vector3 ParseVector3(string text, string optionName)
    {
        string[] parts = text.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 3)
            throw new ArgumentException($"{optionName} must contain exactly three comma-separated numbers.");

        return new Vector3(
            ParseFiniteFloat(parts[0], optionName),
            ParseFiniteFloat(parts[1], optionName),
            ParseFiniteFloat(parts[2], optionName));
    }

    private static float ParseFiniteFloat(string text, string optionName)
    {
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
            || !float.IsFinite(value))
        {
            throw new ArgumentException($"{optionName} requires a finite invariant-culture number; received '{text}'.");
        }

        return value;
    }

    private static string ResolveDefinitionsDirectory(string? explicitDirectory)
    {
        if (!string.IsNullOrWhiteSpace(explicitDirectory))
            return RequireDefinitionsDirectory(explicitDirectory);

        List<string> candidates = [Path.Combine(AppContext.BaseDirectory, "definitions")];
        foreach (string start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            DirectoryInfo? current = new(Path.GetFullPath(start));
            while (current is not null)
            {
                candidates.Add(Path.Combine(current.FullName, "libs", "wowdev", "WoWDBDefs", "definitions"));
                candidates.Add(Path.Combine(current.FullName, "wow-viewer", "libs", "wowdev", "WoWDBDefs", "definitions"));
                current = current.Parent;
            }
        }

        string? resolved = candidates
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(ContainsRequiredDefinitions);
        return resolved is null
            ? throw new DirectoryNotFoundException(
                "Could not locate bundled WoWDBDefs definitions. Provide --dbd-dir <definitions> explicitly.")
            : Path.GetFullPath(resolved);
    }

    private static string RequireDefinitionsDirectory(string path)
    {
        string fullPath = RequireDirectory(path, "WoWDBDefs definitions directory");
        if (!ContainsRequiredDefinitions(fullPath))
        {
            string missing = string.Join(", ", TableNames
                .Where(table => !File.Exists(Path.Combine(fullPath, $"{table}.dbd")))
                .Select(table => $"{table}.dbd"));
            throw new FileNotFoundException(
                $"WoWDBDefs definitions directory is missing required files: {missing}",
                fullPath);
        }

        return fullPath;
    }

    private static bool ContainsRequiredDefinitions(string path) =>
        Directory.Exists(path) && TableNames.All(table => File.Exists(Path.Combine(path, $"{table}.dbd")));

    private static string RequireDirectory(string path, string description)
    {
        string fullPath = Path.GetFullPath(path);
        if (!Directory.Exists(fullPath))
            throw new DirectoryNotFoundException($"{description} was not found: {fullPath}");
        return fullPath;
    }

    private static string RequireSingleOption(string[] args, string longName, string? shortName = null) =>
        GetSingleOption(args, longName, shortName)
        ?? throw new ArgumentException($"Required option {longName} was not provided.");

    private static string? GetSingleOption(string[] args, string longName, string? shortName = null)
    {
        string[] values = GetOptionValues(args, longName, shortName).ToArray();
        if (values.Length > 1)
            throw new ArgumentException($"Option {longName} may be provided only once.");
        return values.SingleOrDefault();
    }

    private static IEnumerable<string> GetOptionValues(string[] args, string longName, string? shortName)
    {
        for (int index = 0; index < args.Length; index += 2)
        {
            if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
                || (!string.IsNullOrWhiteSpace(shortName)
                    && string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase)))
            {
                yield return args[index + 1];
            }
        }
    }

    private static void ValidateOptionShape(string[] args)
    {
        if (args.Length % 2 != 0)
            throw new ArgumentException($"Option '{args[^1]}' is missing its value.");

        for (int index = 0; index < args.Length; index += 2)
        {
            string option = args[index];
            if (!KnownOptions.Contains(option))
                throw new ArgumentException($"Unknown light profile option '{option}'.");
            if (string.IsNullOrWhiteSpace(args[index + 1]))
                throw new ArgumentException($"Option '{option}' requires a non-empty value.");
        }
    }

    private static Vector3 WorldToRenderer(Vector3 world, float mapOrigin) =>
        new(mapOrigin - world.Y, mapOrigin - world.X, world.Z);

    private static Vector3 RendererToWorld(Vector3 renderer, float mapOrigin) =>
        new(mapOrigin - renderer.Y, mapOrigin - renderer.X, renderer.Z);

    private static LightDbcVector3Artifact ToVector(Vector3 value) => new(value.X, value.Y, value.Z);

    private static Vector3 ToNumerics(LightDbcVector3Artifact value) => new(value.X, value.Y, value.Z);

    private sealed class ArchiveDbcProvider(
        string archiveRoot,
        ArchiveCatalogBootstrapOptions archiveOptions) : IDBCProvider
    {
        private readonly Dictionary<string, byte[]> _cache = new(StringComparer.Ordinal);

        public Stream StreamForTableName(string tableName, string build)
        {
            if (!_cache.TryGetValue(tableName, out byte[]? bytes))
            {
                string virtualPath = $"DBFilesClient/{tableName}.dbc";
                bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                    virtualPath,
                    [archiveRoot],
                    archiveOptions);
                if (bytes.Length == 0)
                    throw new InvalidDataException($"Required DBC table '{virtualPath}' is empty for exact build '{build}'.");
                _cache.Add(tableName, bytes);
            }

            return new MemoryStream(bytes, writable: false);
        }
    }
}

internal sealed record LightDbcProfileArtifact(
    string Schema,
    LightDbcSourceArtifact Source,
    LightDbcCatalogArtifact Catalog,
    LightDbcQueryArtifact Query,
    IReadOnlyList<LightDbcSampleArtifact> Samples);

internal sealed record LightDbcSourceArtifact(
    string ExactBuild,
    string ArchiveRoot,
    string WowDbDefsDefinitionsDirectory,
    IReadOnlyList<LightDbcTableSourceEvidence> Tables);

internal sealed record LightDbcTableSourceEvidence(
    string Table,
    string DbcVirtualPath,
    string DbcSha256,
    string DbdPath,
    string DbdSha256);

internal sealed record LightDbcCatalogArtifact(
    int LightRecordCount,
    int LightParamsRecordCount,
    int LightIntBandRecordCount,
    int LightFloatBandRecordCount,
    int LightSkyboxRecordCount,
    int TimedSampleCount);

internal sealed record LightDbcQueryArtifact(
    int MapId,
    LightDbcCoordinateQuery Coordinate,
    IReadOnlyList<LightDbcRequestedTime> GameTimes);

internal sealed record LightDbcCoordinateQuery(
    string InputKind,
    LightDbcVector3Artifact InputPosition,
    float MapOrigin,
    LightDbcVector3Artifact WorldPosition,
    LightDbcVector3Artifact RendererPosition);

internal sealed record LightDbcRequestedTime(
    string Input,
    string InputUnits,
    float InputValue,
    float RequestedNormalized0To1,
    float RequestedRaw0To2880,
    int ResolvedRawTime);

internal sealed record LightDbcSampleArtifact(
    LightDbcRequestedTime RequestedTime,
    int NormalizedRawTime,
    float EvaluatedNormalized0To1,
    LightDbcSpatialBlendArtifact SpatialBlend,
    IReadOnlyList<LightDbcColorBandArtifact> ColorBands,
    IReadOnlyList<LightDbcFloatBandArtifact> FloatBands,
    LightDbcParamsArtifact PrimaryLightParams,
    LightDbcSkyboxArtifact? PrimarySkybox);

internal sealed record LightDbcSpatialBlendArtifact(
    LightDbcZoneBlendArtifact? Global,
    LightDbcZoneBlendArtifact? Local);

internal sealed record LightDbcZoneBlendArtifact(
    float Weight,
    LightDbcZoneArtifact Zone,
    LightDbcProfileJoinArtifact ProfileRecords);

internal sealed record LightDbcZoneArtifact(
    int LightRecordId,
    int ContinentId,
    LightDbcVector3Artifact RawGameCoordsXzy,
    LightDbcVector3Artifact WorldPosition,
    LightDbcVector3Artifact RendererPosition,
    float RawFalloffStart,
    float RawFalloffEnd,
    float FalloffStartWorldUnits,
    float FalloffEndWorldUnits,
    IReadOnlyList<int> LightParamsIds,
    int ClearWeatherLightParamsId);

internal sealed record LightDbcProfileJoinArtifact(
    int ClearWeatherParamsIndex,
    int LightParamsRecordId,
    int? LightSkyboxRecordId,
    IReadOnlyList<int> LightIntBandRecordIds,
    IReadOnlyList<int> LightFloatBandRecordIds);

internal sealed record LightDbcColorBandArtifact(
    int Index,
    string Name,
    LightDbcRgbArtifact Rgb);

internal sealed record LightDbcFloatBandArtifact(
    int Index,
    string Name,
    float Value);

internal sealed record LightDbcParamsArtifact(
    int RecordId,
    int HighlightSky,
    int LightSkyboxId,
    float Glow,
    float WaterShallowAlpha,
    float WaterDeepAlpha,
    float OceanShallowAlpha,
    float OceanDeepAlpha,
    int Flags);

internal sealed record LightDbcSkyboxArtifact(int RecordId, string Name, int Flags);

internal sealed record LightDbcVector3Artifact(float X, float Y, float Z);

internal sealed record LightDbcRgbArtifact(float R, float G, float B);
