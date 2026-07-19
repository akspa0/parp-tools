using System.Collections.Immutable;
using System.Numerics;

namespace WowViewer.Core.IO.Lighting;

/// <summary>
/// Immutable build-scoped Light database catalog and clear-weather evaluator.
/// Spatial queries use world units; raw DBC coordinate evidence remains on each zone record.
/// </summary>
public sealed class LightDbcCatalog
{
    private readonly ImmutableDictionary<int, LightDbcParamsRecord> _params;
    private readonly ImmutableDictionary<int, LightDbcIntBandRecord> _intBands;
    private readonly ImmutableDictionary<int, LightDbcFloatBandRecord> _floatBands;
    private readonly ImmutableDictionary<int, LightDbcSkyboxRecord> _skyboxes;

    private LightDbcCatalog(
        string build,
        ImmutableArray<LightDbcZoneRecord> zones,
        ImmutableDictionary<int, LightDbcParamsRecord> lightParams,
        ImmutableDictionary<int, LightDbcIntBandRecord> intBands,
        ImmutableDictionary<int, LightDbcFloatBandRecord> floatBands,
        ImmutableDictionary<int, LightDbcSkyboxRecord> skyboxes,
        LightDbcSourceHashes sourceHashes,
        ImmutableArray<LightDbcBandCountRecovery> bandCountRecoveries,
        ImmutableArray<LightDbcMissingSkyboxReference> missingSkyboxReferences)
    {
        Build = build;
        Zones = zones;
        _params = lightParams;
        _intBands = intBands;
        _floatBands = floatBands;
        _skyboxes = skyboxes;
        SourceHashes = sourceHashes;
        BandCountRecoveries = bandCountRecoveries;
        MissingSkyboxReferences = missingSkyboxReferences;
    }

    public string Build { get; }

    public ImmutableArray<LightDbcZoneRecord> Zones { get; }

    public int TimedSampleCount =>
        _intBands.Values.Sum(static band => band.Samples.Length) +
        _floatBands.Values.Sum(static band => band.Samples.Length);

    public LightDbcSourceHashes SourceHashes { get; }

    public ImmutableArray<LightDbcBandCountRecovery> BandCountRecoveries { get; }

    public ImmutableArray<LightDbcMissingSkyboxReference> MissingSkyboxReferences { get; }

    public int LightParamsRecordCount => _params.Count;

    public int LightIntBandRecordCount => _intBands.Count;

    public int LightFloatBandRecordCount => _floatBands.Count;

    public int LightSkyboxRecordCount => _skyboxes.Count;

    public static LightDbcCatalog Create(
        string exactBuild,
        IEnumerable<LightDbcZoneRecord> zones,
        IEnumerable<LightDbcParamsRecord> lightParams,
        IEnumerable<LightDbcIntBandRecord> intBands,
        IEnumerable<LightDbcFloatBandRecord> floatBands,
        IEnumerable<LightDbcSkyboxRecord> skyboxes,
        LightDbcSourceHashes? sourceHashes = null,
        IEnumerable<LightDbcBandCountRecovery>? bandCountRecoveries = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(exactBuild);
        ArgumentNullException.ThrowIfNull(zones);
        ArgumentNullException.ThrowIfNull(lightParams);
        ArgumentNullException.ThrowIfNull(intBands);
        ArgumentNullException.ThrowIfNull(floatBands);
        ArgumentNullException.ThrowIfNull(skyboxes);

        ImmutableArray<LightDbcZoneRecord> zoneArray = zones
            .OrderBy(static zone => zone.RecordId)
            .ToImmutableArray();
        ImmutableDictionary<int, LightDbcParamsRecord> paramsById = ToUniqueDictionary(
            lightParams,
            static value => value.RecordId,
            "LightParams",
            exactBuild);
        ImmutableDictionary<int, LightDbcIntBandRecord> intBandsById = ToUniqueDictionary(
            intBands,
            static value => value.RecordId,
            "LightIntBand",
            exactBuild);
        ImmutableDictionary<int, LightDbcFloatBandRecord> floatBandsById = ToUniqueDictionary(
            floatBands,
            static value => value.RecordId,
            "LightFloatBand",
            exactBuild);
        ImmutableDictionary<int, LightDbcSkyboxRecord> skyboxesById = ToUniqueDictionary(
            skyboxes,
            static value => value.RecordId,
            "LightSkybox",
            exactBuild);

        ImmutableArray<LightDbcMissingSkyboxReference>.Builder missingSkyboxes =
            ImmutableArray.CreateBuilder<LightDbcMissingSkyboxReference>();
        foreach (LightDbcParamsRecord lightParam in paramsById.Values)
        {
            for (int band = 0; band < BuildScopedLightDbcProfileResolver.ColorBandCount; band++)
            {
                int bandId = BuildScopedLightDbcProfileResolver.GetIntBandRecordId(
                    lightParam.RecordId,
                    (LightDbcColorBand)band);
                if (!intBandsById.ContainsKey(bandId))
                {
                    throw MissingJoin("LightIntBand", bandId, lightParam.RecordId, exactBuild);
                }
            }

            for (int band = 0; band < BuildScopedLightDbcProfileResolver.FloatBandCount; band++)
            {
                int bandId = BuildScopedLightDbcProfileResolver.GetFloatBandRecordId(
                    lightParam.RecordId,
                    (LightDbcFloatBand)band);
                if (!floatBandsById.ContainsKey(bandId))
                {
                    throw MissingJoin("LightFloatBand", bandId, lightParam.RecordId, exactBuild);
                }
            }

            if (lightParam.LightSkyboxId > 0 && !skyboxesById.ContainsKey(lightParam.LightSkyboxId))
            {
                // Skyboxes decorate the outdoor profile but do not supply terrain direct,
                // ambient, or fog bands. Some exact client tables retain dangling optional
                // references; preserve that evidence without disabling usable terrain lighting.
                missingSkyboxes.Add(new LightDbcMissingSkyboxReference(
                    lightParam.RecordId,
                    lightParam.LightSkyboxId));
            }
        }

        return new LightDbcCatalog(
            exactBuild,
            zoneArray,
            paramsById,
            intBandsById,
            floatBandsById,
            skyboxesById,
            sourceHashes ?? LightDbcSourceHashes.Empty,
            (bandCountRecoveries ?? []).ToImmutableArray(),
            missingSkyboxes.ToImmutable());
    }

    public LightDbcEvaluation EvaluateClearWeather(int continentId, Vector3 worldPosition, int time)
    {
        LightDbcZoneRecord[] candidates = Zones
            .Where(zone => zone.ContinentId == continentId)
            .ToArray();
        if (candidates.Length == 0)
        {
            throw new LightDbcLoadException(
                $"No Light records exist for continent {continentId} in exact build '{Build}'.");
        }

        LightDbcZoneRecord? global = candidates
            .Where(static zone => zone.FalloffEnd <= 0f)
            .OrderBy(static zone => zone.RecordId)
            .FirstOrDefault();

        (LightDbcZoneRecord? Zone, float Weight) local = candidates
            .Where(static zone => zone.FalloffEnd > 0f)
            .Select(zone => (Zone: zone, Weight: CalculateZoneWeight(zone, worldPosition)))
            .Where(static candidate => candidate.Weight > 0f)
            .OrderByDescending(static candidate => candidate.Weight)
            .ThenBy(static candidate => candidate.Zone.FalloffEnd)
            .ThenBy(static candidate => candidate.Zone.RecordId)
            .FirstOrDefault();

        if (global is null && local.Zone is null)
        {
            throw new LightDbcLoadException(
                $"No global or in-range local Light record resolved for continent {continentId}, " +
                $"position {worldPosition}, exact build '{Build}'.");
        }

        EvaluatedProfile? globalProfile = global is null ? null : EvaluateProfile(global, time);
        EvaluatedProfile? localProfile = local.Zone is null ? null : EvaluateProfile(local.Zone, time);

        float localWeight = localProfile is null
            ? 0f
            : globalProfile is null
                ? 1f
                : local.Weight;

        ImmutableArray<Vector3> colors = BlendArrays(
            globalProfile?.Colors,
            localProfile?.Colors,
            localWeight);
        ImmutableArray<float> floats = BlendArrays(
            globalProfile?.Floats,
            localProfile?.Floats,
            localWeight);

        EvaluatedProfile primary = localProfile ?? globalProfile!;
        return new LightDbcEvaluation(
            colors,
            floats,
            primary.Params,
            primary.Skybox,
            new LightDbcEvaluationEvidence(
                Build,
                continentId,
                time,
                BuildScopedLightDbcProfileResolver.NormalizeTime(time),
                globalProfile?.Evidence,
                localProfile?.Evidence,
                localWeight,
                SourceHashes))
        {
            LocalColorBands = localProfile?.Colors ?? ImmutableArray<Vector3>.Empty,
            LocalFloatBands = localProfile?.Floats ?? ImmutableArray<float>.Empty,
        };
    }

    public static float CalculateZoneWeight(LightDbcZoneRecord zone, Vector3 worldPosition)
    {
        ArgumentNullException.ThrowIfNull(zone);
        float end = zone.FalloffEnd;
        if (end <= 0f)
        {
            return 0f;
        }

        float distance = Vector3.Distance(worldPosition, zone.WorldPosition);
        if (distance >= end)
        {
            return 0f;
        }

        float start = Math.Clamp(zone.FalloffStart, 0f, end);
        if (distance <= start || end <= start)
        {
            return 1f;
        }

        return Math.Clamp(1f - ((distance - start) / (end - start)), 0f, 1f);
    }

    private EvaluatedProfile EvaluateProfile(LightDbcZoneRecord zone, int time)
    {
        int paramsId = zone.ClearWeatherLightParamsId;
        if (paramsId <= 0 || !_params.TryGetValue(paramsId, out LightDbcParamsRecord? lightParams))
        {
            throw new LightDbcLoadException(
                $"Light record {zone.RecordId} for exact build '{Build}' has unresolved clear-weather " +
                $"LightParams record {paramsId} in slot {BuildScopedLightDbcProfileResolver.ClearWeatherParamsIndex}.");
        }

        ImmutableArray<Vector3>.Builder colors =
            ImmutableArray.CreateBuilder<Vector3>(BuildScopedLightDbcProfileResolver.ColorBandCount);
        ImmutableArray<int>.Builder colorEvidence =
            ImmutableArray.CreateBuilder<int>(BuildScopedLightDbcProfileResolver.ColorBandCount);
        for (int index = 0; index < BuildScopedLightDbcProfileResolver.ColorBandCount; index++)
        {
            LightDbcColorBand band = (LightDbcColorBand)index;
            int recordId = BuildScopedLightDbcProfileResolver.GetIntBandRecordId(paramsId, band);
            colors.Add(BuildScopedLightDbcProfileResolver.EvaluateColorBand(_intBands[recordId], time));
            colorEvidence.Add(recordId);
        }

        ImmutableArray<float>.Builder floats =
            ImmutableArray.CreateBuilder<float>(BuildScopedLightDbcProfileResolver.FloatBandCount);
        ImmutableArray<int>.Builder floatEvidence =
            ImmutableArray.CreateBuilder<int>(BuildScopedLightDbcProfileResolver.FloatBandCount);
        for (int index = 0; index < BuildScopedLightDbcProfileResolver.FloatBandCount; index++)
        {
            LightDbcFloatBand band = (LightDbcFloatBand)index;
            int recordId = BuildScopedLightDbcProfileResolver.GetFloatBandRecordId(paramsId, band);
            floats.Add(BuildScopedLightDbcProfileResolver.EvaluateFloatBand(_floatBands[recordId], time));
            floatEvidence.Add(recordId);
        }

        LightDbcSkyboxRecord? skybox = lightParams.LightSkyboxId > 0
            && _skyboxes.TryGetValue(lightParams.LightSkyboxId, out LightDbcSkyboxRecord? resolvedSkybox)
                ? resolvedSkybox
                : null;
        return new EvaluatedProfile(
            colors.MoveToImmutable(),
            floats.MoveToImmutable(),
            lightParams,
            skybox,
            new LightDbcProfileEvidence(
                Build,
                zone.RecordId,
                BuildScopedLightDbcProfileResolver.ClearWeatherParamsIndex,
                paramsId,
                skybox?.RecordId,
                colorEvidence.MoveToImmutable(),
                floatEvidence.MoveToImmutable()));
    }

    private static ImmutableArray<Vector3> BlendArrays(
        ImmutableArray<Vector3>? global,
        ImmutableArray<Vector3>? local,
        float localWeight)
    {
        if (global is null)
        {
            return local!.Value;
        }

        if (local is null)
        {
            return global.Value;
        }

        ImmutableArray<Vector3>.Builder result = ImmutableArray.CreateBuilder<Vector3>(global.Value.Length);
        for (int i = 0; i < global.Value.Length; i++)
        {
            result.Add(Vector3.Lerp(global.Value[i], local.Value[i], localWeight));
        }

        return result.MoveToImmutable();
    }

    private static ImmutableArray<float> BlendArrays(
        ImmutableArray<float>? global,
        ImmutableArray<float>? local,
        float localWeight)
    {
        if (global is null)
        {
            return local!.Value;
        }

        if (local is null)
        {
            return global.Value;
        }

        ImmutableArray<float>.Builder result = ImmutableArray.CreateBuilder<float>(global.Value.Length);
        for (int i = 0; i < global.Value.Length; i++)
        {
            result.Add(global.Value[i] + ((local.Value[i] - global.Value[i]) * localWeight));
        }

        return result.MoveToImmutable();
    }

    private static ImmutableDictionary<int, T> ToUniqueDictionary<T>(
        IEnumerable<T> values,
        Func<T, int> getId,
        string table,
        string build)
    {
        ImmutableDictionary<int, T>.Builder builder = ImmutableDictionary.CreateBuilder<int, T>();
        foreach (T value in values)
        {
            int id = getId(value);
            if (id <= 0 || !builder.TryAdd(id, value))
            {
                throw new LightDbcLoadException(
                    $"Table '{table}' for exact build '{build}' contains invalid or duplicate record ID {id}.");
            }
        }

        return builder.ToImmutable();
    }

    private static LightDbcLoadException MissingJoin(
        string table,
        int bandRecordId,
        int paramsRecordId,
        string build)
    {
        return new LightDbcLoadException(
            $"LightParams record {paramsRecordId} for exact build '{build}' requires missing " +
            $"{table} record {bandRecordId}; the implicit dense band join was not satisfied.");
    }

    private sealed record EvaluatedProfile(
        ImmutableArray<Vector3> Colors,
        ImmutableArray<float> Floats,
        LightDbcParamsRecord Params,
        LightDbcSkyboxRecord? Skybox,
        LightDbcProfileEvidence Evidence);
}
