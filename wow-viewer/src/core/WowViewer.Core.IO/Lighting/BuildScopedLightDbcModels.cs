using System.Collections.Immutable;
using System.Numerics;

namespace WowViewer.Core.IO.Lighting;

/// <summary>
/// Semantic positions of the eighteen color bands stored for each LightParams record.
/// The names follow the LightIntBand/LIT track contract; unknown tracks remain explicit
/// so callers never silently collapse or renumber the database data.
/// </summary>
public enum LightDbcColorBand
{
    Direct = 0,
    Ambient = 1,
    SkyTop = 2,
    SkyMiddle = 3,
    SkyMiddleToHorizon = 4,
    SkyAboveHorizon = 5,
    SkyHorizon = 6,
    Fog = 7,
    Unknown8 = 8,
    Sun = 9,
    SunHalo = 10,
    Unknown11 = 11,
    Cloud = 12,
    Unknown13 = 13,
    Unknown14 = 14,
    GroundShadow = 15,
    WaterLight = 16,
    WaterDark = 17,
}

/// <summary>
/// Semantic positions of the six float bands stored for each LightParams record.
/// </summary>
public enum LightDbcFloatBand
{
    FogEnd = 0,
    FogStartScalar = 1,
    SkyData0 = 2,
    SkyData1 = 3,
    SkyData2 = 4,
    SkyData3 = 5,
}

public readonly record struct LightDbcColorSample(int Time, int PackedBgrx);

public readonly record struct LightDbcFloatSample(int Time, float Value);

public sealed record LightDbcIntBandRecord(
    int RecordId,
    ImmutableArray<LightDbcColorSample> Samples);

public sealed record LightDbcFloatBandRecord(
    int RecordId,
    ImmutableArray<LightDbcFloatSample> Samples);

public sealed record LightDbcSkyboxRecord(
    int RecordId,
    string Name,
    int Flags);

public sealed record LightDbcParamsRecord(
    int RecordId,
    int HighlightSky,
    int LightSkyboxId,
    float Glow,
    float WaterShallowAlpha,
    float WaterDeepAlpha,
    float OceanShallowAlpha,
    float OceanDeepAlpha,
    int Flags);

/// <summary>
/// One Light database record. Raw values are retained beside the world-space values.
/// Classic Light.dbc stores GameCoords in X,Z,Y order and coordinates/radii in
/// 1/36-world-unit fixed scale.
/// </summary>
public sealed record LightDbcZoneRecord(
    int RecordId,
    int ContinentId,
    Vector3 RawGameCoordsXzy,
    float RawFalloffStart,
    float RawFalloffEnd,
    ImmutableArray<int> LightParamsIds)
{
    public const float GameUnitsPerWorldUnit = 36f;

    public Vector3 WorldPosition => new(
        RawGameCoordsXzy.X / GameUnitsPerWorldUnit,
        RawGameCoordsXzy.Z / GameUnitsPerWorldUnit,
        RawGameCoordsXzy.Y / GameUnitsPerWorldUnit);

    public float FalloffStart => RawFalloffStart / GameUnitsPerWorldUnit;

    public float FalloffEnd => RawFalloffEnd / GameUnitsPerWorldUnit;

    public int ClearWeatherLightParamsId =>
        LightParamsIds.IsDefaultOrEmpty ? 0 : LightParamsIds[BuildScopedLightDbcProfileResolver.ClearWeatherParamsIndex];

    public Vector3 ToRendererPosition(float mapOrigin) => new(
        mapOrigin - WorldPosition.Y,
        mapOrigin - WorldPosition.X,
        WorldPosition.Z);
}

public sealed record LightDbcProfileEvidence(
    string Build,
    int LightRecordId,
    int ClearWeatherParamsIndex,
    int LightParamsRecordId,
    int? LightSkyboxRecordId,
    ImmutableArray<int> LightIntBandRecordIds,
    ImmutableArray<int> LightFloatBandRecordIds);

public sealed record LightDbcSourceHashes(
    ImmutableDictionary<string, string> DatabaseTableSha256,
    ImmutableDictionary<string, string> WowDbDefsDefinitionSha256)
{
    public static LightDbcSourceHashes Empty { get; } = new(
        ImmutableDictionary<string, string>.Empty,
        ImmutableDictionary<string, string>.Empty);
}

public sealed record LightDbcEvaluationEvidence(
    string Build,
    int ContinentId,
    int RequestedTime,
    int NormalizedTime,
    LightDbcProfileEvidence? GlobalProfile,
    LightDbcProfileEvidence? LocalProfile,
    float LocalWeight,
    LightDbcSourceHashes Sources);

/// <summary>
/// Immutable, fully evaluated clear-weather lighting state. ColorBands always contains
/// eighteen entries and FloatBands always contains six entries in enum order.
/// </summary>
public sealed record LightDbcEvaluation(
    ImmutableArray<Vector3> ColorBands,
    ImmutableArray<float> FloatBands,
    LightDbcParamsRecord PrimaryParams,
    LightDbcSkyboxRecord? PrimarySkybox,
    LightDbcEvaluationEvidence Evidence)
{
    public Vector3 this[LightDbcColorBand band] => ColorBands[(int)band];

    public float this[LightDbcFloatBand band] => FloatBands[(int)band];
}

public sealed class LightDbcLoadException : IOException
{
    public LightDbcLoadException(string message)
        : base(message)
    {
    }

    public LightDbcLoadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
