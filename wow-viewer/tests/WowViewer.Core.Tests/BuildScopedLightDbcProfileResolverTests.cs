using System.Collections.Immutable;
using System.Numerics;
using WowViewer.Core.IO.Lighting;

namespace WowViewer.Core.Tests;

public sealed class BuildScopedLightDbcProfileResolverTests
{
    [Fact]
    public void ResolveBandSampleCount_RecoversExact243MalformedWaterBandPrefix()
    {
        int[] times = [360, 360, 720, 1440, 2520, 2640, 0, 0];
        int[] colors = [6052956, 6052956, 7237230, 5789784, 7237230, 6052956, 0, 0];

        int count = BuildScopedLightDbcProfileResolver.ResolveBandSampleCount(360, times, colors);

        Assert.Equal(6, count);
    }

    [Fact]
    public void ResolveBandSampleCount_PreservesValidZeroCount()
    {
        Assert.Equal(
            0,
            BuildScopedLightDbcProfileResolver.ResolveBandSampleCount(0, new int[16], new int[16]));
    }

    [Fact]
    public void Catalog_PreservesMissingOptionalSkyboxWithoutRejectingTerrainBands()
    {
        LightDbcZoneRecord zone = Zone(recordId: 10, clearParamsId: 1, rawEnd: 0f);
        LightDbcParamsRecord parameters = Params(1) with { LightSkyboxId = 18 };
        (IEnumerable<LightDbcIntBandRecord> ints, IEnumerable<LightDbcFloatBandRecord> floats) =
            Bands([(1, 0x00112233, 100f)]);

        LightDbcCatalog catalog = LightDbcCatalog.Create(
            "2.4.3.8606",
            [zone],
            [parameters],
            ints,
            floats,
            []);

        LightDbcEvaluation evaluation = catalog.EvaluateClearWeather(0, Vector3.Zero, 0);
        Assert.Null(evaluation.PrimarySkybox);
        Assert.Equal(new LightDbcMissingSkyboxReference(1, 18), Assert.Single(catalog.MissingSkyboxReferences));
    }

    [Fact]
    public void EvaluateColorBand_WrapsAcrossMidnight()
    {
        LightDbcIntBandRecord band = new(
            1,
            [
                new LightDbcColorSample(240, 0x00ff0000),
                new LightDbcColorSample(2640, 0x000000ff),
            ]);

        Vector3 color = BuildScopedLightDbcProfileResolver.EvaluateColorBand(band, 0);

        AssertVector(color, new Vector3(0.5f, 0f, 0.5f));
    }

    [Fact]
    public void UnpackBgrx_UsesDiskBgrByteOrder()
    {
        Vector3 color = BuildScopedLightDbcProfileResolver.UnpackBgrx(0x00112233);

        AssertVector(color, new Vector3(0x11 / 255f, 0x22 / 255f, 0x33 / 255f));
    }

    [Fact]
    public void EvaluateClearWeather_SelectsLocalZoneAndBlendsEveryBand()
    {
        LightDbcCatalog catalog = CreateCatalog(
            zones:
            [
                Zone(recordId: 10, clearParamsId: 1, rawEnd: 0f),
                Zone(recordId: 20, clearParamsId: 2, rawEnd: 360f),
            ],
            parameterValues: [(1, 0x00000000, 100f), (2, 0x00ffffff, 300f)]);

        LightDbcEvaluation result = catalog.EvaluateClearWeather(
            continentId: 0,
            worldPosition: new Vector3(5f, 0f, 0f),
            time: 720);

        Assert.All(result.ColorBands, color => AssertVector(color, new Vector3(0.5f)));
        Assert.All(result.FloatBands, value => Assert.Equal(200f, value, 4));
        Assert.Equal(2, result.PrimaryParams.RecordId);
        Assert.Equal(0.5f, result.Evidence.LocalWeight, 4);
        Assert.Equal(10, result.Evidence.GlobalProfile!.LightRecordId);
        Assert.Equal(1, result.Evidence.GlobalProfile.LightParamsRecordId);
        Assert.Equal(20, result.Evidence.LocalProfile!.LightRecordId);
        Assert.Equal(2, result.Evidence.LocalProfile.LightParamsRecordId);
        Assert.Equal(Enumerable.Range(1, 18), result.Evidence.GlobalProfile.LightIntBandRecordIds);
        Assert.Equal(Enumerable.Range(19, 18), result.Evidence.LocalProfile.LightIntBandRecordIds);
        Assert.Equal(Enumerable.Range(1, 6), result.Evidence.GlobalProfile.LightFloatBandRecordIds);
        Assert.Equal(Enumerable.Range(7, 6), result.Evidence.LocalProfile.LightFloatBandRecordIds);
        Assert.Equal("1.12.1.5875", result.Evidence.Build);
        Assert.Equal(720, result.Evidence.NormalizedTime);
        Assert.Equal("table-light-hash", result.Evidence.Sources.DatabaseTableSha256["Light"]);
        Assert.Equal("definition-light-hash", result.Evidence.Sources.WowDbDefsDefinitionSha256["Light"]);
    }

    [Fact]
    public void EvaluateClearWeather_UsesFirstParamsSlotOnly()
    {
        LightDbcZoneRecord zone = Zone(recordId: 10, clearParamsId: 1, rawEnd: 0f) with
        {
            LightParamsIds = [1, 999, 998, 997, 996],
        };
        LightDbcCatalog catalog = CreateCatalog([zone], [(1, 0x00112233, 100f)]);

        LightDbcEvaluation result = catalog.EvaluateClearWeather(0, Vector3.Zero, 0);

        Assert.Equal(1, result.PrimaryParams.RecordId);
        Assert.Equal(BuildScopedLightDbcProfileResolver.ClearWeatherParamsIndex,
            result.Evidence.GlobalProfile!.ClearWeatherParamsIndex);
    }

    [Fact]
    public void ZoneRecord_RetainsRawValuesAndExposesWorldUnits()
    {
        LightDbcZoneRecord zone = new(
            1,
            0,
            new Vector3(360f, 720f, 1080f),
            180f,
            360f,
            [1]);

        AssertVector(zone.WorldPosition, new Vector3(10f, 30f, 20f));
        AssertVector(zone.ToRendererPosition(100f), new Vector3(70f, 90f, 20f));
        Assert.Equal(5f, zone.FalloffStart);
        Assert.Equal(10f, zone.FalloffEnd);
    }

    [Fact]
    public void Create_FailsClosedWhenExactImplicitBandRecordIsMissing()
    {
        (IEnumerable<LightDbcIntBandRecord> ints, IEnumerable<LightDbcFloatBandRecord> floats) =
            Bands([(1, 0, 1f)]);

        LightDbcLoadException error = Assert.Throws<LightDbcLoadException>(() => LightDbcCatalog.Create(
            "1.12.1.5875",
            [Zone(1, 1, 0f)],
            [Params(1)],
            ints.Where(static band => band.RecordId != 18),
            floats,
            []));

        Assert.Contains("LightIntBand record 18", error.Message, StringComparison.Ordinal);
        Assert.Contains("LightParams record 1", error.Message, StringComparison.Ordinal);
        Assert.Contains("1.12.1.5875", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BandRecordIdJoin_IsBuildDataDenseByParamsRecordId()
    {
        Assert.Equal(199, BuildScopedLightDbcProfileResolver.GetIntBandRecordId(12, LightDbcColorBand.Direct));
        Assert.Equal(216, BuildScopedLightDbcProfileResolver.GetIntBandRecordId(12, LightDbcColorBand.WaterDark));
        Assert.Equal(67, BuildScopedLightDbcProfileResolver.GetFloatBandRecordId(12, LightDbcFloatBand.FogEnd));
        Assert.Equal(72, BuildScopedLightDbcProfileResolver.GetFloatBandRecordId(12, LightDbcFloatBand.SkyData3));
    }

    private static LightDbcCatalog CreateCatalog(
        IEnumerable<LightDbcZoneRecord> zones,
        IEnumerable<(int ParamsId, int PackedColor, float FloatValue)> parameterValues)
    {
        (IEnumerable<LightDbcIntBandRecord> ints, IEnumerable<LightDbcFloatBandRecord> floats) =
            Bands(parameterValues);
        return LightDbcCatalog.Create(
            "1.12.1.5875",
            zones,
            parameterValues.Select(value => Params(value.ParamsId)),
            ints,
            floats,
            [],
            SyntheticHashes());
    }

    private static (IEnumerable<LightDbcIntBandRecord>, IEnumerable<LightDbcFloatBandRecord>) Bands(
        IEnumerable<(int ParamsId, int PackedColor, float FloatValue)> parameterValues)
    {
        List<LightDbcIntBandRecord> ints = [];
        List<LightDbcFloatBandRecord> floats = [];
        foreach ((int paramsId, int color, float value) in parameterValues)
        {
            for (int index = 0; index < BuildScopedLightDbcProfileResolver.ColorBandCount; index++)
            {
                int recordId = BuildScopedLightDbcProfileResolver.GetIntBandRecordId(
                    paramsId,
                    (LightDbcColorBand)index);
                ints.Add(new LightDbcIntBandRecord(recordId, [new LightDbcColorSample(0, color)]));
            }

            for (int index = 0; index < BuildScopedLightDbcProfileResolver.FloatBandCount; index++)
            {
                int recordId = BuildScopedLightDbcProfileResolver.GetFloatBandRecordId(
                    paramsId,
                    (LightDbcFloatBand)index);
                floats.Add(new LightDbcFloatBandRecord(recordId, [new LightDbcFloatSample(0, value)]));
            }
        }

        return (ints, floats);
    }

    private static LightDbcZoneRecord Zone(int recordId, int clearParamsId, float rawEnd)
    {
        return new LightDbcZoneRecord(
            recordId,
            0,
            Vector3.Zero,
            0f,
            rawEnd,
            [clearParamsId]);
    }

    private static LightDbcParamsRecord Params(int id)
    {
        return new LightDbcParamsRecord(id, 0, 0, 0f, 0.5f, 0.5f, 1f, 0.75f, 0);
    }

    private static LightDbcSourceHashes SyntheticHashes()
    {
        return new LightDbcSourceHashes(
            new Dictionary<string, string> { ["Light"] = "table-light-hash" }.ToImmutableDictionary(),
            new Dictionary<string, string> { ["Light"] = "definition-light-hash" }.ToImmutableDictionary());
    }

    private static void AssertVector(Vector3 actual, Vector3 expected)
    {
        Assert.Equal(expected.X, actual.X, 4);
        Assert.Equal(expected.Y, actual.Y, 4);
        Assert.Equal(expected.Z, actual.Z, 4);
    }
}
