using System.Numerics;
using WowViewer.Core.IO.Lit;
using WowViewer.Core.Renderer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class LitTerrainDayNightProfileTests
{
    [Fact]
    public void EvaluateGlobalClear_UsesExactTimedLitColorsAndLabelsHybridInputs()
    {
        LitFileProfile profile = BuildProfile(includeDefault: true, omitAmbient: false);

        LitTerrainLightingEvaluation result = LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f);

        Assert.Equal(new Vector3(0.5f, 0f, 0.5f), result.Lighting.DirectionalColor);
        Assert.Equal(new Vector3(0f, 0.5f, 0.5f), result.Lighting.AmbientColor);
        Assert.Equal(new Vector3(0.5f, 0.5f, 0f), result.Lighting.FogColor);
        Assert.Equal(LitTerrainDayNightProfile.EvidenceState, result.Lighting.EvidenceState);
        Assert.Equal(LitTerrainDayNightProfile.DirectionEvidenceState, result.DirectionEvidenceState);
        Assert.Equal([0, 1, 7], result.ContributingTrackIds);
        Assert.Equal(720f, result.LitTimeOfDay);
    }

    [Fact]
    public void EvaluateGlobalClear_RejectsMissingDefaultLight()
    {
        LitFileProfile profile = BuildProfile(includeDefault: false, omitAmbient: false);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f));

        Assert.Contains("default/global", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EvaluateGlobalClear_RejectsMissingRequiredTrackSamples()
    {
        LitFileProfile profile = BuildProfile(includeDefault: true, omitAmbient: true);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f));

        Assert.Contains("track 1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EvaluateGlobalClear_AcceptsModernNegativeCountPartialProfile()
    {
        LitLightGroupProfile group = BuildGroup(LitLightGroupKind.Partial, omitAmbient: false);
        var profile = new LitFileProfile(
            "partial.lit",
            LitProfileReader.Version85,
            -1,
            18,
            0x15F0,
            [new LitLightProfile(0, null, [group])]);

        LitTerrainLightingEvaluation result = LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f);

        Assert.Equal(LitProfileReader.Version85, result.LitVersion);
        Assert.Equal("Default partial LIT profile", result.LightName);
        Assert.Equal(new Vector3(0.5f, 0f, 0.5f), result.Lighting.DirectionalColor);
    }

    [Fact]
    public void EvaluateGlobalClear_RejectsAmbiguousDefaultLights()
    {
        LitLightGroupProfile group = BuildGroup(LitLightGroupKind.Clear, omitAmbient: false);
        LitLightHeaderProfile firstHeader = BuildHeader(0, isDefault: true);
        LitLightHeaderProfile secondHeader = BuildHeader(1, isDefault: true);
        var profile = new LitFileProfile(
            "ambiguous.lit",
            LitProfileReader.Version84,
            2,
            18,
            0x1550,
            [
                new LitLightProfile(0, firstHeader, [group]),
                new LitLightProfile(1, secondHeader, [group]),
            ]);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f));

        Assert.Contains("exactly one default/global", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EvaluateGlobalClear_RejectsAmbiguousClearGroups()
    {
        LitLightGroupProfile first = BuildGroup(LitLightGroupKind.Clear, omitAmbient: false);
        LitLightGroupProfile second = BuildGroup(LitLightGroupKind.Clear, omitAmbient: false);
        LitLightHeaderProfile header = BuildHeader(0, isDefault: true);
        var profile = new LitFileProfile(
            "ambiguous.lit",
            LitProfileReader.Version84,
            1,
            18,
            0x1550,
            [new LitLightProfile(0, header, [first, second])]);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => LitTerrainDayNightProfile.EvaluateGlobalClear(profile, 0.25f));

        Assert.Contains("exactly one clear group", error.Message, StringComparison.Ordinal);
    }

    private static LitFileProfile BuildProfile(bool includeDefault, bool omitAmbient)
    {
        LitLightGroupProfile group = BuildGroup(LitLightGroupKind.Clear, omitAmbient);
        LitLightHeaderProfile header = BuildHeader(0, includeDefault);
        var light = new LitLightProfile(0, header, [group]);
        return new LitFileProfile("fixture.lit", LitProfileReader.Version84, 1, 18, 0x1550, [light]);
    }

    private static LitLightGroupProfile BuildGroup(LitLightGroupKind kind, bool omitAmbient)
    {
        static LitColorTrack Track(int index, Vector3 atMidnight, Vector3 atNoon)
        {
            return new LitColorTrack(index, 2,
            [
                new LitColorKeyframe(0, 0, atMidnight),
                new LitColorKeyframe(1440, 0, atNoon),
            ]);
        }

        var tracks = Enumerable.Range(0, 18)
            .Select(index => new LitColorTrack(index, 0, []))
            .ToArray();
        tracks[0] = Track(0, Vector3.Zero, new Vector3(1f, 0f, 1f));
        if (!omitAmbient)
            tracks[1] = Track(1, Vector3.Zero, new Vector3(0f, 1f, 1f));
        tracks[7] = Track(7, Vector3.Zero, new Vector3(1f, 1f, 0f));

        return new LitLightGroupProfile(
            0,
            kind,
            tracks,
            [],
            0,
            0,
            [],
            0x1550);
    }

    private static LitLightHeaderProfile BuildHeader(int index, bool isDefault)
    {
        return new LitLightHeaderProfile(
            index,
            isDefault ? -1 : 0,
            isDefault ? -1 : 0,
            isDefault ? -1 : 0,
            Vector3.Zero,
            0f,
            0f,
            "Global");
    }
}
