using System.Numerics;
using WowViewer.Core.IO.Lit;

internal static class LitProfileCommandSupport
{
    public const string Schema = "wowviewer.lit-profile.v1";

    public static LitProfileArtifact Build(
        LitFileProfile profile,
        LitProfileSourceEvidence source,
        IReadOnlyList<float> normalizedTimes)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(normalizedTimes);

        if (normalizedTimes.Count == 0)
            throw new ArgumentException("At least one normalized game time is required.", nameof(normalizedTimes));

        (LitLightProfile light, LitLightGroupProfile group) = SelectGlobalClearGroup(profile);

        LitColorTrack direct = RequirePopulatedTrack(group, 0, "global diffuse/direct");
        LitColorTrack ambient = RequirePopulatedTrack(group, 1, "ambient");
        LitColorTrack fog = RequirePopulatedTrack(group, 7, "fog");

        int[] contributingTrackIds = Enumerable.Range(0, 8)
            .Where(trackId => TryGetPopulatedTrack(group, trackId, out _))
            .ToArray();

        LitProfileSampleEvidence[] samples = normalizedTimes
            .Select(normalizedTime => BuildSample(group, direct, ambient, fog, normalizedTime, contributingTrackIds))
            .ToArray();

        LitLightHeaderProfile? header = light.Header;
        return new LitProfileArtifact(
            Schema,
            source,
            new LitProfileLayoutEvidence(
                profile.VersionNumber,
                $"0x{profile.VersionNumber:X8}",
                profile.RawLightCount,
                profile.TrackCount,
                profile.GroupStride,
                $"0x{profile.GroupStride:X}"),
            new LitProfileSelectionEvidence(
                light.Index,
                header?.Name,
                light.IsPartial,
                header is null
                    ? null
                    : new LitProfileHeaderEvidence(
                        header.ChunkX,
                        header.ChunkY,
                        header.ChunkRadius,
                        header.Position.X,
                        header.Position.Y,
                        header.Position.Z,
                        header.Radius,
                        header.Dropoff,
                        light.IsPartial || header.IsDefault),
                group.Index,
                group.Kind.ToString(),
                contributingTrackIds),
            samples);
    }

    private static (LitLightProfile Light, LitLightGroupProfile Group) SelectGlobalClearGroup(
        LitFileProfile profile)
    {
        if (profile.IsSinglePartialProfile)
        {
            LitLightProfile partialLight = profile.Lights.Count == 1
                ? profile.Lights[0]
                : throw new InvalidDataException(
                    $"LIT partial profile must contain exactly one light; found {profile.Lights.Count}.");
            LitLightGroupProfile partialGroup = partialLight.Groups
                .FirstOrDefault(group => group.Kind == LitLightGroupKind.Partial)
                ?? throw new InvalidDataException(
                    "LIT partial profile does not contain its required primary partial group.");
            return (partialLight, partialGroup);
        }

        LitLightProfile[] defaults = profile.Lights
            .Where(light => light.Header?.IsDefault == true)
            .ToArray();
        if (defaults.Length != 1)
        {
            throw new InvalidDataException(
                $"LIT profile must contain exactly one default/global light header; found {defaults.Length}.");
        }

        LitLightGroupProfile[] clearGroups = defaults[0].Groups
            .Where(group => group.Kind == LitLightGroupKind.Clear)
            .ToArray();
        if (clearGroups.Length != 1)
        {
            throw new InvalidDataException(
                $"LIT default/global light must contain exactly one clear group; found {clearGroups.Length}.");
        }

        return (defaults[0], clearGroups[0]);
    }

    private static LitProfileSampleEvidence BuildSample(
        LitLightGroupProfile group,
        LitColorTrack direct,
        LitColorTrack ambient,
        LitColorTrack fog,
        float normalizedTime,
        IReadOnlyList<int> contributingTrackIds)
    {
        if (!float.IsFinite(normalizedTime) || normalizedTime is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(normalizedTime),
                normalizedTime,
                "Normalized LIT game time must be within 0..1.");
        }

        float timeUnits = normalizedTime * LitProfileReader.TimeUnitsPerDay;
        return new LitProfileSampleEvidence(
            normalizedTime,
            timeUnits,
            SampleRequired(direct, timeUnits),
            SampleRequired(ambient, timeUnits),
            SampleOptional(group, 2, timeUnits),
            SampleOptional(group, 3, timeUnits),
            SampleOptional(group, 4, timeUnits),
            SampleOptional(group, 5, timeUnits),
            SampleOptional(group, 6, timeUnits),
            SampleRequired(fog, timeUnits),
            contributingTrackIds);
    }

    private static LitProfileTrackSampleEvidence SampleRequired(LitColorTrack track, float timeUnits)
    {
        return new LitProfileTrackSampleEvidence(track.Index, true, ToRgb(track.Evaluate(timeUnits)));
    }

    private static LitProfileTrackSampleEvidence SampleOptional(
        LitLightGroupProfile group,
        int trackId,
        float timeUnits)
    {
        return TryGetPopulatedTrack(group, trackId, out LitColorTrack track)
            ? new LitProfileTrackSampleEvidence(trackId, true, ToRgb(track.Evaluate(timeUnits)))
            : new LitProfileTrackSampleEvidence(trackId, false, null);
    }

    private static LitColorTrack RequirePopulatedTrack(
        LitLightGroupProfile group,
        int trackId,
        string semanticName)
    {
        if (!TryGetPopulatedTrack(group, trackId, out LitColorTrack track))
        {
            throw new InvalidDataException(
                $"LIT {group.Kind} group is missing required track {trackId} ({semanticName}).");
        }

        return track;
    }

    private static bool TryGetPopulatedTrack(
        LitLightGroupProfile group,
        int trackId,
        out LitColorTrack track)
    {
        return group.TryGetTrack(trackId, out track) && track.DeclaredLength > 0;
    }

    private static LitProfileRgb ToRgb(Vector3 color)
    {
        return new LitProfileRgb(color.X, color.Y, color.Z);
    }
}

internal sealed record LitProfileArtifact(
    string Schema,
    LitProfileSourceEvidence Source,
    LitProfileLayoutEvidence Lit,
    LitProfileSelectionEvidence Selection,
    IReadOnlyList<LitProfileSampleEvidence> Samples);

internal sealed record LitProfileSourceEvidence(
    string Kind,
    string Label,
    string? Path,
    string? ArchiveRoot,
    string? VirtualPath,
    string Sha256);

internal sealed record LitProfileLayoutEvidence(
    uint Version,
    string VersionHex,
    int RawLightCount,
    int TrackCount,
    int GroupStride,
    string GroupStrideHex);

internal sealed record LitProfileSelectionEvidence(
    int LightIndex,
    string? LightName,
    bool IsPartial,
    LitProfileHeaderEvidence? Header,
    int GroupIndex,
    string GroupKind,
    IReadOnlyList<int> ContributingTrackIds);

internal sealed record LitProfileHeaderEvidence(
    int ChunkX,
    int ChunkY,
    int ChunkRadius,
    float PositionX,
    float PositionY,
    float PositionZ,
    float Radius,
    float Dropoff,
    bool IsDefault);

internal sealed record LitProfileSampleEvidence(
    float NormalizedTime,
    float Time0To2880,
    LitProfileTrackSampleEvidence Direct,
    LitProfileTrackSampleEvidence Ambient,
    LitProfileTrackSampleEvidence SkyTop,
    LitProfileTrackSampleEvidence SkyMiddle,
    LitProfileTrackSampleEvidence SkyMiddleToHorizon,
    LitProfileTrackSampleEvidence SkyAboveHorizon,
    LitProfileTrackSampleEvidence SkyHorizon,
    LitProfileTrackSampleEvidence Fog,
    IReadOnlyList<int> ContributingTrackIds);

internal sealed record LitProfileTrackSampleEvidence(
    int TrackId,
    bool Present,
    LitProfileRgb? Rgb);

internal sealed record LitProfileRgb(float R, float G, float B);
