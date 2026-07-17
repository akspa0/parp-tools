using System.Numerics;
using WowViewer.Core.IO.Lit;

namespace WowViewer.Core.Renderer.Terrain;

/// <summary>
/// Evaluates global clear-weather colors for LIT inspection and world-light diagnostics.
/// Synthesized minimaps intentionally do not consume these tracks.
/// LIT supplies colors but not the directional vector or the MCSH attenuation coefficient;
/// those two inputs remain explicitly authored in this hybrid profile.
/// </summary>
public static class LitTerrainDayNightProfile
{
    public const string ProfileRevision = "lit-global-clear-colors-v1";
    public const string EvidenceState = "client_lit_colors_authored_direction_and_mcsh_strength";
    public const string DirectionEvidenceState = "authored_solar_direction_not_lit_data";
    public const string McshEvidenceState = "authored_mcsh_strength_not_client_exact";

    public const int TrackDirectColor = 0;
    public const int TrackAmbientColor = 1;
    public const int TrackFogColor = 7;

    public static LitTerrainLightingEvaluation EvaluateGlobalClear(
        LitFileProfile profile,
        float gameTime)
    {
        TerrainLightingSample authored = AuthoredTerrainDayNightProfile.Evaluate(gameTime);
        return EvaluateGlobalClear(
            profile,
            gameTime,
            authored.LightDirection,
            ProfileRevision,
            EvidenceState,
            AuthoredTerrainDayNightProfile.LightingModel,
            DirectionEvidenceState,
            authored.McshShadowStrength);
    }

    /// <summary>
    /// Evaluates LIT colors with an independently recovered build-scoped world-light direction.
    /// The caller owns the evidence state of that direction and must not call this client-exact
    /// unless its native-to-viewer transform has been calibrated.
    /// </summary>
    public static LitTerrainLightingEvaluation EvaluateGlobalClear(
        LitFileProfile profile,
        float gameTime,
        NativeWorldLightDirectionSample direction)
    {
        ArgumentNullException.ThrowIfNull(direction);
        return EvaluateGlobalClear(
            profile,
            gameTime,
            direction.ViewerSourceDirection,
            $"{ProfileRevision}+{direction.DirectionModelRevision}+{direction.CoordinateTransformRevision}",
            "partially_proven_client_lit_colors_native_0533368_ray_unproven_viewer_transform",
            direction.LightingModel,
            direction.EvidenceState,
            AuthoredTerrainDayNightProfile.DefaultMcshShadowStrength);
    }

    private static LitTerrainLightingEvaluation EvaluateGlobalClear(
        LitFileProfile profile,
        float gameTime,
        Vector3 lightDirection,
        string profileRevision,
        string evidenceState,
        string lightingModel,
        string directionEvidenceState,
        float mcshShadowStrength)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (!float.IsFinite(gameTime) || gameTime < 0f || gameTime >= 1f)
            throw new ArgumentOutOfRangeException(nameof(gameTime), "Game time must be in [0, 1).");
        if (!IsFiniteDirection(lightDirection))
            throw new ArgumentOutOfRangeException(nameof(lightDirection), "Light direction must be finite and non-zero.");

        (LitLightProfile light, LitLightGroupProfile group) = SelectUniqueGlobalClear(profile);

        float litTime = gameTime * LitProfileReader.TimeUnitsPerDay;
        Vector3 direct = EvaluateRequiredTrack(group, TrackDirectColor, litTime, "global diffuse");
        Vector3 ambient = EvaluateRequiredTrack(group, TrackAmbientColor, litTime, "global ambient");
        Vector3 fog = EvaluateRequiredTrack(group, TrackFogColor, litTime, "fog");

        var sample = new TerrainLightingSample(
            profileRevision,
            evidenceState,
            lightingModel,
            gameTime,
            Vector3.Normalize(lightDirection),
            direct,
            1f,
            ambient,
            1f,
            fog,
            mcshShadowStrength);

        return new LitTerrainLightingEvaluation(
            sample,
            profile.VersionNumber,
            light.Index,
            light.Header?.Name ?? "Default partial LIT profile",
            group.Index,
            litTime,
            [TrackDirectColor, TrackAmbientColor, TrackFogColor],
            directionEvidenceState,
            McshEvidenceState);
    }

    private static bool IsFiniteDirection(Vector3 value)
    {
        return float.IsFinite(value.X)
            && float.IsFinite(value.Y)
            && float.IsFinite(value.Z)
            && value.LengthSquared() > 1e-10f;
    }

    private static (LitLightProfile Light, LitLightGroupProfile Group) SelectUniqueGlobalClear(
        LitFileProfile profile)
    {
        if (profile.IsSinglePartialProfile)
        {
            LitLightProfile partial = profile.Lights.Count == 1
                ? profile.Lights[0]
                : throw new InvalidDataException(
                    $"LIT partial profile must contain exactly one light; found {profile.Lights.Count}.");
            LitLightGroupProfile group = partial.IsPartial
                && partial.Groups.Count == 1
                && partial.Groups[0].Kind == LitLightGroupKind.Partial
                    ? partial.Groups[0]
                    : throw new InvalidDataException(
                        "LIT partial profile does not contain exactly one partial group.");
            return (partial, group);
        }

        LitLightProfile[] defaults = profile.Lights
            .Where(candidate => candidate.Header?.IsDefault == true)
            .ToArray();
        if (defaults.Length != 1)
        {
            throw new InvalidDataException(
                $"LIT profile must contain exactly one default/global light header; found {defaults.Length}.");
        }

        LitLightGroupProfile[] clearGroups = defaults[0].Groups
            .Where(candidate => candidate.Kind == LitLightGroupKind.Clear)
            .ToArray();
        if (clearGroups.Length != 1)
        {
            throw new InvalidDataException(
                $"LIT default/global light must contain exactly one clear group; found {clearGroups.Length}.");
        }

        return (defaults[0], clearGroups[0]);
    }

    private static Vector3 EvaluateRequiredTrack(
        LitLightGroupProfile group,
        int trackIndex,
        float timeOfDay,
        string description)
    {
        if (!group.TryGetTrack(trackIndex, out LitColorTrack track) ||
            !track.TryEvaluate(timeOfDay, out Vector3 color))
        {
            throw new InvalidDataException(
                $"LIT group {group.Index} has no timed samples for required track {trackIndex} ({description}).");
        }

        return color;
    }
}

public sealed record LitTerrainLightingEvaluation(
    TerrainLightingSample Lighting,
    uint LitVersion,
    int LightIndex,
    string LightName,
    int GroupIndex,
    float LitTimeOfDay,
    IReadOnlyList<int> ContributingTrackIds,
    string DirectionEvidenceState,
    string McshEvidenceState);
