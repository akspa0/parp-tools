using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Renderer.Terrain;

/// <summary>
/// Versioned fallback lighting used when an exact client LIT profile or supported Light* DBC
/// profile is not available.
/// The MCNR Lambert and MCSH modulation mechanics are evidence-backed; these colors and the
/// shadow-strength coefficient are authored approximations and must remain labeled as such.
/// </summary>
public static class AuthoredTerrainDayNightProfile
{
    public const string ProfileRevision = "wow-1.0.0-authored-day-night-v1";
    public const string EvidenceState = "authored_fallback_not_client_light_data";
    public const string LightingModel = "mcnr_lambert_plus_mcsh_directional_v1";
    public const float DefaultGameTime = 0.35f;
    public const float DefaultMcshShadowStrength = 0.60f;

    public static TerrainLightingSample Evaluate(float gameTime)
    {
        float wrappedTime = gameTime - MathF.Floor(gameTime);
        float sunAngle = wrappedTime * MathF.Tau;
        float sunHeight = MathF.Sin(sunAngle - (MathF.PI * 0.5f));
        float sunHorizontal = MathF.Cos(sunAngle - (MathF.PI * 0.5f));
        Vector3 direction = Vector3.Normalize(new Vector3(
            sunHorizontal * 0.5f,
            0.3f,
            MathF.Max(sunHeight, 0.05f)));

        float dayFactor = MathF.Max(0f, sunHeight);
        Vector3 directionalColor = Vector3.Lerp(
            new Vector3(0.20f, 0.20f, 0.35f),
            new Vector3(0.80f, 0.78f, 0.70f),
            dayFactor);
        Vector3 ambientColor = Vector3.Lerp(
            new Vector3(0.25f, 0.25f, 0.35f),
            new Vector3(0.55f, 0.55f, 0.60f),
            dayFactor);
        Vector3 fogColor = Vector3.Lerp(
            new Vector3(0.08f, 0.08f, 0.15f),
            new Vector3(0.60f, 0.70f, 0.85f),
            dayFactor);

        return new TerrainLightingSample(
            ProfileRevision,
            EvidenceState,
            LightingModel,
            wrappedTime,
            direction,
            directionalColor,
            1f,
            ambientColor,
            1f,
            fogColor,
            DefaultMcshShadowStrength);
    }

    public static Vector3 Shade(
        Vector3 albedo,
        Vector3 rendererSpaceNormal,
        float mcshShadow,
        TerrainLightingSample sample)
    {
        Vector3 lighting = TerrainLightingMath.Evaluate(
            rendererSpaceNormal,
            sample.LightDirection,
            sample.DirectionalColor * sample.DirectionalIntensity,
            sample.AmbientColor * sample.AmbientIntensity,
            mcshShadow,
            sample.McshShadowStrength);
        return Vector3.Max(Vector3.Zero, albedo * lighting);
    }
}

public sealed record TerrainLightingSample(
    string ProfileRevision,
    string EvidenceState,
    string LightingModel,
    float GameTime,
    Vector3 LightDirection,
    Vector3 DirectionalColor,
    float DirectionalIntensity,
    Vector3 AmbientColor,
    float AmbientIntensity,
    Vector3 FogColor,
    float McshShadowStrength);
