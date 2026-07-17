using System.Numerics;

namespace WowViewer.Core.Renderer.Terrain;

/// <summary>
/// Recovered outdoor world-light direction computation for the 0.5.3.3368 client.
/// The native client stores a downward ray. Consumers that use Lambert source semantics
/// must invert that ray before applying their coordinate transform.
/// </summary>
public static class Native0533368WorldLightDirection
{
    public const string BuildIdentity = "0.5.3.3368";
    public const string DirectionModelRevision = "wow-0.5.3.3368-native-world-light-ray-v1";
    public const string CoordinateTransformRevision = "terrain-source-identity-unproven-v1";
    public const string DirectionEvidenceState = "native_0533368_ray_recovered_viewer_transform_unproven";
    public const string LightingModel = "mcnr_lambert_plus_mcsh_native_0533368_world_ray_provisional_transform_v1";

    private const float ThetaRadians = 3.926991f; // Native downward-ray azimuth: 225 degrees.
    private static readonly float[] PolarAngleRadians =
    [
        2.216568f, // 127 degrees
        1.919862f, // 110 degrees
        2.216568f, // 127 degrees
        1.919862f  // 110 degrees
    ];

    /// <summary>
    /// Evaluates the native downward world-light ray. The table is periodic over the normalized day.
    /// </summary>
    public static Vector3 EvaluateNativeRay(float gameTime)
    {
        if (!float.IsFinite(gameTime))
            throw new ArgumentOutOfRangeException(nameof(gameTime), "Game time must be finite.");

        float wrappedTime = gameTime - MathF.Floor(gameTime);
        float sample = wrappedTime * PolarAngleRadians.Length;
        int current = (int)MathF.Floor(sample) % PolarAngleRadians.Length;
        int next = (current + 1) % PolarAngleRadians.Length;
        float fraction = sample - MathF.Floor(sample);
        float phi = PolarAngleRadians[current]
            + ((PolarAngleRadians[next] - PolarAngleRadians[current]) * fraction);

        return Vector3.Normalize(new Vector3(
            MathF.Sin(phi) * MathF.Cos(ThetaRadians),
            MathF.Sin(phi) * MathF.Sin(ThetaRadians),
            MathF.Cos(phi)));
    }

    /// <summary>
    /// Returns the recovered ray and the temporary identity native-to-viewer source transform.
    /// The transform is intentionally labeled unproven: it is a world-light research/diagnostic
    /// path, never a minimap profile or client-exact profile, until the native/viewer image
    /// calibration in Spec 106 is accepted.
    /// </summary>
    public static bool TryEvaluateProvisionalViewerSource(
        string? buildIdentity,
        float gameTime,
        out NativeWorldLightDirectionSample sample)
    {
        if (!string.Equals(buildIdentity, BuildIdentity, StringComparison.OrdinalIgnoreCase))
        {
            sample = default!;
            return false;
        }

        Vector3 nativeRay = EvaluateNativeRay(gameTime);
        // The renderer's Lambert path expects a vector from terrain toward the source. The
        // native 0.5.3 ray is source-to-terrain. Native/viewer terrain axes are not calibrated
        // yet, so preserve the identity axis mapping as explicit provisional provenance.
        Vector3 provisionalViewerSource = -nativeRay;
        sample = new NativeWorldLightDirectionSample(
            nativeRay,
            provisionalViewerSource,
            DirectionModelRevision,
            CoordinateTransformRevision,
            DirectionEvidenceState,
            LightingModel);
        return true;
    }
}

public sealed record NativeWorldLightDirectionSample(
    Vector3 NativeLightRay,
    Vector3 ViewerSourceDirection,
    string DirectionModelRevision,
    string CoordinateTransformRevision,
    string EvidenceState,
    string LightingModel);
