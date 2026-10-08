namespace WowViewer.Core.Maps;

/// <summary>
/// Parameter set defining the physical lighting, solar direction, and specular response of terrain minimap generation.
/// </summary>
public readonly record struct MinimapLightingParameters(
    float SolarAzimuth,        // θ in radians, [0, 2π)
    float SolarElevation,      // φ in radians, (0, π/2]
    float AmbientIntensity,    // A in [0.0, 1.0]
    float DiffuseIntensity,    // D in [0.0, 1.0]
    float CastShadowStrength,  // [0.0, 1.0]
    float SpecularIntensity,   // ks in [0.0, 1.0]
    float SpecularPower,       // p >= 1.0
    bool ApplyDxt1Quantization = true
)
{
    public static MinimapLightingParameters Default => new(
        SolarAzimuth: 2.3561945f, // ~135 degrees (North-West)
        SolarElevation: 0.6457718f, // ~37 degrees
        AmbientIntensity: 0.25f,
        DiffuseIntensity: 0.75f,
        CastShadowStrength: 0.30f,
        SpecularIntensity: 0.05f,
        SpecularPower: 8.0f,
        ApplyDxt1Quantization: true
    );
}
