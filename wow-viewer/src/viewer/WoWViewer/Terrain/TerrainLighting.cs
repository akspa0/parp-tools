using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WoWViewer.Terrain;

/// <summary>
/// Day/night cycle lighting for terrain rendering.
/// Drives sun direction, light color, ambient color, and fog color based on game time.
/// Based on Ghidra analysis of the Alpha 0.5.3 lighting system (08_Terrain_Lighting.md).
/// </summary>
public class TerrainLighting
{
    private bool _hasExternalOverride;
    private Vector3 _externalLightColor;
    private Vector3 _externalAmbientColor;
    private Vector3 _externalFogColor;
    private bool _hasExternalLightDirectionOverride;
    private Vector3 _externalLightDirection;

    /// <summary>Game time as fraction of day: 0.0 = midnight, 0.5 = noon, 1.0 = midnight.</summary>
    public float GameTime { get; set; } = 0.5f; // Default: noon so an unprofiled world remains legible.

    /// <summary>
    /// True once the user has manually set <see cref="GameTime"/> from a UI control. Manual input
    /// freezes the automatic clock until the user explicitly resumes it.
    /// </summary>
    public bool HasManualGameTimeOverride { get; set; }

    /// <summary>
    /// Whether the Alpha 0.5.3 real-time clock is enabled. It is on by default for the interactive
    /// viewer; synthetic minimap generation uses its own frozen lighting contract.
    /// </summary>
    public bool AutomaticTimeOfDayEnabled { get; private set; } = true;

    public bool IsAutomaticTimeOfDay => AutomaticTimeOfDayEnabled && !HasManualGameTimeOverride;

    /// <summary>Current game time in the native Light/LIT 0..2880 unit range.</summary>
    public int TimeOfDayUnits => WorldTimeCycle.ToTimeUnits(GameTime);

    public void SetAutomaticTimeOfDay(bool enabled)
    {
        AutomaticTimeOfDayEnabled = enabled;
        if (enabled)
            HasManualGameTimeOverride = false;
        else
            HasManualGameTimeOverride = true;
    }

    public void AdvanceAutomaticTime(double realSeconds)
    {
        if (!IsAutomaticTimeOfDay)
            return;

        GameTime = WorldTimeCycle.AdvanceNormalized(GameTime, realSeconds);
    }

    /// <summary>Current sun/light direction (normalized, pointing toward light).</summary>
    public Vector3 LightDirection { get; private set; } = TerrainSolarDirection.Evaluate(0.5f);

    public bool HasExternalLightDirectionOverride => _hasExternalLightDirectionOverride;

    public Vector3 ExternalLightDirection => _externalLightDirection;

    /// <summary>Current directional light color.</summary>
    public Vector3 LightColor { get; private set; } = new Vector3(1.0f, 0.95f, 0.85f);

    /// <summary>Current ambient light color.</summary>
    public Vector3 AmbientColor { get; private set; } = new Vector3(0.35f, 0.35f, 0.4f);

    /// <summary>Current fog color.</summary>
    public Vector3 FogColor { get; private set; } = new Vector3(0.6f, 0.7f, 0.85f);

    /// <summary>Fog start distance.</summary>
    public float FogStart { get; set; } = 200f;

    /// <summary>Fog end distance.</summary>
    public float FogEnd { get; set; } = 1500f;

    public void ApplyExternalLighting(Vector3 lightColor, Vector3 ambientColor, Vector3 fogColor)
    {
        _hasExternalOverride = true;
        _externalLightColor = lightColor;
        _externalAmbientColor = ambientColor;
        _externalFogColor = fogColor;
    }

    public void ClearExternalLighting()
    {
        _hasExternalOverride = false;
    }

    public void ApplyExternalLightDirection(Vector3 lightDirection)
    {
        if (lightDirection.LengthSquared() <= 1e-6f)
            return;

        _hasExternalLightDirectionOverride = true;
        _externalLightDirection = Vector3.Normalize(lightDirection);
    }

    public void ClearExternalLightDirection()
    {
        _hasExternalLightDirectionOverride = false;
    }

    /// <summary>
    /// Update lighting parameters based on current game time.
    /// </summary>
    public void Update()
    {
        // Sun angle: 0.0 = below horizon (midnight), 0.25 = sunrise, 0.5 = zenith, 0.75 = sunset
        float sunAngle = GameTime * MathF.PI * 2f;
        float sunHeight = MathF.Sin(sunAngle - MathF.PI * 0.5f); // -1 at midnight, +1 at noon
        // The shared profile owns terrain-world versus minimap-raster direction mapping so
        // interactive terrain and exported minimaps use the same solar path.
        LightDirection = TerrainSolarDirection.Evaluate(GameTime);

        if (_hasExternalLightDirectionOverride)
            LightDirection = _externalLightDirection;

        // Interpolate colors based on time of day
        // Night (0.0-0.2, 0.8-1.0), Dawn (0.2-0.3), Day (0.3-0.7), Dusk (0.7-0.8)
        float dayFactor = MathF.Max(0, sunHeight); // 0 at night, 1 at noon

        // Light color: warm yellow during day, cool blue at night
        // WoW uses relatively high directional intensity with strong ambient fill
        LightColor = Vector3.Lerp(
            new Vector3(0.2f, 0.2f, 0.35f),    // night: dim blue
            new Vector3(0.8f, 0.78f, 0.7f),    // day: warm white (moderated to avoid blow-out with high ambient)
            dayFactor);

        // Ambient: WoW ambient is high — objects are always well-lit even in shadow.
        // Real WoW uses hemisphere lighting + Light.dbc; these values approximate the look.
        AmbientColor = Vector3.Lerp(
            new Vector3(0.25f, 0.25f, 0.35f),  // night: still visible, blue-tinted
            new Vector3(0.55f, 0.55f, 0.6f),   // day: strong ambient fill
            dayFactor);

        // Fog color follows sky color
        FogColor = Vector3.Lerp(
            new Vector3(0.08f, 0.08f, 0.15f),  // night: dark blue
            new Vector3(0.6f, 0.7f, 0.85f),    // day: sky blue
            dayFactor);

        if (_hasExternalOverride)
        {
            LightColor = _externalLightColor;
            AmbientColor = _externalAmbientColor;
            FogColor = _externalFogColor;
        }
    }
}
