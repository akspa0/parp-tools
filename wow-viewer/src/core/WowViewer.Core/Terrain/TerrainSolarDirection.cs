using System.Numerics;

namespace WowViewer.Core.Terrain;

/// <summary>
/// Shared authored solar direction in terrain world space.
///
/// AXIS CONVENTION (from wow-1.0.0-world-lighting-shadow-model-2026-07-15.md): MCNR/MCVT store
/// vertex data directly in raw WoW world axes -- +X = North, +Y = West, +Z = Up. Confirmed by
/// <c>AdtTensorPackBuilder.AssembleNormals</c> decoding MCNR with no axis swap, and by
/// <c>TerrainMeshBuilder</c> deriving vertex world-X from row/tileY-indexed quantities that decrease
/// southward.
///
/// AZIMUTH IS AN OPEN QUESTION. The traced native <c>SetDirection</c> ray holds theta = 225 degrees
/// in all four sampled <c>thetaTable</c> entries, i.e. a CONSTANT bearing with only elevation (phi)
/// cycling. But that trace is of WoW <b>1.0.0</b>, while the synthetic-minimap target is 0.5.3.3368.
/// Whether 0.5.3 shares the constant azimuth or sweeps east-to-west has not been measured; both
/// models are available here and <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/> can score
/// them against authored minimaps.
///
/// PRIOR FAILED ATTEMPT, do not repeat verbatim: an earlier pass swept the horizontal bias as
/// <c>cos(sunAngle - pi/2)</c>, which collapses to zero exactly at solar noon/midnight -- the sun
/// pointing straight up with no horizontal bias at all, producing a washed-out, near-symmetric
/// shadow ring on bowl/crater terrain. Any sweep model must keep the horizontal magnitude constant
/// and rotate only the bearing, which is what <see cref="EvaluateSweepAzimuthDegrees"/> does.
/// </summary>
public static class TerrainSolarDirection
{
    /// <summary>
    /// The clamped elevation scalar alone (before the fixed-bearing horizontal components are
    /// mixed in). Exposed separately from <see cref="Evaluate"/> because this value is exactly
    /// what determines whether two different hours render an IDENTICAL lighting direction: the
    /// bearing never changes, so any two hours with the same elevation are indistinguishable to
    /// every consumer of this class, including <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/>.
    /// </summary>
    public static float EvaluateElevation(float gameTime)
    {
        float wrappedTime = gameTime - MathF.Floor(gameTime);
        float sunAngle = wrappedTime * MathF.Tau;
        float sunHeight = MathF.Sin(sunAngle - (MathF.PI * 0.5f));
        return MathF.Max(sunHeight, 0.05f);
    }

    /// <summary>
    /// Source-bearing of the traced native ray, in degrees within the world XY plane measured from
    /// +X (North) toward +Y (West). The traced <c>thetaTable</c> ray sits at 225 degrees; a vector
    /// pointing *toward* the source is its inverse, 45 degrees -- north-west.
    /// </summary>
    public const float TracedSourceAzimuthDegrees = 45f;

    /// <summary>
    /// Fixed horizontal magnitude of the light vector before normalization. Preserved exactly as the
    /// original hardcoded pair (0.3535534, 0.3535534), whose length is 0.5, so swapping azimuth
    /// changes only the bearing and never the elevation the vector actually resolves to.
    /// </summary>
    private const float HorizontalMagnitude = 0.5f;

    public static Vector3 Evaluate(float gameTime) => Evaluate(gameTime, TracedSourceAzimuthDegrees);

    /// <summary>
    /// Evaluates the light vector at an explicit source bearing instead of the traced constant.
    ///
    /// WHY THIS IS PARAMETERISED: the fixed 225-degree ray comes from an x32dbg trace of WoW
    /// <b>1.0.0</b>'s <c>SetDirection</c>, but the synthetic-minimap target is 0.5.3.3368 -- an
    /// earlier, different build. Nobody has measured whether 0.5.3 shares that constant azimuth or
    /// sweeps east-to-west like a conventional sun. <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/>
    /// sweeps this parameter against authored minimaps so the real client data can answer it.
    /// Callers that do not pass an azimuth keep the traced behaviour exactly.
    /// </summary>
    /// <param name="azimuthDegrees">
    /// Bearing of the light SOURCE in the world XY plane, degrees from +X (North) toward +Y (West).
    /// 45 is the traced north-west. 270 is East (-Y), 90 is West (+Y).
    /// </param>
    public static Vector3 Evaluate(float gameTime, float azimuthDegrees)
    {
        float sunHeight = EvaluateElevation(gameTime);
        float azimuth = float.IsFinite(azimuthDegrees) ? azimuthDegrees : TracedSourceAzimuthDegrees;
        float radians = azimuth * (MathF.PI / 180f);
        return Vector3.Normalize(new Vector3(
            HorizontalMagnitude * MathF.Cos(radians),
            HorizontalMagnitude * MathF.Sin(radians),
            sunHeight));
    }

    /// <summary>
    /// East, as a source bearing: world +X is North and +Y is West, so East is -Y = 270 degrees.
    /// </summary>
    public const float EastAzimuthDegrees = 270f;

    /// <summary>West as a source bearing (+Y).</summary>
    public const float WestAzimuthDegrees = 90f;

    /// <summary>
    /// A conventional sun that rises in the east and sets in the west, sweeping the source bearing
    /// through <paramref name="noonAzimuthDegrees"/> at solar noon. This is the hypothesis the traced
    /// 1.0.0 constant-azimuth model contradicts; it exists so both can be scored against real
    /// authored 0.5.3 minimaps rather than argued about.
    /// </summary>
    public static float EvaluateSweepAzimuthDegrees(
        float gameTime,
        float noonAzimuthDegrees = TracedSourceAzimuthDegrees)
    {
        // A full day is one full revolution, so sunrise at 06:00 (gameTime 0.25) through sunset at
        // 18:00 (0.75) covers the visible half-circle: 90 degrees either side of the noon bearing.
        // Scaling by 180 instead of 360 is the easy off-by-two here -- it would sweep only 90
        // degrees between sunrise and sunset and never reach east or west.
        float wrapped = gameTime - MathF.Floor(gameTime);
        return noonAzimuthDegrees + ((wrapped - 0.5f) * 360f);
    }
}
