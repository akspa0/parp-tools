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
    /// <summary>
    /// Sine of the source elevation, i.e. the Z component of the unit light direction. Consumers
    /// that only need to know whether two times render the SAME direction (such as
    /// <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/>'s distinctness check) can compare
    /// this scalar; consumers that need the actual vector must use <see cref="Evaluate(float)"/>.
    /// </summary>
    public static float EvaluateElevation(float gameTime) =>
        MathF.Sin(EvaluateElevationDegrees(gameTime) * (MathF.PI / 180f));

    /// <summary>
    /// Source-bearing of the traced native ray, in degrees within the world XY plane measured from
    /// +X (North) toward +Y (West). The traced <c>thetaTable</c> ray sits at 225 degrees; a vector
    /// pointing *toward* the source is its inverse, 45 degrees -- north-west.
    /// </summary>
    public const float TracedSourceAzimuthDegrees = 45f;

    /// <summary>
    /// Lowest source elevation over the day, from the traced <c>phiTable</c> entry 1.919862 rad
    /// (110 degrees): the ray points 20 degrees below horizontal, so the source sits 20 degrees above it.
    /// </summary>
    public const float TracedMinimumElevationDegrees = 20f;

    /// <summary>
    /// Highest source elevation over the day, from the traced <c>phiTable</c> entry 2.216568 rad
    /// (127 degrees). The client's sun never climbs anywhere near overhead -- it stays low, which is
    /// why authored minimaps show long, strongly directional shadows all day.
    /// </summary>
    public const float TracedMaximumElevationDegrees = 37f;

    /// <summary>
    /// Source elevation in degrees at a given time of day.
    ///
    /// THE BUG THIS REPLACED: the previous model pinned the horizontal magnitude at a constant 0.5
    /// and varied only Z. That is not a spherical direction -- the resulting elevation is
    /// <c>atan(z / 0.5)</c>, which produced 5.7 degrees at 06:00 and then 45 degrees by 08:00,
    /// topping out at 63.4 degrees. So the sun leapt off the horizon in two hours, sat almost
    /// overhead all day, and its horizontal push halved exactly when shadows should have been
    /// longest. Rendered output read as light sliding around the terrain rather than a sun crossing
    /// the sky.
    ///
    /// The traced client vector <c>(-0.6481626, -0.6481628, -0.3997127)</c> has horizontal magnitude
    /// 0.9166 and elevation 23.6 degrees, sitting inside the phi-table band below -- roughly half our
    /// old midday elevation with twice the horizontal component.
    /// </summary>
    /// <remarks>
    /// The BAND is traced; the interpolation SHAPE across the day is modelled (a smooth peak at solar
    /// noon), because the four sampled table entries fix the endpoints but not the curve between them.
    /// </remarks>
    public static float EvaluateElevationDegrees(float gameTime)
    {
        float wrapped = gameTime - MathF.Floor(gameTime);
        // 0 at midnight, 1 at solar noon.
        float dayFactor = (MathF.Sin((wrapped * MathF.Tau) - (MathF.PI * 0.5f)) + 1f) * 0.5f;
        return TracedMinimumElevationDegrees
            + (dayFactor * (TracedMaximumElevationDegrees - TracedMinimumElevationDegrees));
    }

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
        float azimuth = float.IsFinite(azimuthDegrees) ? azimuthDegrees : TracedSourceAzimuthDegrees;
        return FromSpherical(azimuth, EvaluateElevationDegrees(gameTime));
    }

    /// <summary>
    /// Builds a unit light-source direction from bearing and elevation. Horizontal magnitude is
    /// <c>cos(elevation)</c>, so it shrinks as the sun climbs -- the property the old fixed-0.5
    /// horizontal broke, and the reason its shadow lengths did not track time of day correctly.
    /// </summary>
    public static Vector3 FromSpherical(float azimuthDegrees, float elevationDegrees)
    {
        float azimuth = azimuthDegrees * (MathF.PI / 180f);
        float elevation = Math.Clamp(elevationDegrees, -89f, 89f) * (MathF.PI / 180f);
        float horizontal = MathF.Cos(elevation);
        return Vector3.Normalize(new Vector3(
            horizontal * MathF.Cos(azimuth),
            horizontal * MathF.Sin(azimuth),
            MathF.Sin(elevation)));
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
