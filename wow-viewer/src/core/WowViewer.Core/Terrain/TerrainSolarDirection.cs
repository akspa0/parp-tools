using System.Numerics;

namespace WowViewer.Core.Terrain;

/// <summary>
/// Shared authored solar direction in terrain world space.
/// </summary>
public static class TerrainSolarDirection
{
    public static Vector3 Evaluate(float gameTime)
    {
        float wrappedTime = gameTime - MathF.Floor(gameTime);
        float sunAngle = wrappedTime * MathF.Tau;
        float sunHeight = MathF.Sin(sunAngle - (MathF.PI * 0.5f));

        // wow-1.0.0-world-lighting-shadow-model-2026-07-15.md ties this down two ways:
        //
        // 1. Axis convention: MCNR/MCVT store vertex data directly in raw WoW world axes
        //    (+X = North, +Y = West, +Z = Up; confirmed by AdtTensorPackBuilder.AssembleNormals
        //    decoding MCNR with no axis swap, and by TerrainMeshBuilder deriving vertex world-X
        //    from the row/tileY-indexed quantities that decrease southward).
        // 2. Azimuth: the traced native SetDirection ray (theta = 225 degrees in all four sampled
        //    table entries, §2.1) holds a CONSTANT bearing and only cycles elevation (phi) for
        //    day/night. Its ray is negative on X/Y/Z, so the light *source* (-ray) sits at a fixed
        //    north-west bearing (positive X/Y), never swinging through zero.
        //
        // A previous pass swept the horizontal bias with time of day (cos(sunAngle-pi/2)), which
        // collapses to zero exactly at solar noon/midnight -- the sun pointed straight up with no
        // horizontal bias at all, producing a washed-out, near-symmetric shadow ring on bowl/crater
        // terrain instead of the client's persistently one-sided hillshade (confirmed by a real
        // 0.5.3 minimap: a clean bright-north/dark-south gradient at every sampled time, not just
        // away from noon). Keep the bearing fixed north-west all day; only elevation cycles.
        const float diagonalHorizontalScale = 0.3535534f; // 0.5 / sqrt(2), matches the traced 225-degree bearing
        return Vector3.Normalize(new Vector3(
            diagonalHorizontalScale,
            diagonalHorizontalScale,
            MathF.Max(sunHeight, 0.05f)));
    }
}
