using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Lit;

/// <summary>
/// Converts LIT and Light* spatial records from their client fixed-point XZY layout into the
/// semantic game-world and renderer-space conventions used by the viewer.
/// </summary>
public static class LitCoordinateTransform
{
    /// <summary>
    /// Converts a raw client XZY position to game-world XYZ units. LIT/Light* spatial values use
    /// 36 fixed units per world unit, and the file's second and third components are Z/Y.
    /// </summary>
    public static Vector3 ToGameWorldPosition(Vector3 rawXzyPosition)
    {
        float scale = 1f / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;
        return new(
            rawXzyPosition.X * scale,
            rawXzyPosition.Z * scale,
            rawXzyPosition.Y * scale);
    }

    /// <summary>
    /// Converts a semantic game-world XYZ position to the renderer's origin-relative axes.
    /// </summary>
    public static Vector3 ToRendererPosition(Vector3 gameWorldPosition, float mapOrigin)
    {
        return new(
            mapOrigin - gameWorldPosition.Y,
            mapOrigin - gameWorldPosition.X,
            gameWorldPosition.Z);
    }

    /// <summary>Converts a raw client XZY position directly to renderer coordinates.</summary>
    public static Vector3 ToRendererFromRawXzy(Vector3 rawXzyPosition, float mapOrigin)
    {
        return ToRendererPosition(ToGameWorldPosition(rawXzyPosition), mapOrigin);
    }
}
