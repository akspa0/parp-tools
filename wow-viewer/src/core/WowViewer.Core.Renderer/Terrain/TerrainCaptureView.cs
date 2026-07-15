using System.Numerics;

namespace WowViewer.Core.Renderer.Terrain;

/// <summary>Deterministic, axis-aligned view used for terrain training captures.</summary>
public static class TerrainCaptureView
{
    private const float CameraClearance = 1024f;
    private const float MinimumDepthRange = 2048f;

    public static TerrainCaptureCamera CreateTopDown(int tileX, int tileY, float minHeight, float maxHeight)
    {
        if (!float.IsFinite(minHeight) || !float.IsFinite(maxHeight) || maxHeight < minHeight)
            throw new ArgumentException("Terrain capture height bounds must be finite and ordered.");

        float centerX = TerrainConstants.MapOrigin - ((tileY + 0.5f) * TerrainConstants.TileSize);
        float centerY = TerrainConstants.MapOrigin - ((tileX + 0.5f) * TerrainConstants.TileSize);
        Vector3 position = new(centerX, centerY, maxHeight + CameraClearance);

        // ADT tile-X advances toward renderer -Y and tile-Y advances toward renderer -X.
        // Looking down with +X as camera-up makes image-right follow tile-X and image-down
        // follow tile-Y, matching height_257[row=tileY, column=tileX].
        Matrix4x4 view = Matrix4x4.CreateLookAt(position, position - Vector3.UnitZ, Vector3.UnitX);
        float farPlane = MathF.Max(MinimumDepthRange, (maxHeight - minHeight) + (CameraClearance * 2f));
        Matrix4x4 projection = Matrix4x4.CreateOrthographic(
            TerrainConstants.TileSize,
            TerrainConstants.TileSize,
            0.1f,
            farPlane);

        return new TerrainCaptureCamera(position, view, projection, farPlane);
    }
}

public sealed record TerrainCaptureCamera(
    Vector3 Position,
    Matrix4x4 View,
    Matrix4x4 Projection,
    float FarPlane);
