using System.Numerics;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;

namespace WowViewer.Tool.Converter;

internal static class TerrainHeightmapGlbExporter
{
    private const int TileSize = 257;

    public static void Export(
        string outputPath,
        IReadOnlyList<float> heightmap257,
        string? texturePath,
        float tileWorldSize,
        bool centerMesh,
        float heightOffset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(heightmap257);
        if (heightmap257.Count != TileSize * TileSize)
            throw new ArgumentException($"Terrain heightmap must contain exactly {TileSize * TileSize} samples.", nameof(heightmap257));

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");

        MaterialBuilder material = new("terrain");
        material.WithDoubleSide(true);
        if (!string.IsNullOrWhiteSpace(texturePath) && File.Exists(texturePath))
            material.WithBaseColor(new SharpGLTF.Memory.MemoryImage(File.ReadAllBytes(texturePath)));
        else
            material.WithBaseColor(new Vector4(0.65f, 0.65f, 0.65f, 1.0f));

        MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty> mesh = new("terrain");
        var primitive = mesh.UsePrimitive(material);
        VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>[] vertices = new VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>[heightmap257.Count];

        float spacing = tileWorldSize / (TileSize - 1);
        float origin = centerMesh ? tileWorldSize * 0.5f : 0f;
        for (int row = 0; row < TileSize; row++)
        {
            for (int column = 0; column < TileSize; column++)
            {
                int index = (row * TileSize) + column;
                float x = (column * spacing) - origin;
                float z = (row * spacing) - origin;
                float y = heightmap257[index] + heightOffset;
                Vector3 normal = ComputeNormal(heightmap257, column, row);
                Vector2 uv = new(column / (float)(TileSize - 1), 1f - (row / (float)(TileSize - 1)));
                vertices[index] = new VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>(
                    new VertexPositionNormal(new Vector3(x, y, z), normal),
                    new VertexTexture1(uv));
            }
        }

        for (int row = 0; row < TileSize - 1; row++)
        {
            for (int column = 0; column < TileSize - 1; column++)
            {
                int topLeft = (row * TileSize) + column;
                int topRight = topLeft + 1;
                int bottomLeft = topLeft + TileSize;
                int bottomRight = bottomLeft + 1;
                primitive.AddTriangle(vertices[topLeft], vertices[bottomLeft], vertices[bottomRight]);
                primitive.AddTriangle(vertices[topLeft], vertices[bottomRight], vertices[topRight]);
            }
        }

        SceneBuilder scene = new();
        scene.AddRigidMesh(mesh, Matrix4x4.Identity);
        scene.ToGltf2().SaveGLB(outputPath);
    }

    private static Vector3 ComputeNormal(IReadOnlyList<float> heights, int x, int y)
    {
        int leftX = Math.Max(x - 1, 0);
        int rightX = Math.Min(x + 1, TileSize - 1);
        int upY = Math.Max(y - 1, 0);
        int downY = Math.Min(y + 1, TileSize - 1);

        float left = heights[(y * TileSize) + leftX];
        float right = heights[(y * TileSize) + rightX];
        float up = heights[(upY * TileSize) + x];
        float down = heights[(downY * TileSize) + x];

        Vector3 normal = new(-(right - left), 2f, -(down - up));
        if (normal.LengthSquared() <= float.Epsilon)
            return Vector3.UnitY;

        return Vector3.Normalize(normal);
    }
}