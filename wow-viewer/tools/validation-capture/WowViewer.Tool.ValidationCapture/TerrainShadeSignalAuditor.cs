using System.Numerics;
using System.Text.Json;
using WowViewer.App;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Tools.ValidationCapture;

internal static class TerrainShadeSignalAuditor
{
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;
    private const float HalfStepSize = ChunkSize / 16f;
    private const float MapOrigin = 32f * TileSize;

    public static string Write(
        string pngPath,
        WowViewerWorldRuntimeFrameResult frame,
        ValidationCaptureCameraFrame camera,
        byte[] rgbaPixels,
        int width,
        int height,
        bool sourceOriginBottomLeft)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pngPath);
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(rgbaPixels);

        List<double> luminance = [];
        List<double> vertexZ = [];
        List<double> slope = [];
        List<double> expectedLighting = [];
        Matrix4x4 viewProjection = camera.View * camera.Projection;
        Vector3 lightDirection = Vector3.Normalize(ValidationTerrainShadeContract.LightDirection);

        foreach (var chunk in frame.TerrainTileData.Chunks)
        {
            if (!chunk.HasHeights || chunk.Heights is null || chunk.Heights.Length != TerrainVertexLattice.SamplesPerChunk)
                continue;

            Vector3[] positions = BuildPositions(frame.SelectedTileX, frame.SelectedTileY, chunk.IndexX, chunk.IndexY, chunk.Heights);
            Vector3[] normals = ComputeNormals(positions);
            for (int index = 0; index < positions.Length; index++)
            {
                Vector4 clip = Vector4.Transform(new Vector4(positions[index], 1f), viewProjection);
                if (MathF.Abs(clip.W) < 1e-6f)
                    continue;
                float ndcX = clip.X / clip.W;
                float ndcY = clip.Y / clip.W;
                if (ndcX is < -1f or > 1f || ndcY is < -1f or > 1f)
                    continue;

                int pixelX = Math.Clamp((int)MathF.Round(((ndcX + 1f) * 0.5f) * (width - 1)), 0, width - 1);
                int topOriginY = Math.Clamp((int)MathF.Round((1f - ((ndcY + 1f) * 0.5f)) * (height - 1)), 0, height - 1);
                int sourceY = sourceOriginBottomLeft ? height - 1 - topOriginY : topOriginY;
                int offset = ((sourceY * width) + pixelX) * 4;
                double luma = ((0.2126 * rgbaPixels[offset]) + (0.7152 * rgbaPixels[offset + 1]) + (0.0722 * rgbaPixels[offset + 2])) / 255.0;
                double ndotl = Math.Max(Vector3.Dot(normals[index], lightDirection), 0f);

                luminance.Add(luma);
                vertexZ.Add(positions[index].Z);
                slope.Add(1.0 - Math.Clamp(normals[index].Z, -1f, 1f));
                expectedLighting.Add(ndotl);
            }
        }

        var report = new
        {
            schema = "wow-viewer.terrain-shade-signal-audit.v1",
            renderer_contract = ValidationTerrainShadeContract.Revision,
            sampled_vertex_count = luminance.Count,
            correlation_luminance_to_vertex_z = Pearson(luminance, vertexZ),
            correlation_luminance_to_slope = Pearson(luminance, slope),
            correlation_luminance_to_directional_ndotl = Pearson(luminance, expectedLighting),
            note = "Guidance-only upper-bound probe; PNG is not a deployment input or mesh target.",
        };

        string reportPath = Path.ChangeExtension(pngPath, ".signal-audit.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        return reportPath;
    }

    internal static double Pearson(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        if (left.Count != right.Count || left.Count < 2)
            return 0.0;
        double leftMean = left.Average();
        double rightMean = right.Average();
        double covariance = 0.0;
        double leftVariance = 0.0;
        double rightVariance = 0.0;
        for (int index = 0; index < left.Count; index++)
        {
            double dl = left[index] - leftMean;
            double dr = right[index] - rightMean;
            covariance += dl * dr;
            leftVariance += dl * dl;
            rightVariance += dr * dr;
        }
        double denominator = Math.Sqrt(leftVariance * rightVariance);
        return denominator > 1e-12 ? covariance / denominator : 0.0;
    }

    private static Vector3[] BuildPositions(int tileX, int tileY, int chunkX, int chunkY, IReadOnlyList<float> heights)
    {
        Vector3[] positions = new Vector3[heights.Count];
        for (int index = 0; index < positions.Length; index++)
        {
            TerrainVertexLattice.ResolveLocalHalfStepCoordinates(index, out int localX, out int localY);
            positions[index] = new Vector3(
                MapOrigin - (tileY * TileSize) - (chunkY * ChunkSize) - (localY * HalfStepSize),
                MapOrigin - (tileX * TileSize) - (chunkX * ChunkSize) - (localX * HalfStepSize),
                heights[index]);
        }
        return positions;
    }

    private static Vector3[] ComputeNormals(IReadOnlyList<Vector3> positions)
    {
        Vector3[] accumulated = new Vector3[positions.Count];
        int[,] triangles = TerrainVertexLattice.ChunkTriangleIndices;
        for (int triangle = 0; triangle < triangles.GetLength(0); triangle++)
        {
            int i0 = triangles[triangle, 0];
            int i1 = triangles[triangle, 1];
            int i2 = triangles[triangle, 2];
            Vector3 normal = Vector3.Cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
            if (normal.LengthSquared() <= 1e-10f)
                continue;
            normal = Vector3.Normalize(normal);
            accumulated[i0] += normal;
            accumulated[i1] += normal;
            accumulated[i2] += normal;
        }

        for (int index = 0; index < accumulated.Length; index++)
            accumulated[index] = accumulated[index].LengthSquared() > 1e-10f ? Vector3.Normalize(accumulated[index]) : Vector3.UnitZ;
        return accumulated;
    }
}
