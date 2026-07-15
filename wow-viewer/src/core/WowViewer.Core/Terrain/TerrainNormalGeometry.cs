using System.Numerics;

namespace WowViewer.Core.Maps;

public readonly record struct TerrainNormalAgreementReport(
    int ComparedVertexCount,
    double MeanDot,
    double MeanAngularErrorDegrees);

/// <summary>Numeric mesh-normal derivation and MCNR agreement metrics for real MCVT vertices.</summary>
public static class TerrainNormalGeometry
{
    /// <summary>
    /// Converts an ADT grid-space XYZ normal into the renderer's world axes.
    /// Terrain grid X advances along -world Y and grid Y advances along -world X.
    /// </summary>
    public static Vector3 TransformAdtNormalToRenderer(Vector3 adtNormal)
    {
        Vector3 rendererNormal = new(-adtNormal.Y, -adtNormal.X, adtNormal.Z);
        return rendererNormal.LengthSquared() > 1e-10f
            ? Vector3.Normalize(rendererNormal)
            : Vector3.UnitZ;
    }

    public static TerrainNormalAgreementReport CompareWithNativeDenseNormals(
        TerrainVertexLattice terrain,
        float[,,] nativeNormalXyz257,
        bool[,] nativePresent257)
    {
        ArgumentNullException.ThrowIfNull(terrain);
        ArgumentNullException.ThrowIfNull(nativeNormalXyz257);
        ArgumentNullException.ThrowIfNull(nativePresent257);
        if (nativeNormalXyz257.GetLength(0) != TerrainVertexLattice.DenseGridSize
            || nativeNormalXyz257.GetLength(1) != TerrainVertexLattice.DenseGridSize
            || nativeNormalXyz257.GetLength(2) != 3)
        {
            throw new ArgumentException("Native normal arrays must have shape [257,257,3].", nameof(nativeNormalXyz257));
        }
        if (nativePresent257.GetLength(0) != TerrainVertexLattice.DenseGridSize
            || nativePresent257.GetLength(1) != TerrainVertexLattice.DenseGridSize)
        {
            throw new ArgumentException("Native normal masks must have shape [257,257].", nameof(nativePresent257));
        }

        double dotSum = 0.0;
        double angleSum = 0.0;
        int count = 0;
        for (int chunkY = 0; chunkY < TerrainVertexLattice.ChunksPerAxis; chunkY++)
        {
            for (int chunkX = 0; chunkX < TerrainVertexLattice.ChunksPerAxis; chunkX++)
            {
                Vector3[] geometric = ComputeChunkNormals(terrain, chunkX, chunkY);
                for (int sample = 0; sample < TerrainVertexLattice.SamplesPerChunk; sample++)
                {
                    if (!terrain.Present[chunkY, chunkX, sample])
                        continue;
                    TerrainVertexLattice.ResolveDenseCoordinates(chunkX, chunkY, sample, out int x, out int y);
                    if (!nativePresent257[y, x])
                        continue;
                    Vector3 native = new(
                        nativeNormalXyz257[y, x, 0],
                        nativeNormalXyz257[y, x, 1],
                        nativeNormalXyz257[y, x, 2]);
                    if (native.LengthSquared() <= 1e-10f)
                        continue;
                    native = Vector3.Normalize(native);
                    double dot = Math.Clamp(Vector3.Dot(geometric[sample], native), -1f, 1f);
                    dotSum += dot;
                    angleSum += Math.Acos(dot) * 180.0 / Math.PI;
                    count++;
                }
            }
        }

        return count == 0
            ? new TerrainNormalAgreementReport(0, 0.0, 0.0)
            : new TerrainNormalAgreementReport(count, dotSum / count, angleSum / count);
    }

    public static Vector3[] ComputeChunkNormals(TerrainVertexLattice terrain, int chunkX, int chunkY)
    {
        ArgumentNullException.ThrowIfNull(terrain);
        if ((uint)chunkX >= TerrainVertexLattice.ChunksPerAxis)
            throw new ArgumentOutOfRangeException(nameof(chunkX));
        if ((uint)chunkY >= TerrainVertexLattice.ChunksPerAxis)
            throw new ArgumentOutOfRangeException(nameof(chunkY));

        Vector3[] positions = new Vector3[TerrainVertexLattice.SamplesPerChunk];
        for (int sample = 0; sample < positions.Length; sample++)
        {
            positions[sample] = new Vector3(
                terrain.WorldX[chunkY, chunkX, sample],
                terrain.WorldY[chunkY, chunkX, sample],
                terrain.VertexZ[chunkY, chunkX, sample]);
        }

        Vector3[] accumulated = new Vector3[positions.Length];
        int[,] triangles = TerrainVertexLattice.ChunkTriangleIndices;
        for (int triangle = 0; triangle < triangles.GetLength(0); triangle++)
        {
            int i0 = triangles[triangle, 0];
            int i1 = triangles[triangle, 1];
            int i2 = triangles[triangle, 2];
            if (!terrain.Present[chunkY, chunkX, i0]
                || !terrain.Present[chunkY, chunkX, i1]
                || !terrain.Present[chunkY, chunkX, i2])
            {
                continue;
            }
            Vector3 normal = Vector3.Cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
            if (normal.LengthSquared() <= 1e-10f)
                continue;
            normal = Vector3.Normalize(normal);
            accumulated[i0] += normal;
            accumulated[i1] += normal;
            accumulated[i2] += normal;
        }

        for (int sample = 0; sample < accumulated.Length; sample++)
            accumulated[sample] = accumulated[sample].LengthSquared() > 1e-10f ? Vector3.Normalize(accumulated[sample]) : Vector3.UnitZ;
        return accumulated;
    }
}
