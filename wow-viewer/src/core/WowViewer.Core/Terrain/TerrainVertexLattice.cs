namespace WowViewer.Core.Maps;

/// <summary>
/// Canonical per-tile ADT terrain mesh data. Each of the 16x16 MCNK chunks
/// contributes exactly 145 MCVT vertices in native file order. The dense
/// 257x257 grid is only a coordinate view; positions outside
/// <see cref="DenseValidMask"/> are not terrain vertices.
/// </summary>
public sealed class TerrainVertexLattice
{
    public const int ChunksPerAxis = 16;
    public const int SamplesPerChunk = 145;
    public const int HalfStepsPerChunk = 16;
    public const int DenseGridSize = (ChunksPerAxis * HalfStepsPerChunk) + 1;

    public TerrainVertexLattice(
        float[,,] vertexZ,
        float[,,] worldX,
        float[,,] worldY,
        bool[,,] present,
        bool[,] denseValidMask)
    {
        ArgumentNullException.ThrowIfNull(vertexZ);
        ArgumentNullException.ThrowIfNull(present);
        ArgumentNullException.ThrowIfNull(denseValidMask);
        ValidateShape(vertexZ, nameof(vertexZ));
        ValidateShape(worldX, nameof(worldX));
        ValidateShape(worldY, nameof(worldY));
        ValidateShape(present, nameof(present));
        if (denseValidMask.GetLength(0) != DenseGridSize || denseValidMask.GetLength(1) != DenseGridSize)
            throw new ArgumentException($"Dense vertex masks must be {DenseGridSize}x{DenseGridSize}.", nameof(denseValidMask));

        VertexZ = vertexZ;
        WorldX = worldX;
        WorldY = worldY;
        Present = present;
        DenseValidMask = denseValidMask;
    }

    /// <summary>Absolute world-space Z in [chunkY, chunkX, native MCVT sample].</summary>
    public float[,,] VertexZ { get; }

    /// <summary>Fixed world X in [chunkY, chunkX, native MCVT sample].</summary>
    public float[,,] WorldX { get; }

    /// <summary>Fixed world Y in [chunkY, chunkX, native MCVT sample].</summary>
    public float[,,] WorldY { get; }

    /// <summary>True only when that native MCVT sample was present in the source.</summary>
    public bool[,,] Present { get; }

    /// <summary>True at 257x257 half-step coordinates backed by at least one real MCVT vertex.</summary>
    public bool[,] DenseValidMask { get; }

    /// <summary>Canonical 256-triangle MCNK topology as local native MCVT indices.</summary>
    public static int[,] ChunkTriangleIndices { get; } = BuildChunkTriangleIndices();

    public bool TryGetVertexAtDenseCoordinates(int sampleX, int sampleY, out float vertexZ)
    {
        if ((uint)sampleX >= DenseGridSize || (uint)sampleY >= DenseGridSize)
            throw new ArgumentOutOfRangeException(sampleX is < 0 or >= DenseGridSize ? nameof(sampleX) : nameof(sampleY));
        if ((sampleX & 1) != (sampleY & 1) || !DenseValidMask[sampleY, sampleX])
        {
            vertexZ = 0f;
            return false;
        }

        int chunkX = Math.Min(sampleX / HalfStepsPerChunk, ChunksPerAxis - 1);
        int chunkY = Math.Min(sampleY / HalfStepsPerChunk, ChunksPerAxis - 1);
        int localX = sampleX - (chunkX * HalfStepsPerChunk);
        int localY = sampleY - (chunkY * HalfStepsPerChunk);
        int sampleIndex = ResolveSampleIndex(localX, localY);
        if (!Present[chunkY, chunkX, sampleIndex])
        {
            vertexZ = 0f;
            return false;
        }

        vertexZ = VertexZ[chunkY, chunkX, sampleIndex];
        return true;
    }

    public static void ResolveDenseCoordinates(int chunkX, int chunkY, int sampleIndex, out int sampleX, out int sampleY)
    {
        if ((uint)chunkX >= ChunksPerAxis)
            throw new ArgumentOutOfRangeException(nameof(chunkX));
        if ((uint)chunkY >= ChunksPerAxis)
            throw new ArgumentOutOfRangeException(nameof(chunkY));
        ResolveLocalHalfStepCoordinates(sampleIndex, out int localX, out int localY);
        sampleX = (chunkX * HalfStepsPerChunk) + localX;
        sampleY = (chunkY * HalfStepsPerChunk) + localY;
    }

    public static void ResolveLocalHalfStepCoordinates(int sampleIndex, out int localX, out int localY)
    {
        if ((uint)sampleIndex >= SamplesPerChunk)
            throw new ArgumentOutOfRangeException(nameof(sampleIndex));

        int remaining = sampleIndex;
        for (int row = 0; row < 17; row++)
        {
            bool inner = (row & 1) != 0;
            int rowSize = inner ? 8 : 9;
            if (remaining < rowSize)
            {
                localX = inner ? (remaining * 2) + 1 : remaining * 2;
                localY = inner ? row : row;
                return;
            }
            remaining -= rowSize;
        }

        throw new InvalidOperationException($"Could not resolve MCVT sample index {sampleIndex}.");
    }

    public static int ResolveSampleIndex(int localX, int localY)
    {
        if ((uint)localX > HalfStepsPerChunk || (uint)localY > HalfStepsPerChunk)
            throw new ArgumentOutOfRangeException(localX is < 0 or > HalfStepsPerChunk ? nameof(localX) : nameof(localY));
        if ((localX & 1) != (localY & 1))
            throw new ArgumentException("MCVT vertices occupy only even/even outer nodes or odd/odd inner nodes.");

        return (localY & 1) == 0
            ? ((localY / 2) * 17) + (localX / 2)
            : (((localY - 1) / 2) * 17) + 9 + ((localX - 1) / 2);
    }

    private static void ValidateShape(Array array, string parameterName)
    {
        if (array.Rank != 3
            || array.GetLength(0) != ChunksPerAxis
            || array.GetLength(1) != ChunksPerAxis
            || array.GetLength(2) != SamplesPerChunk)
        {
            throw new ArgumentException(
                $"Raw MCVT arrays must have shape [{ChunksPerAxis},{ChunksPerAxis},{SamplesPerChunk}].",
                parameterName);
        }
    }

    private static int[,] BuildChunkTriangleIndices()
    {
        int[,] triangles = new int[256, 3];
        int triangle = 0;
        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                int topLeft = (cellY * 17) + cellX;
                int topRight = topLeft + 1;
                int bottomLeft = ((cellY + 1) * 17) + cellX;
                int bottomRight = bottomLeft + 1;
                int center = (cellY * 17) + 9 + cellX;
                Add(center, topRight, topLeft);
                Add(center, bottomRight, topRight);
                Add(center, bottomLeft, bottomRight);
                Add(center, topLeft, bottomLeft);
            }
        }
        return triangles;

        void Add(int a, int b, int c)
        {
            triangles[triangle, 0] = a;
            triangles[triangle, 1] = b;
            triangles[triangle, 2] = c;
            triangle++;
        }
    }
}
