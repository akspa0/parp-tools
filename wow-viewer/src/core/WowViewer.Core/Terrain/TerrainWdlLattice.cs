namespace WowViewer.Core.Maps;

/// <summary>
/// Mesh-native WDL MARE sampling contract: 17x17 outer samples at dense
/// half-step coordinates 0,16,...,256 and 16x16 inner samples at 8,24,...,248.
/// Both families are sampled from real MCVT vertices, not a stride-8 raster.
/// </summary>
public sealed class TerrainWdlLattice
{
    public const int OuterDimension = 17;
    public const int InnerDimension = 16;
    public const int SampleCount = (OuterDimension * OuterDimension) + (InnerDimension * InnerDimension);

    private TerrainWdlLattice(float[,] outer17, float[,] inner16, bool[,] outerPresent, bool[,] innerPresent)
    {
        Outer17 = outer17;
        Inner16 = inner16;
        OuterPresent = outerPresent;
        InnerPresent = innerPresent;
    }

    public float[,] Outer17 { get; }

    public float[,] Inner16 { get; }

    public bool[,] OuterPresent { get; }

    public bool[,] InnerPresent { get; }

    public static TerrainWdlLattice FromTerrainVertices(TerrainVertexLattice terrain)
    {
        ArgumentNullException.ThrowIfNull(terrain);
        float[,] outer = new float[OuterDimension, OuterDimension];
        float[,] inner = new float[InnerDimension, InnerDimension];
        bool[,] outerPresent = new bool[OuterDimension, OuterDimension];
        bool[,] innerPresent = new bool[InnerDimension, InnerDimension];

        for (int row = 0; row < OuterDimension; row++)
        {
            for (int column = 0; column < OuterDimension; column++)
            {
                int denseY = row * TerrainVertexLattice.HalfStepsPerChunk;
                int denseX = column * TerrainVertexLattice.HalfStepsPerChunk;
                outerPresent[row, column] = terrain.TryGetVertexAtDenseCoordinates(denseX, denseY, out outer[row, column]);
            }
        }

        for (int row = 0; row < InnerDimension; row++)
        {
            for (int column = 0; column < InnerDimension; column++)
            {
                int denseY = (row * TerrainVertexLattice.HalfStepsPerChunk) + 8;
                int denseX = (column * TerrainVertexLattice.HalfStepsPerChunk) + 8;
                innerPresent[row, column] = terrain.TryGetVertexAtDenseCoordinates(denseX, denseY, out inner[row, column]);
            }
        }

        return new TerrainWdlLattice(outer, inner, outerPresent, innerPresent);
    }
}
