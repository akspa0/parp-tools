using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TerrainNormalGeometryTests
{
    [Fact]
    public void CompareWithNativeDenseNormals_FlatChunkMatchesUnitZ()
    {
        float[,,] z = new float[16, 16, 145];
        float[,,] worldX = new float[16, 16, 145];
        float[,,] worldY = new float[16, 16, 145];
        bool[,,] present = new bool[16, 16, 145];
        bool[,] dense = new bool[257, 257];
        float[,,] native = new float[257, 257, 3];
        bool[,] nativePresent = new bool[257, 257];

        for (int sample = 0; sample < 145; sample++)
        {
            TerrainVertexLattice.ResolveLocalHalfStepCoordinates(sample, out int x, out int y);
            worldX[0, 0, sample] = -y;
            worldY[0, 0, sample] = -x;
            present[0, 0, sample] = true;
            dense[y, x] = true;
            native[y, x, 2] = 1f;
            nativePresent[y, x] = true;
        }

        TerrainVertexLattice lattice = new(z, worldX, worldY, present, dense);
        TerrainNormalAgreementReport report = TerrainNormalGeometry.CompareWithNativeDenseNormals(lattice, native, nativePresent);

        Assert.Equal(145, report.ComparedVertexCount);
        Assert.Equal(1.0, report.MeanDot, 8);
        Assert.Equal(0.0, report.MeanAngularErrorDegrees, 8);
    }
}
