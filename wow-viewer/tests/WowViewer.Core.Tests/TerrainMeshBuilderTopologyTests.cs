using System.Reflection;
using WowViewer.Core.Renderer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class TerrainMeshBuilderTopologyTests
{
    [Fact]
    public void BuildIndices_NoHoles_Generates256CellFanTopology()
    {
        int[] indices = InvokeBuildIndices(0);

        Assert.Equal(64 * 4 * 3, indices.Length);
        Assert.All(indices, static idx => Assert.InRange(idx, 0, 144));
    }

    [Fact]
    public void BuildIndices_WithSingleHoleGroup_RemovesOnlyAffected2x2Cells()
    {
        int[] indices = InvokeBuildIndices(0x0001);

        // One 2x2 hole group removes 4 cells -> 4 * 12 indices removed.
        Assert.Equal((64 - 4) * 4 * 3, indices.Length);
    }

    [Fact]
    public void BuildIndices_FullHoleMask_GeneratesNoTriangles()
    {
        int[] indices = InvokeBuildIndices(0xFFFF);

        Assert.Empty(indices);
    }

    [Fact]
    public void GetVertexPosition_UsesInterleaved145Layout_NotFlat17x17Layout()
    {
        (int row, int col, bool isInner) v0 = InvokeGetVertexPosition(0);
        (int row, int col, bool isInner) v8 = InvokeGetVertexPosition(8);
        (int row, int col, bool isInner) v9 = InvokeGetVertexPosition(9);
        (int row, int col, bool isInner) v16 = InvokeGetVertexPosition(16);
        (int row, int col, bool isInner) v17 = InvokeGetVertexPosition(17);
        (int row, int col, bool isInner) v144 = InvokeGetVertexPosition(144);

        Assert.Equal((0, 0, false), v0);
        Assert.Equal((0, 8, false), v8);
        Assert.Equal((1, 0, true), v9);
        Assert.Equal((1, 7, true), v16);
        Assert.Equal((2, 0, false), v17);
        Assert.Equal((16, 8, false), v144);
    }

    private static int[] InvokeBuildIndices(ushort holeMask)
    {
        MethodInfo method = typeof(TerrainMeshBuilder).GetMethod("BuildIndices", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("BuildIndices method was not found.");

        object? result = method.Invoke(null, [holeMask]);
        return Assert.IsType<int[]>(result);
    }

    private static (int row, int col, bool isInner) InvokeGetVertexPosition(int index)
    {
        MethodInfo method = typeof(TerrainMeshBuilder).GetMethod("GetVertexPosition", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("GetVertexPosition method was not found.");

        object?[] args = [index, 0, 0, false];
        method.Invoke(null, args);

        int row = Assert.IsType<int>(args[1]);
        int col = Assert.IsType<int>(args[2]);
        bool isInner = Assert.IsType<bool>(args[3]);
        return (row, col, isInner);
    }
}
