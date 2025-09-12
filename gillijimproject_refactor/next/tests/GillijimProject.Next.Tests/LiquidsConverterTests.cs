using System;
using GillijimProject.Next.Core.Domain.Liquids;
using GillijimProject.Next.Core.Transform.Liquids;
using Xunit;

namespace GillijimProject.Next.Tests;

public class LiquidsConverterTests
{
    [Fact]
    public void MclqToMh2o_ProducesSingleInstanceForRectangle_WithHeightsAndDepth()
    {
        // Arrange: 2x2 tiles at (2,3) of type River with non-zero heights/depth
        var heights = new float[MclqData.VertexGrid * MclqData.VertexGrid];
        var depth = new byte[MclqData.VertexGrid * MclqData.VertexGrid];
        var types = new MclqLiquidType[MclqData.TileGrid, MclqData.TileGrid];
        var flags = new MclqTileFlags[MclqData.TileGrid, MclqData.TileGrid];

        for (int ty = 3; ty < 5; ty++)
        for (int tx = 2; tx < 4; tx++)
            types[tx, ty] = MclqLiquidType.River;

        for (int vy = 3; vy <= 5; vy++)
        for (int vx = 2; vx <= 4; vx++)
        {
            int idx = VertexIndex(vx, vy);
            heights[idx] = vx + vy; // non-zero pattern
            depth[idx] = 7;
        }

        var mclq = new MclqData(heights, depth, types, flags);
        var liquidsOpts = new LiquidsOptions();

        // Act
        var mh2o = LiquidsConverter.MclqToMh2o(mclq, liquidsOpts);

        // Assert
        Assert.NotNull(mh2o);
        Assert.Single(mh2o.Instances);
        var inst = mh2o.Instances[0];
        Assert.Equal((byte)2, inst.XOffset);
        Assert.Equal((byte)3, inst.YOffset);
        Assert.Equal((byte)2, inst.Width);
        Assert.Equal((byte)2, inst.Height);
        Assert.Equal(LiquidVertexFormat.HeightDepth, inst.Lvf);
        Assert.NotNull(inst.HeightMap);
        Assert.NotNull(inst.DepthMap);
        Assert.Equal((inst.Width + 1) * (inst.Height + 1), inst.HeightMap!.Length);
        Assert.Equal((inst.Width + 1) * (inst.Height + 1), inst.DepthMap!.Length);

        // Check min/max heights align with our pattern in the 3x3 vertex block
        float expectedMin = float.PositiveInfinity, expectedMax = float.NegativeInfinity;
        for (int vy = 3; vy <= 5; vy++)
        for (int vx = 2; vx <= 4; vx++)
        {
            float hv = heights[VertexIndex(vx, vy)];
            if (hv < expectedMin) expectedMin = hv;
            if (hv > expectedMax) expectedMax = hv;
        }
        Assert.Equal(expectedMin, inst.MinHeightLevel);
        Assert.Equal(expectedMax, inst.MaxHeightLevel);

        // Check LiquidType mapping (defaults map River -> 4)
        Assert.Equal((ushort)4, inst.LiquidTypeId);
    }

    [Fact]
    public void Mh2oToMclq_RespectsPrecedence_AndMapsDeepToFatigue()
    {
        // Arrange
        var mapping = LiquidTypeMapping.CreateDefault();
        var opts = new LiquidsOptions { Mapping = mapping };

        // Magma instance: 4x4 at (1,1)
        var magma = new Mh2oInstance
        {
            LiquidTypeId = mapping.ToLiquidTypeId(MclqLiquidType.Magma),
            Lvf = LiquidVertexFormat.HeightDepth,
            XOffset = 1, YOffset = 1, Width = 4, Height = 4,
            HeightMap = CreateFilledFloatGrid(5, 5, 10f),
            DepthMap = CreateFilledByteGrid(5, 5, 2)
        };

        // River instance: 2x2 at (2,2) overlapping with magma
        var river = new Mh2oInstance
        {
            LiquidTypeId = mapping.ToLiquidTypeId(MclqLiquidType.River),
            Lvf = LiquidVertexFormat.HeightDepth,
            XOffset = 2, YOffset = 2, Width = 2, Height = 2,
            HeightMap = CreateFilledFloatGrid(3, 3, 20f),
            DepthMap = CreateFilledByteGrid(3, 3, 9)
        };

        var chunk = new Mh2oChunk
        {
            Attributes = new Mh2oAttributes(FishableMask: 0, DeepMask: SetTileBit(0UL, 2, 2))
        };
        chunk.Add(magma);
        chunk.Add(river);

        // Act
        var mclq = LiquidsConverter.Mh2oToMclq(chunk, opts);

        // Assert: Overlap tile (2,2) should be Magma due to precedence
        Assert.Equal(MclqLiquidType.Magma, mclq.Types[2, 2]);
        // Deep mask maps to Fatigue flag
        Assert.True((mclq.Flags[2, 2] & MclqTileFlags.Fatigue) != 0);

        // Height at a vertex within magma-only region should reflect magma value (10f)
        int vIdx = VertexIndex(2, 2);
        Assert.Equal(10f, mclq.Heights[vIdx]);
    }

    private static int VertexIndex(int vx, int vy) => vy * MclqData.VertexGrid + vx;

    private static float[] CreateFilledFloatGrid(int cols, int rows, float val)
    {
        var arr = new float[cols * rows];
        for (int i = 0; i < arr.Length; i++) arr[i] = val;
        return arr;
    }

    private static byte[] CreateFilledByteGrid(int cols, int rows, byte val)
    {
        var arr = new byte[cols * rows];
        for (int i = 0; i < arr.Length; i++) arr[i] = val;
        return arr;
    }

    private static ulong SetTileBit(ulong mask, int x, int y)
        => mask | (1UL << (y * 8 + x));
}
