using WowViewer.Core.Maps;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class LiquidBasicTypePackBuilderTests
{
    [Fact]
    public void Build_NoInputs_ReturnsNull()
    {
        Assert.Null(LiquidBasicTypePackBuilder.Build(null, null, null, null, null));
    }

    [Fact]
    public void Build_Mh2oPresence_ResolvesFromMh2oTypeMask()
    {
        bool[,] presence = new bool[4, 4];
        int[,] typeMask = new int[4, 4];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                presence[y, x] = true;
                typeMask[y, x] = (int)AdtLiquidBasicType.Magma;
            }

        byte[,]? result = LiquidBasicTypePackBuilder.Build(presence, typeMask, null, null, null);

        Assert.NotNull(result);
        Assert.Equal(4, result!.GetLength(0));
        Assert.Equal(4, result.GetLength(1));
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                Assert.Equal((byte)AdtLiquidBasicType.Magma, result[y, x]);
    }

    [Fact]
    public void Build_Mh2oPresence_ClampsOutOfRangeTypeMask()
    {
        bool[,] presence = new bool[2, 2];
        int[,] typeMask = new int[2, 2] { { 99, -1 }, { 1, 2 } };
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                presence[y, x] = true;

        byte[,]? result = LiquidBasicTypePackBuilder.Build(presence, typeMask, null, null, null);

        Assert.NotNull(result);
        Assert.Equal((byte)3, result![0, 0]);
        Assert.Equal((byte)0, result[0, 1]);
        Assert.Equal((byte)1, result[1, 0]);
        Assert.Equal((byte)2, result[1, 1]);
    }

    [Fact]
    public void Build_Mh2oPresence_Absent_ProducesNoLiquidSentinel()
    {
        bool[,] presence = new bool[2, 2];
        int[,] typeMask = new int[2, 2];
        presence[0, 0] = true;
        typeMask[0, 0] = 0;

        byte[,]? result = LiquidBasicTypePackBuilder.Build(presence, typeMask, null, null, null);

        Assert.NotNull(result);
        Assert.Equal((byte)0, result![0, 0]);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, result[0, 1]);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, result[1, 0]);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, result[1, 1]);
    }

    [Fact]
    public void Build_MclqPresence_ResolvesResolvedValues()
    {
        bool[,] presence = new bool[3, 3];
        int[,] typeMask = new int[3, 3];
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
            {
                presence[y, x] = true;
                typeMask[y, x] = (int)AdtLiquidBasicType.Slime;
            }

        byte[,]? result = LiquidBasicTypePackBuilder.Build(null, null, presence, typeMask, null);

        Assert.NotNull(result);
        Assert.Equal((byte)AdtLiquidBasicType.Slime, result![1, 1]);
    }

    [Fact]
    public void Build_MclqPresence_RawMclqLiquidType_MapsToCorrectBasicType()
    {
        bool[,] presence = new bool[2, 2];
        int[,] rawType = new int[2, 2] { { 4, 6 }, { 1, 0x0F } };
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                presence[y, x] = true;

        byte[,]? result = LiquidBasicTypePackBuilder.Build(null, null, presence, rawType, null);

        Assert.NotNull(result);
        Assert.Equal((byte)AdtLiquidBasicType.Water, result![0, 0]);          // 4 = River → Water
        Assert.Equal((byte)AdtLiquidBasicType.Magma, result[0, 1]);           // 6 = Magma → Magma
        Assert.Equal((byte)AdtLiquidBasicType.Ocean, result[1, 0]);           // 1 = Ocean → Ocean
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, result[1, 1]);        // 0x0F = DontRender → NoLiquid
    }

    [Fact]
    public void Build_McnkFlags_MagmaOnly_ProducesMagmaType()
    {
        int[,] mcnkFlags16 = new int[16, 16];
        mcnkFlags16[0, 0] = 0x10;

        byte[,]? result = LiquidBasicTypePackBuilder.Build(null, null, null, null, mcnkFlags16);

        Assert.NotNull(result);
        Assert.Equal(257, result!.GetLength(0));
        Assert.Equal(257, result.GetLength(1));
        for (int y = 0; y < 17; y++)
            for (int x = 0; x < 17; x++)
                Assert.Equal((byte)AdtLiquidBasicType.Magma, result[y, x]);
    }

    [Fact]
    public void Build_McnkFlags_Zero_ProducesNoLiquidSentinel()
    {
        int[,] mcnkFlags16 = new int[16, 16];

        byte[,]? result = LiquidBasicTypePackBuilder.Build(null, null, null, null, mcnkFlags16);

        Assert.NotNull(result);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, result![0, 0]);
    }

    [Fact]
    public void Build_PrioritizesMh2oOverMclqAndMcnk()
    {
        bool[,] mh2oPresence = new bool[2, 2];
        int[,] mh2oType = new int[2, 2] { { 0, 0 }, { 0, 0 } };
        mh2oPresence[0, 0] = true;
        mh2oType[0, 0] = (int)AdtLiquidBasicType.Ocean;

        bool[,] mclqPresence = new bool[2, 2];
        int[,] mclqType = new int[2, 2];
        mclqPresence[0, 0] = true;
        mclqType[0, 0] = (int)AdtLiquidBasicType.Slime;

        int[,] mcnkFlags16 = new int[16, 16];
        mcnkFlags16[0, 0] = 0x10;

        byte[,]? result = LiquidBasicTypePackBuilder.Build(mh2oPresence, mh2oType, mclqPresence, mclqType, mcnkFlags16);

        Assert.NotNull(result);
        Assert.Equal((byte)AdtLiquidBasicType.Ocean, result![0, 0]);
    }
}
