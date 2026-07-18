using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 112 T006: the mcnk_flags_16 dataset signal was 0% populated across the entire real
/// 0.5.3.3368 corpus because the Alpha path parsed every chunk's MCNK header flags but only
/// retained them for liquid chunks. These tests pin both halves of the fix: the reader carries a
/// full per-chunk flags grid on AlphaTileData, and the tensor-pack builder forwards it (with the
/// signal registered) so the serializer's existing unconditional write finally has real data.
/// </summary>
public sealed class AlphaMcnkFlagsTests
{
    [Fact]
    public void Build_ForwardsMcnkFlagsGridAndRegistersTheSignal()
    {
        var flags16 = new int[16, 16];
        flags16[7, 3] = 0x0C; // [chunkY, chunkX] -- row-major, LK ReadMcnkFlags convention
        flags16[0, 0] = 0x01;

        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(1,2)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: Array.Empty<AlphaLiquidChunk>(),
            mcnkFlags16: flags16);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 1, 2);

        Assert.NotNull(pack.McnkFlags16);
        Assert.Equal(0x0C, pack.McnkFlags16![7, 3]);
        Assert.Equal(0x01, pack.McnkFlags16[0, 0]);
        Assert.Equal(0, pack.McnkFlags16[5, 5]);
        Assert.Contains("mcnk_flags_16", pack.AvailableSignals);
    }

    [Fact]
    public void Build_AllZeroFlagsGridDoesNotClaimTheSignal()
    {
        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: Array.Empty<AlphaLiquidChunk>(),
            mcnkFlags16: new int[16, 16]);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 0, 0);

        // The grid is still forwarded (zeros are honest data), but the signal is not claimed.
        Assert.NotNull(pack.McnkFlags16);
        Assert.DoesNotContain("mcnk_flags_16", pack.AvailableSignals);
    }

    [Fact]
    public void WrittenAlphaWdt_ReadsBackPerChunkHeaderFlags_NotOnlyForLiquidChunks()
    {
        // The frozen AlphaWdtWriter derives header flags from liquid presence, so a liquid chunk
        // at (chunkX=3, chunkY=2) yields a nonzero MCNK header flag there while plain terrain
        // chunks stay zero -- exactly the byte-level pattern the reader must now surface for
        // EVERY chunk, not just re-derive from AlphaLiquidChunk entries.
        var liquid = new AlphaLiquidChunk(
            2 * 16 + 3, 3, 2, MinHeight: 10f, MaxHeight: 12f,
            TileFlags: null, McnkFlags: 0x08, Heights: new float[81]);

        AlphaTileData tile = new(
            sourcePath: "written.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: new[] { "terrain_a.blp" },
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: new[] { liquid });

        byte[] wdt = AlphaWdtWriter.Build("flagstest", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        Assert.True(AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? roundTrip));
        Assert.NotNull(roundTrip!.McnkFlags16);
        Assert.NotEqual(0, roundTrip.McnkFlags16![2, 3]); // liquid chunk carries its header flags
        Assert.Equal(0, roundTrip.McnkFlags16[9, 9]);      // plain chunk honestly zero
    }
}
