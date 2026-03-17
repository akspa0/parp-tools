using MdxViewer.Rendering;
using MdxViewer.Terrain;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class WdlPreviewDataBuilderTests
{
    [Fact]
    public void BuildFromParsed_CapturesTileMaskHeightsAndPreviewPixels()
    {
        var parsed = new WdlParser.WdlData();
        parsed.Tiles[0] = CreateTile(fillHeight: 10, centerHeight: 12, minZ: 8, maxZ: 16);
        parsed.Tiles[1] = CreateTile(fillHeight: 40, centerHeight: 48, minZ: 36, maxZ: 52);

        var preview = WdlPreviewDataBuilder.BuildFromParsed("Azeroth", parsed);

        Assert.Equal(64, preview.Width);
        Assert.Equal(64, preview.Height);
        Assert.Equal(64 * 64 * 4, preview.PreviewRgba.Length);
        Assert.Equal(64 * 64, preview.TileCenterHeights.Length);
        Assert.Equal(64 * 64, preview.TileDataMask.Length);

        Assert.Equal((byte)1, preview.TileDataMask[0]);
        Assert.Equal((byte)1, preview.TileDataMask[1]);
        Assert.Equal((byte)0, preview.TileDataMask[2]);
        Assert.Equal(12f, preview.TileCenterHeights[0]);
        Assert.Equal(48f, preview.TileCenterHeights[1]);

        Assert.Equal((byte)255, preview.PreviewRgba[3]);
        Assert.Equal((byte)255, preview.PreviewRgba[7]);

        int missingPixelOffset = 2 * 4;
        Assert.Equal((byte)25, preview.PreviewRgba[missingPixelOffset + 0]);
        Assert.Equal((byte)25, preview.PreviewRgba[missingPixelOffset + 1]);
        Assert.Equal((byte)25, preview.PreviewRgba[missingPixelOffset + 2]);
        Assert.Equal((byte)255, preview.PreviewRgba[missingPixelOffset + 3]);
    }

    [Fact]
    public void BuildFromParsed_UsesScreenMajorPreviewOrientation()
    {
        var parsed = new WdlParser.WdlData();
        parsed.Tiles[2 * 64 + 5] = CreateTile(fillHeight: 25, centerHeight: 77, minZ: 20, maxZ: 80);

        var preview = WdlPreviewDataBuilder.BuildFromParsed("Azeroth", parsed);

        int previewIndex = 2 * 64 + 5;
        Assert.Equal((byte)1, preview.TileDataMask[previewIndex]);
        Assert.Equal(77f, preview.TileCenterHeights[previewIndex]);
    }

    [Fact]
    public void PreviewTileToSourceTile_ConvertsScreenCoordinatesToTerrainGrid()
    {
        var sourceTile = WdlPreviewRenderer.PreviewTileToSourceTile(previewTileX: 5, previewTileY: 2);

        Assert.Equal(2, sourceTile.tileX);
        Assert.Equal(5, sourceTile.tileY);
    }

    [Fact]
    public void GetTileSpawnPosition_UsesActiveTerrainGridCenterWithoutAxisSwap()
    {
        var position = WdlPreviewRenderer.GetTileSpawnPosition(tileX: 10, tileY: 20, height: 345f);

        Assert.Equal(WoWConstants.MapOrigin - ((10.5f) * WoWConstants.ChunkSize), position.X);
        Assert.Equal(WoWConstants.MapOrigin - ((20.5f) * WoWConstants.ChunkSize), position.Y);
        Assert.Equal(445f, position.Z);
    }

    [Fact]
    public void GetTileSpawnPosition_StaysWithinActiveWorldBounds()
    {
        var position = WdlPreviewRenderer.GetTileSpawnPosition(tileX: 63, tileY: 63, height: 0f);

        Assert.InRange(position.X, -WoWConstants.MapOrigin, WoWConstants.MapOrigin);
        Assert.InRange(position.Y, -WoWConstants.MapOrigin, WoWConstants.MapOrigin);
    }

    private static WdlParser.WdlTile CreateTile(short fillHeight, short centerHeight, float minZ, float maxZ)
    {
        var tile = new WdlParser.WdlTile
        {
            HasData = true,
            MinZ = minZ,
            MaxZ = maxZ,
        };

        for (int row = 0; row < 17; row++)
        {
            for (int col = 0; col < 17; col++)
            {
                tile.Height17[row, col] = fillHeight;
                tile.Heights[row * 17 + col] = fillHeight;
            }
        }

        tile.Height17[8, 8] = centerHeight;
        tile.Heights[8 * 17 + 8] = centerHeight;

        int innerBase = 17 * 17;
        for (int row = 0; row < 16; row++)
        {
            for (int col = 0; col < 16; col++)
            {
                tile.Height16[row, col] = fillHeight;
                tile.Heights[innerBase + row * 16 + col] = fillHeight;
            }
        }

        return tile;
    }
}