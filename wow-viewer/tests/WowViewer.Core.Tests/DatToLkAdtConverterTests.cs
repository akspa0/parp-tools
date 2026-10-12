using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 247 US3: DAT -> LK v18 ADT conversion. Built on a synthetic v22-shaped tile so the tests are portable;
/// the real corpus lives outside the repo. The v22 shape is what matters here: it omits empty ACNK, carries
/// area ids and a live ASHD, and its AMAP is an encoding the project cannot decode.
/// </summary>
public sealed class DatToLkAdtConverterTests
{
    private const int PresentChunks = 6;

    [Fact]
    public void Convert_WritesAFull256ChunkGrid_EvenWhenTheSourceOmitsEmptyChunks()
    {
        AdtAhdrTile tile = BuildV22Tile();
        var report = new DatToLkConversionReport();

        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, report);

        Assert.Equal(256, adt.Chunks.Count);
        Assert.Equal(PresentChunks, report.ChunksWritten);
        Assert.Equal(256 - PresentChunks, report.ChunksSynthesizedEmpty);
    }

    [Fact]
    public void Convert_AddressesChunksByAcnkIndex_NotByOrdinalPosition()
    {
        // The synthetic source stores chunk (3, 2) only, at ordinal 0. Reading ordinally would land it at (0, 0).
        var tile = BuildTile([new AdtAhdrChunk
        {
            IndexX = 3,
            IndexY = 2,
            HeaderRaw = BuildHeader(areaId: 4242, holes: 0),
            Layers = [new AdtAhdrLayer(1, 0, null)],
        }]);

        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, new DatToLkConversionReport());

        LkMcnkData at32 = adt.Chunks.Single(c => c.IndexX == 3 && c.IndexY == 2);
        LkMcnkData at00 = adt.Chunks.Single(c => c.IndexX == 0 && c.IndexY == 0);
        Assert.Equal(4242, at32.AreaId);
        Assert.Equal(0, at00.AreaId);
        Assert.Equal(1, at32.NLayers);
        Assert.Equal(0, at00.NLayers);
    }

    [Fact]
    public void Convert_CarriesAreaIdsAndLiveShadows()
    {
        AdtAhdrTile tile = BuildV22Tile();
        var report = new DatToLkConversionReport();

        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, report);

        Assert.Equal(PresentChunks, report.AreaIdsCarried);
        Assert.All(adt.Chunks.Where(static c => c.NLayers > 0), c => Assert.Equal(3519, c.AreaId));

        // Only the chunks whose ASHD is non-zero become MCSH, and they set the MCNK has-shadow flag.
        Assert.Equal(PresentChunks / 2, report.ShadowMapsCarried);
        foreach (LkMcnkData c in adt.Chunks.Where(static c => c.ShadowMap is not null))
        {
            Assert.Equal(512, c.ShadowMap!.Length);
            Assert.Equal(0x01, c.Flags & 0x01);
        }
    }

    [Fact]
    public void Convert_DropsUpperLayersWhenAlphaCannotBeDecoded()
    {
        // v22: layer 0 has no AMAP and the rest are an unidentified encoding, so upper layers must not be
        // emitted -- an alpha-less upper layer is opaque and would hide the base layer entirely.
        AdtAhdrTile tile = BuildV22Tile();
        var report = new DatToLkConversionReport();

        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, report);

        Assert.Equal(0, report.AlphaMapsWritten);
        Assert.Equal(PresentChunks * 2, report.LayersDroppedNoAlpha);
        Assert.All(adt.Chunks.Where(static c => c.NLayers > 0), c =>
        {
            Assert.Equal(1, c.NLayers);
            Assert.Null(c.AlphaMapData);
        });
        Assert.Contains(report.Notes, n => n.Contains("unidentified encoding", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Convert_ConvertsInchesToYards()
    {
        AdtAhdrTile tile = BuildV22Tile();

        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, new DatToLkConversionReport());

        // Every source height is 3600 inches = 100 yards, so the base is 100 and the offsets are 0.
        LkMcnkData chunk = adt.Chunks.First(static c => c.NLayers > 0);
        Assert.Equal(100f, chunk.BaseHeight, 3);
        Assert.All(chunk.Heights, h => Assert.Equal(0f, h, 3));
    }

    [Fact]
    public void WrittenAdt_ReadsBackWithItsTerrainAndAreaIds()
    {
        AdtAhdrTile tile = BuildV22Tile();
        LkAdtData adt = DatToLkAdtConverter.Convert(tile, 24, 38, "Test", null, new DatToLkConversionReport());

        byte[] bytes = LkAdtWriter.Build(adt);
        LkAdtData round = LkAdtReader.Read(bytes, null, null, 24, 38);

        Assert.Equal(256, round.Chunks.Count);
        LkMcnkData source = adt.Chunks.Single(c => c is { IndexX: 0, IndexY: 0 });
        LkMcnkData target = round.Chunks.Single(c => c is { IndexX: 0, IndexY: 0 });
        Assert.Equal(source.AreaId, target.AreaId);
        Assert.Equal(source.BaseHeight, target.BaseHeight, 3);
        Assert.Equal(source.Heights.Length, target.Heights.Length);
    }

    private static AdtAhdrTile BuildV22Tile()
    {
        var chunks = new List<AdtAhdrChunk>();
        for (int i = 0; i < PresentChunks; i++)
        {
            byte[] shadow = new byte[512];
            if (i % 2 == 0)
                shadow[i] = 0xFF; // half the chunks carry real shadow

            chunks.Add(new AdtAhdrChunk
            {
                IndexX = i % 4,
                IndexY = i / 4,
                HeaderRaw = BuildHeader(areaId: 3519, holes: 0),
                ShadowRaw = shadow,
                // Layer 0 has no AMAP; the upper two carry short payloads the project cannot decode.
                Layers =
                [
                    new AdtAhdrLayer(0, 0, null),
                    new AdtAhdrLayer(1, 0x100, new byte[300]),
                    new AdtAhdrLayer(2, 0x100, new byte[512]),
                ],
            });
        }

        return BuildTile(chunks);
    }

    private static AdtAhdrTile BuildTile(IReadOnlyList<AdtAhdrChunk> chunks)
    {
        const int outer = 129 * 129, inner = 128 * 128;
        var outerHeights = new float[outer];
        var innerHeights = new float[inner];
        Array.Fill(outerHeights, 3600f); // inches
        Array.Fill(innerHeights, 3600f);

        return new AdtAhdrTile
        {
            SourcePath = "area_24_38.dat",
            MverVersion = 22,
            Version = 22,
            VerticesX = 129,
            VerticesY = 129,
            ChunksX = 16,
            ChunksY = 16,
            OuterHeights = outerHeights,
            InnerHeights = innerHeights,
            TextureNames = ["tileset\\a.blp", "tileset\\b.blp", "tileset\\c.blp"],
            ModelNames = [],
            Chunks = chunks,
        };
    }

    /// <summary>0x40 ACNK header with the measured v22 fields: +0x0C area id, +0x10 holes.</summary>
    private static byte[] BuildHeader(int areaId, ushort holes)
    {
        var header = new byte[0x40];
        BitConverter.GetBytes(areaId).CopyTo(header, 0x0C);
        BitConverter.GetBytes(holes).CopyTo(header, 0x10);
        return header;
    }
}
