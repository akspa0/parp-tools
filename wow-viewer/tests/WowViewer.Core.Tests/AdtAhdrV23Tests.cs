using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 237 follow-up: v22/v23 DAT files (e.g. the Wrath "area_XX_YY" corpus) carry the same
/// AHDR-family vocabulary as v26 but no ALOC chunk, so the tile location must come from the file
/// name. These tests pin that fallback and the version-agnostic read.
/// </summary>
public sealed class AdtAhdrV23Tests
{
    [Theory]
    [InlineData("area_51_31.dat", 51, 31)]
    [InlineData("area_51_32.dat", 51, 32)]
    [InlineData("Kalimdor_12_7.dat", 12, 7)]
    [InlineData("area_0_0.dat", 0, 0)]
    public void TryParseTileLocationFromName_ReadsTrailingXxYy(string name, int expectedX, int expectedY)
    {
        Assert.True(AdtAhdrReader.TryParseTileLocationFromName(name, out int x, out int y));
        Assert.Equal(expectedX, x);
        Assert.Equal(expectedY, y);
    }

    [Theory]
    [InlineData("area.dat")]
    [InlineData("kalimdor.dat")]
    [InlineData("area_51.dat")]
    public void TryParseTileLocationFromName_RejectsNamesWithoutTwoIntegers(string name)
    {
        Assert.False(AdtAhdrReader.TryParseTileLocationFromName(name, out _, out _));
    }

    [Fact]
    public void Read_V23FileWithoutAloc_IsAhdrFamilyAndReadsWithoutThrowing()
    {
        byte[] file = BuildV23Tile();

        Assert.True(AdtAhdrReader.IsAhdrFamily(file));
        // No ALOC: the reader cannot resolve the tile location from the file itself.
        Assert.False(AdtAhdrReader.TryReadTileLocation(file, out _, out _));

        AdtAhdrTile tile = AdtAhdrReader.Read(file, "area_51_31.dat");

        Assert.Equal(23u, tile.MverVersion);
        Assert.Equal(23u, tile.Version);
        Assert.Equal((129, 129, 16, 16), (tile.VerticesX, tile.VerticesY, tile.ChunksX, tile.ChunksY));
        Assert.Equal(256, tile.Chunks.Count);
        Assert.Contains(tile.Diagnostics, d => d.Contains("no ALOC", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Minimal v23-shaped file: MVER 23, AHDR, AVTX, ANRM, ATEX, ACNK×256 (one ALYR each), no ALOC.</summary>
    private static byte[] BuildV23Tile()
    {
        const int outer = 129 * 129, inner = 128 * 128;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        WriteChunk(w, "MVER", b => b.Write(23u));
        WriteChunk(w, "AHDR", b =>
        {
            b.Write(23u); b.Write(129u); b.Write(129u); b.Write(16u); b.Write(16u);
            for (int i = 0; i < 11; i++) b.Write(0u);
        });
        WriteChunk(w, "AVTX", b => { for (int i = 0; i < outer + inner; i++) b.Write(0f); });
        WriteChunk(w, "ANRM", b => { for (int i = 0; i < outer + inner; i++) { b.Write((byte)0); b.Write((byte)127); b.Write((byte)0); } });
        WriteChunk(w, "ATEX", b => b.Write(Encoding.ASCII.GetBytes("tileset\\a.blp\0")));
        for (int c = 0; c < 256; c++)
        {
            int index = c;
            WriteChunk(w, "ACNK", b =>
            {
                b.Write(index % 16); b.Write(index / 16); b.Write(0xD000u);
                for (int i = 0; i < 13; i++) b.Write(0u);
                WriteChunk(b, "ALYR", l =>
                {
                    l.Write(0);            // texture index
                    l.Write(0x100u);       // flags
                    for (int i = 0; i < 7; i++) l.Write(0u);
                    WriteChunk(l, "AMAP", a => { for (int p = 0; p < AdtAhdrAlpha.Pixels; p++) a.Write((byte)255); });
                });
            });
        }

        return ms.ToArray();
    }

    private static void WriteChunk(BinaryWriter w, string id, Action<BinaryWriter> body)
    {
        using var payload = new MemoryStream();
        using (var pw = new BinaryWriter(payload, Encoding.ASCII, leaveOpen: true))
            body(pw);
        w.Write(Encoding.ASCII.GetBytes(new string(id.Reverse().ToArray())));
        w.Write((uint)payload.Length);
        w.Write(payload.ToArray());
    }
}
