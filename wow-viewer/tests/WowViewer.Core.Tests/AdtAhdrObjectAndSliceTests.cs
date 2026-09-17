using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.Tests;

/// <summary>Spec 237: DAT v26 ACDO/ADST decoding and the measured placement, normal and colour frames.</summary>
public sealed class AdtAhdrObjectAndSliceTests
{
    [Fact]
    public void Read_DecodesAcdoFieldsAndAdstRows()
    {
        byte[] file = BuildTile(chunk0Objects: [BuildAcdo(modelIndex: 1, local: new Vector3(150f, 36f, -300f), rotation: new Vector3(1f, 90f, 2f), scale: 1.5f, uniqueId: 77, trailing: [2])]);

        AdtAhdrTile tile = AdtAhdrReader.Read(file, "synthetic");

        Assert.Empty(tile.Diagnostics);
        AdtAhdrObjectDefinition obj = Assert.Single(tile.Chunks[0].Objects);
        Assert.Equal(1, obj.ModelIndex);
        Assert.Equal(new Vector3(150f, 36f, -300f), obj.LocalPositionInches);
        Assert.Equal(new Vector3(1f, 90f, 2f), obj.RotationDegrees);
        Assert.Equal(1.5f, obj.Scale);
        Assert.Equal(77u, obj.UniqueId);
        Assert.Equal([2u], obj.TrailingValues);

        AdtAhdrModelFileReference reference = Assert.Single(tile.ModelFileReferences);
        Assert.Equal((63420377u, 190719u, 1u), (reference.UniqueId, reference.FileDataId, reference.Field8));
    }

    [Fact]
    public void ResolveObjectGridPosition_IsChunkCentreRelativeInInchesAndAboveChunkMean()
    {
        byte[] file = BuildTile(heightInches: 720f, chunk0Objects: [BuildAcdo(0, new Vector3(150f, 36f, -300f), Vector3.Zero, 1f, 1, [])]);
        AdtAhdrTile tile = AdtAhdrReader.Read(file, "synthetic");
        AdtAhdrChunk chunk = tile.Chunks[0];

        (float column, float row, float height) = AdtAhdrTileSlicer.ResolveObjectGridPosition(tile, chunk, chunk.Objects[0]);

        Assert.Equal(4f + 1f, column);   // chunk centre (4 cells) + 150 in = 1 cell along the column axis
        Assert.Equal(4f - 2f, row);      // -300 in = -2 cells along the row axis
        Assert.Equal(720f + 36f, height); // chunk mean + vertical offset
    }

    [Fact]
    public void SliceStoredNormals_MapsColumnVerticalRowToRendererFrame()
    {
        byte[] file = BuildTile(normal: (column: 127, vertical: 0, row: 0));
        AdtAhdrTile tile = AdtAhdrReader.Read(file, "synthetic");

        Vector3[]? normals = AdtAhdrTileSlicer.SliceStoredNormals(tile, 0, 0);

        Assert.NotNull(normals);
        // Renderer Y decreases along the column axis, so +column becomes -Y.
        Assert.All(normals!, n => Assert.Equal(new Vector3(0f, -1f, 0f), n));
    }

    [Fact]
    public void SliceVertexColors_PassesBytesThroughInInterleavedOrder()
    {
        byte[] file = BuildTile();
        AdtAhdrTile tile = AdtAhdrReader.Read(file, "synthetic");

        byte[]? colors = AdtAhdrTileSlicer.SliceVertexColors(tile, 0, 0);

        Assert.NotNull(colors);
        Assert.Equal(145 * 4, colors!.Length);
        Assert.Equal(new byte[] { 127, 127, 127, 255 }, colors[..4]);
    }

    private static byte[] BuildAcdo(int modelIndex, Vector3 local, Vector3 rotation, float scale, uint uniqueId, uint[] trailing)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(modelIndex);
        WriteVector(w, local);
        WriteVector(w, rotation);
        w.Write(scale);
        w.Write(1f);
        w.Write(0u);
        w.Write(0f);
        w.Write(uniqueId);
        w.Write((uint)trailing.Length);
        w.Write(0u);
        foreach (uint value in trailing)
            w.Write(value);
        return ms.ToArray();
    }

    private static byte[] BuildTile(float heightInches = 0f, (sbyte column, sbyte vertical, sbyte row)? normal = null, byte[][]? chunk0Objects = null)
    {
        const int outer = 129 * 129, inner = 128 * 128;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        WriteChunk(w, "MVER", b => b.Write(26u));
        WriteChunk(w, "AHDR", b =>
        {
            b.Write(26u); b.Write(129u); b.Write(129u); b.Write(16u); b.Write(16u);
            for (int i = 0; i < 11; i++) b.Write(0u);
        });
        WriteChunk(w, "ALOC", b => { b.Write(2869u); b.Write(31u); b.Write(27u); b.Write(31u); b.Write(27u); });
        WriteChunk(w, "AVTX", b => { for (int i = 0; i < outer + inner; i++) b.Write(heightInches); });
        (sbyte column, sbyte vertical, sbyte row) n = normal ?? (0, 127, 0);
        WriteChunk(w, "ANRM", b => { for (int i = 0; i < outer + inner; i++) { b.Write(n.column); b.Write(n.vertical); b.Write(n.row); } });
        WriteChunk(w, "ADOO", b => b.Write(Encoding.ASCII.GetBytes("World\\a.m2\0")));
        WriteChunk(w, "ADOO", b => b.Write(Encoding.ASCII.GetBytes("World\\b.wmo\0")));
        for (int c = 0; c < 256; c++)
        {
            int index = c;
            WriteChunk(w, "ACNK", b =>
            {
                b.Write(index % 16); b.Write(index / 16); b.Write(0xD000u);
                for (int i = 0; i < 13; i++) b.Write(0u);
                if (index == 0 && chunk0Objects is not null)
                {
                    foreach (byte[] record in chunk0Objects)
                        WriteChunk(b, "ACDO", r => r.Write(record));
                }
            });
        }

        WriteChunk(w, "ADST", b => { b.Write(63420377u); b.Write(190719u); b.Write(1u); });
        WriteChunk(w, "ACVT", b => { for (int i = 0; i < outer + inner; i++) { b.Write((byte)127); b.Write((byte)127); b.Write((byte)127); b.Write((byte)255); } });
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

    private static void WriteVector(BinaryWriter w, Vector3 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
        w.Write(v.Z);
    }
}
