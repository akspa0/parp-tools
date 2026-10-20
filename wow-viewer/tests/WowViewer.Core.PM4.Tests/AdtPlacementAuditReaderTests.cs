using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.PM4.Tests;

public sealed class AdtPlacementAuditReaderTests
{
    [Fact]
    public void ValidCapturedBytesRetainNamedScaledPlacements()
    {
        var catalog = new AdtPlacementCatalog("development_0_0_obj0.adt", MapFileKind.AdtObj,
            ["world/test.m2"], [], [new(0, "world/test.m2", 1, new Vector3(1, 2, 3), Vector3.Zero, 2)], []);
        byte[] bytes = SplitAdtObjWriter.Build(catalog);
        var read = AdtPlacementAuditReader.Read(bytes, catalog.SourcePath);
        Assert.Equal(2, Assert.Single(read.ModelPlacements).Scale);
        Assert.Equal("world/test.m2", read.ModelPlacements[0].ModelPath);
    }

    [Fact]
    public void PartialPlacementRecordCannotBeCalledComplete()
    {
        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true))
        {
            Chunk(bw, "MVER", BitConverter.GetBytes(18u));
            Chunk(bw, "MDDF", new byte[35]);
        }
        Assert.Throws<InvalidDataException>(() => AdtPlacementAuditReader.Read(ms.ToArray(), "development_0_0_obj0.adt"));
    }

    [Fact]
    public void UnresolvedNamesCannotBecomeAnAuthoredModelFamily()
    {
        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true))
        {
            Chunk(bw, "MVER", BitConverter.GetBytes(18u));
            Chunk(bw, "MMDX", Encoding.ASCII.GetBytes("world/test.m2\0"));
            Chunk(bw, "MMID", BitConverter.GetBytes(uint.MaxValue));
            byte[] row = new byte[36];
            BinaryPrimitives.WriteUInt16LittleEndian(row.AsSpan(32), 1024);
            Chunk(bw, "MDDF", row);
        }
        Assert.Throws<InvalidDataException>(() => AdtPlacementAuditReader.Read(ms.ToArray(), "development_0_0_obj0.adt"));
    }

    private static void Chunk(BinaryWriter writer, string id, byte[] payload)
    {
        writer.Write(Encoding.ASCII.GetBytes(new string(id.Reverse().ToArray())));
        writer.Write((uint)payload.Length);
        writer.Write(payload);
    }
}
