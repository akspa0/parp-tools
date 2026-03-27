using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupNormalSummaryReaderTests
{
    [Fact]
    public void Read_MonrBuffer_ProducesNormalSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MONR", CreateNormalPayload(new Vector3(1f, 0f, 0f), new Vector3(0f, -1f, 0f), new Vector3(0.5f, 0.5f, 0.5f))))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupNormalSummary summary = WmoGroupNormalSummaryReader.Read(stream, "synthetic_monr_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(36, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.NormalCount);
        Assert.Equal(0f, summary.MinX);
        Assert.Equal(1f, summary.MaxX);
        Assert.Equal(-1f, summary.MinY);
        Assert.Equal(0.5f, summary.MaxY);
        Assert.Equal(0f, summary.MinZ);
        Assert.Equal(0.5f, summary.MaxZ);
        Assert.Equal(0.8660254f, summary.MinLength, 4);
        Assert.Equal(1f, summary.MaxLength, 4);
        Assert.Equal(0.95534176f, summary.AverageLength, 4);
        Assert.Equal(2, summary.NearUnitCount);
    }

    private static byte[] CreateMogpPayload(int headerSize, params (string Id, byte[] Payload)[] subchunks)
    {
        byte[] header = new byte[headerSize];
        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static byte[] CreateNormalPayload(params Vector3[] normals)
    {
        byte[] bytes = new byte[normals.Length * 12];
        for (int index = 0; index < normals.Length; index++)
        {
            int offset = index * 12;
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(normals[index].X));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 4, 4), BitConverter.SingleToInt32Bits(normals[index].Y));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 8, 4), BitConverter.SingleToInt32Bits(normals[index].Z));
        }

        return bytes;
    }
}
