using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTextureChunkReaderTests
{
    [Fact]
    public void Read_MalformedSubchunkSize_DoesNotThrowAndReturnsNoLayers()
    {
        byte[] payload = CreateRootTextureMcnkPayload(
            indexX: 0,
            indexY: 0,
            flags: 0u,
            mclyPayload: CreateMclyPayload([0u], [0u]),
            mcalPayload: new byte[0]);

        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(128 + 4, 4), uint.MaxValue);

        AdtTextureChunk chunk = AdtTextureChunkReader.Read(0, payload, MapFileKind.Adt, ["base.blp"]);

        Assert.Empty(chunk.Layers);
        Assert.Equal(0, chunk.AlphaPayloadBytes);
    }

    private static byte[] CreateRootTextureMcnkPayload(uint indexX, uint indexY, uint flags, byte[] mclyPayload, byte[] mcalPayload)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x28, 4), (uint)(8 + mcalPayload.Length));

        using MemoryStream stream = new();
        stream.Write(header);
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCLY", mclyPayload));
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCAL", mcalPayload));
        return stream.ToArray();
    }

    private static byte[] CreateMclyPayload(uint[] layerFlags, uint[] layerOffsets)
    {
        byte[] payload = new byte[layerFlags.Length * 16];
        for (int index = 0; index < layerFlags.Length; index++)
        {
            int offset = index * 16;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 0, 4), (uint)index);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 4, 4), layerFlags[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 8, 4), layerOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 12, 4), 0u);
        }

        return payload;
    }
}
