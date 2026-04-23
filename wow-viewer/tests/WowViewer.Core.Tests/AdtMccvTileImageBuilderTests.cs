using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMccvTileImageBuilderTests
{
    private const int RootMcnkHeaderSize = 128;
    private const int McinEntrySize = 16;

    [Fact]
    public void ReadChunkColors_ResolvesChunkIndexFromMcinPayload()
    {
        byte[] expectedMccv = CreateChunkColors(10, 20, 30, 255);
        byte[] sourceBytes = CreateSyntheticAdt(chunkX: 1, chunkY: 0, expectedMccv);

        IReadOnlyDictionary<int, byte[]> chunkColors = AdtMccvTileImageBuilder.ReadChunkColors(sourceBytes, "synthetic.adt");

        Assert.True(chunkColors.TryGetValue(1, out byte[]? actualMccv));
        Assert.Equal(expectedMccv, actualMccv);
    }

    [Fact]
    public void RenderTileImageRgba_PreservesRawStoredChannelOrder()
    {
        byte[] chunkColors = CreateChunkColors(1, 2, 3, 255);

        byte[] image = AdtMccvTileImageBuilder.RenderTileImageRgba(new Dictionary<int, byte[]>
        {
            [0] = chunkColors,
        });

        Assert.Equal((byte)1, image[0]);
        Assert.Equal((byte)2, image[1]);
        Assert.Equal((byte)3, image[2]);
        Assert.Equal((byte)255, image[3]);
    }

    [Fact]
    public void RenderTileImageRgba_MissingChunksUseNeutralColor()
    {
        byte[] image = AdtMccvTileImageBuilder.RenderTileImageRgba(new Dictionary<int, byte[]>());

        Assert.Equal((byte)127, image[0]);
        Assert.Equal((byte)127, image[1]);
        Assert.Equal((byte)127, image[2]);
        Assert.Equal((byte)127, image[3]);
    }

    private static byte[] CreateSyntheticAdt(int chunkX, int chunkY, byte[] mccvPayload)
    {
        byte[] mcnkPayload = CreateSyntheticMcnk(chunkX, chunkY, mccvPayload);
        byte[] mver = CreateChunk(MapChunkIds.Mver, CreateUInt32Payload(18));
        byte[] mhdr = CreateChunk(MapChunkIds.Mhdr, new byte[64]);
        byte[] mcin = CreateChunk(MapChunkIds.Mcin, new byte[256 * McinEntrySize]);
        byte[] mcnk = CreateChunk(MapChunkIds.Mcnk, mcnkPayload);

        int mverOffset = 0;
        int mhdrOffset = mverOffset + GetStoredChunkLength(mver);
        int mcinOffset = mhdrOffset + GetStoredChunkLength(mhdr);
        int mcnkOffset = mcinOffset + GetStoredChunkLength(mcin);

        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(8, 4), checked((uint)mcnkOffset));
        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(12, 4), checked((uint)GetStoredChunkLength(mcnk)));

        using MemoryStream stream = new();
        stream.Write(mver, 0, mver.Length);
        WritePadding(stream, mver.Length - 8);
        stream.Write(mhdr, 0, mhdr.Length);
        WritePadding(stream, mhdr.Length - 8);
        stream.Write(mcin, 0, mcin.Length);
        WritePadding(stream, mcin.Length - 8);
        stream.Write(mcnk, 0, mcnk.Length);
        WritePadding(stream, mcnk.Length - 8);
        return stream.ToArray();
    }

    private static byte[] CreateSyntheticMcnk(int chunkX, int chunkY, byte[] mccvPayload)
    {
        byte[] header = new byte[RootMcnkHeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), checked((uint)chunkX));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), checked((uint)chunkY));

        List<(FourCC id, byte[] data)> subchunks =
        [
            (AdtChunkIds.Mcvt, new byte[145 * sizeof(float)]),
            (AdtChunkIds.Mcnr, new byte[0x1C0]),
            (AdtChunkIds.Mccv, mccvPayload)
        ];

        int offset = RootMcnkHeaderSize;
        foreach ((FourCC id, byte[] data) in subchunks)
        {
            if (id == AdtChunkIds.Mcvt)
                BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x14, 4), checked((uint)offset));
            else if (id == AdtChunkIds.Mcnr)
                BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x18, 4), checked((uint)offset));

            offset += 8 + data.Length;
        }

        using MemoryStream stream = new();
        byte[] sizeBytes = new byte[sizeof(uint)];
        stream.Write(header, 0, header.Length);
        foreach ((FourCC id, byte[] data) in subchunks)
        {
            stream.Write(id.ToFileBytes(), 0, 4);
            BinaryPrimitives.WriteUInt32LittleEndian(sizeBytes, checked((uint)data.Length));
            stream.Write(sizeBytes);
            stream.Write(data, 0, data.Length);
        }

        return stream.ToArray();
    }

    private static byte[] CreateChunkColors(byte blue, byte green, byte red, byte alpha)
    {
        byte[] bytes = new byte[145 * 4];
        for (int index = 0; index < 145; index++)
        {
            int offset = index * 4;
            bytes[offset + 0] = blue;
            bytes[offset + 1] = green;
            bytes[offset + 2] = red;
            bytes[offset + 3] = alpha;
        }

        return bytes;
    }

    private static byte[] CreateChunk(FourCC id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(id.ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), checked((uint)payload.Length));
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static int GetStoredChunkLength(byte[] chunkBytes)
    {
        int payloadLength = chunkBytes.Length - 8;
        return chunkBytes.Length + ((payloadLength & 1) != 0 ? 1 : 0);
    }

    private static void WritePadding(Stream stream, int payloadLength)
    {
        if ((payloadLength & 1) != 0)
            stream.WriteByte(0);
    }
}