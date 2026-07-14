using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Tests;

public sealed class AdtTerrainWriterTests
{
    private const int TileHeightmapSize = 257;
    private const int RootMcnkHeaderSize = 128;
    private const int McinEntrySize = 16;

    [Fact]
    public void ApplyHeightmap_PreservesNonMcvtPayloads()
    {
        byte[] sourceMccv = CreatePatternedMccv();
        byte[] sourceBytes = CreateSyntheticAdt(includeMccv: true, sourceMccv);
        float[] heightmap = CreateHeightmap(12.5f);

        byte[] updated = AdtTerrainWriter.ApplyHeightmap(sourceBytes, "synthetic.adt", heightmap);

        Assert.Equal(sourceBytes.Length, updated.Length);
        Assert.Equal(GetMcinPayload(sourceBytes), GetMcinPayload(updated));
        Assert.Equal(GetMccvPayload(sourceBytes), GetMccvPayload(updated));
        Assert.NotEqual(GetMcvtPayload(sourceBytes), GetMcvtPayload(updated));
        Assert.NotEqual(GetMcnrPayload(sourceBytes), GetMcnrPayload(updated));

        byte[] updatedMccv = GetMccvPayload(updated);
        Assert.Equal(sourceMccv, updatedMccv);
    }

    [Fact]
    public void ApplyHeightmap_UpdatesMcvtPayloadFromModelHeights()
    {
        byte[] sourceBytes = CreateSyntheticAdt(includeMccv: false, mccvPayload: null);
        float[] heightmap = CreateHeightmap(42f);

        byte[] updated = AdtTerrainWriter.ApplyHeightmap(sourceBytes, "synthetic.adt", heightmap);

        byte[] mcvtPayload = GetMcvtPayload(updated);
        Assert.Equal(BitConverter.SingleToInt32Bits(42f), BinaryPrimitives.ReadInt32LittleEndian(mcvtPayload.AsSpan(0, sizeof(float))));

        byte[] mcnrPayload = GetMcnrPayload(updated);
        Assert.Equal(unchecked((byte)(sbyte)0), mcnrPayload[0]);
        Assert.Equal(unchecked((byte)(sbyte)127), mcnrPayload[1]);
        Assert.Equal(unchecked((byte)(sbyte)0), mcnrPayload[2]);
    }

    [Fact]
    public void ApplyHeightmap_MultipleMcinChunks_UpdatesAllResolvedChunks()
    {
        byte[] sourceBytes = CreateSyntheticAdtWithTwoChunks();
        float[] heightmap = CreateHeightmap(37f);

        byte[] updated = AdtTerrainWriter.ApplyHeightmap(sourceBytes, "synthetic-two-chunk.adt", heightmap);
        AdtChunkChangeAudit audit = AdtTerrainPatchAudit.AnalyzeChunkChanges(sourceBytes, updated, "synthetic-two-chunk.adt");

        Assert.Equal(2, audit.PresentChunkCount);
        Assert.Equal(2, audit.ChangedMcvtChunkCount);
        Assert.Equal(2, audit.ChangedMcnrChunkCount);
    }

    private static byte[] CreateSyntheticAdt(bool includeMccv, byte[]? mccvPayload)
    {
        byte[] mcnkPayload = CreateSyntheticMcnk(includeMccv, mccvPayload);
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

    private static byte[] CreateSyntheticMcnk(bool includeMccv, byte[]? mccvPayload)
    {
        return CreateSyntheticMcnk(chunkX: 0, chunkY: 0, includeMccv, mccvPayload);
    }

    private static byte[] CreateSyntheticMcnk(int chunkX, int chunkY, bool includeMccv, byte[]? mccvPayload)
    {
        byte[] header = new byte[RootMcnkHeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), checked((uint)chunkX));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), checked((uint)chunkY));

        List<(FourCC id, byte[] data)> subchunks =
        [
            (AdtChunkIds.Mcvt, new byte[145 * sizeof(float)]),
            (AdtChunkIds.Mcnr, new byte[0x1C0])
        ];

        if (includeMccv)
            subchunks.Add((AdtChunkIds.Mccv, mccvPayload ?? throw new ArgumentNullException(nameof(mccvPayload))));

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

    private static byte[] CreateSyntheticAdtWithTwoChunks()
    {
        byte[] firstMcnkPayload = CreateSyntheticMcnk(chunkX: 0, chunkY: 0, includeMccv: false, mccvPayload: null);
        byte[] secondMcnkPayload = CreateSyntheticMcnk(chunkX: 1, chunkY: 0, includeMccv: false, mccvPayload: null);
        byte[] mver = CreateChunk(MapChunkIds.Mver, CreateUInt32Payload(18));
        byte[] mhdr = CreateChunk(MapChunkIds.Mhdr, new byte[64]);
        byte[] mcin = CreateChunk(MapChunkIds.Mcin, new byte[256 * McinEntrySize]);
        byte[] firstMcnk = CreateChunk(MapChunkIds.Mcnk, firstMcnkPayload);
        byte[] secondMcnk = CreateChunk(MapChunkIds.Mcnk, secondMcnkPayload);

        int mverOffset = 0;
        int mhdrOffset = mverOffset + GetStoredChunkLength(mver);
        int mcinOffset = mhdrOffset + GetStoredChunkLength(mhdr);
        int firstMcnkOffset = mcinOffset + GetStoredChunkLength(mcin);
        int secondMcnkOffset = firstMcnkOffset + GetStoredChunkLength(firstMcnk);

        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(8, 4), checked((uint)firstMcnkOffset));
        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(12, 4), checked((uint)GetStoredChunkLength(firstMcnk)));
        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(8 + McinEntrySize, 4), checked((uint)secondMcnkOffset));
        BinaryPrimitives.WriteUInt32LittleEndian(mcin.AsSpan(12 + McinEntrySize, 4), checked((uint)GetStoredChunkLength(secondMcnk)));

        using MemoryStream stream = new();
        stream.Write(mver, 0, mver.Length);
        WritePadding(stream, mver.Length - 8);
        stream.Write(mhdr, 0, mhdr.Length);
        WritePadding(stream, mhdr.Length - 8);
        stream.Write(mcin, 0, mcin.Length);
        WritePadding(stream, mcin.Length - 8);
        stream.Write(firstMcnk, 0, firstMcnk.Length);
        WritePadding(stream, firstMcnk.Length - 8);
        stream.Write(secondMcnk, 0, secondMcnk.Length);
        WritePadding(stream, secondMcnk.Length - 8);
        return stream.ToArray();
    }

    private static byte[] CreatePatternedMccv()
    {
        byte[] bytes = new byte[145 * 4];
        for (int index = 0; index < 145; index++)
        {
            int offset = index * 4;
            bytes[offset + 0] = (byte)(index + 1);
            bytes[offset + 1] = (byte)(index + 2);
            bytes[offset + 2] = (byte)(index + 3);
            bytes[offset + 3] = 255;
        }

        return bytes;
    }

    private static float[] CreateHeightmap(float value)
    {
        float[] heights = new float[TileHeightmapSize * TileHeightmapSize];
        Array.Fill(heights, value);
        return heights;
    }

    private static byte[] GetMccvPayload(byte[] bytes)
    {
        byte[] mcnkPayload = GetTerrainChunkPayload(bytes);
        int position = RootMcnkHeaderSize;
        while (position <= mcnkPayload.Length - 8)
        {
            FourCC id = FourCC.FromFileBytes(mcnkPayload.AsSpan(position, 4));
            int size = BinaryPrimitives.ReadInt32LittleEndian(mcnkPayload.AsSpan(position + 4, 4));
            if (id == AdtChunkIds.Mccv)
                return mcnkPayload.AsSpan(position + 8, size).ToArray();

            position += 8 + size;
        }

        throw new InvalidOperationException("Synthetic MCNK payload did not contain MCCV.");
    }

    private static byte[] GetMcnrPayload(byte[] bytes)
    {
        return GetSubchunkPayload(bytes, AdtChunkIds.Mcnr);
    }

    private static byte[] GetMcvtPayload(byte[] bytes)
    {
        return GetSubchunkPayload(bytes, AdtChunkIds.Mcvt);
    }

    private static byte[] GetSubchunkPayload(byte[] bytes, FourCC expectedId)
    {
        byte[] mcnkPayload = GetTerrainChunkPayload(bytes);
        int position = RootMcnkHeaderSize;
        while (position <= mcnkPayload.Length - 8)
        {
            FourCC id = FourCC.FromFileBytes(mcnkPayload.AsSpan(position, 4));
            int size = BinaryPrimitives.ReadInt32LittleEndian(mcnkPayload.AsSpan(position + 4, 4));
            if (id == expectedId)
                return mcnkPayload.AsSpan(position + 8, size).ToArray();

            position += 8 + size;
        }

        throw new InvalidOperationException($"Synthetic MCNK payload did not contain {expectedId}.");
    }

    private static byte[] GetMcinPayload(byte[] bytes)
    {
        int position = 0;
        while (position <= bytes.Length - 8)
        {
            FourCC id = FourCC.FromFileBytes(bytes.AsSpan(position, 4));
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4, 4));
            if (id == MapChunkIds.Mcin)
                return bytes.AsSpan(position + 8, size).ToArray();

            position += 8 + size;
            if ((size & 1) != 0)
                position++;
        }

        throw new InvalidOperationException("Synthetic ADT did not contain MCIN.");
    }

    private static byte[] GetTerrainChunkPayload(byte[] bytes)
    {
        int position = 0;
        while (position <= bytes.Length - 8)
        {
            FourCC id = FourCC.FromFileBytes(bytes.AsSpan(position, 4));
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4, 4));
            if (id == MapChunkIds.Mcnk)
                return bytes.AsSpan(position + 8, size).ToArray();

            position += 8 + size;
            if ((size & 1) != 0)
                position++;
        }

        throw new InvalidOperationException("Synthetic ADT did not contain MCNK.");
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