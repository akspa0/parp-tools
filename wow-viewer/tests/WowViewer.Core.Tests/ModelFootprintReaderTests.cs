using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Models;

namespace WowViewer.Core.Tests;

public sealed class ModelFootprintReaderTests
{
    [Fact]
    public void TryRead_SyntheticM2_ReturnsSingleXzHull()
    {
        byte[] bytes = CreateM2GeometryBytes(
            "SyntheticFootprint",
            [
                new Vector3(0f, 2f, 0f),
                new Vector3(2f, 3f, 0f),
                new Vector3(2f, 4f, 3f),
                new Vector3(0f, 5f, 3f),
            ]);

        Vector2[][]? polygons = ModelFootprintReader.TryRead(bytes, "Creature\\SyntheticFootprint\\SyntheticFootprint.m2");

        Assert.NotNull(polygons);
        Assert.Single(polygons!);
        Assert.Equal(4, polygons[0].Length);
        Assert.Contains(new Vector2(0f, 0f), polygons[0]);
        Assert.Contains(new Vector2(2f, 0f), polygons[0]);
        Assert.Contains(new Vector2(2f, 3f), polygons[0]);
        Assert.Contains(new Vector2(0f, 3f), polygons[0]);
    }

    [Fact]
    public void TryRead_SyntheticMdx_ReturnsPerGeosetHulls()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticFootprintMdx",
            boundsMin: new Vector3(-1f, -1f, -1f),
            boundsMax: new Vector3(4f, 4f, 4f),
            extraChunks:
            [
                CreateChunk("GEOS", CreateClassicGeosPayload(
                [
                    [new Vector3(0f, 1f, 0f), new Vector3(3f, 1f, 0f), new Vector3(0f, 1f, 2f)],
                    [new Vector3(5f, 2f, 5f), new Vector3(6f, 2f, 5f), new Vector3(5f, 2f, 6f)],
                ])),
            ]);

        Vector2[][]? polygons = ModelFootprintReader.TryRead(bytes, "World\\Generic\\SyntheticFootprintMdx.mdx");

        Assert.NotNull(polygons);
        Assert.Equal(2, polygons!.Length);
        Assert.Equal(3, polygons[0].Length);
        Assert.Equal(3, polygons[1].Length);
        Assert.Contains(new Vector2(0f, 0f), polygons[0]);
        Assert.Contains(new Vector2(3f, 0f), polygons[0]);
        Assert.Contains(new Vector2(0f, 2f), polygons[0]);
        Assert.Contains(new Vector2(5f, 5f), polygons[1]);
        Assert.Contains(new Vector2(6f, 5f), polygons[1]);
        Assert.Contains(new Vector2(5f, 6f), polygons[1]);
    }

    private static byte[] CreateM2GeometryBytes(string modelName, IReadOnlyList<Vector3> vertices)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x110;
        int vertexOffset = Align(nameOffset + nameBytes.Length, 0x10);
        int dataLength = vertexOffset + (vertices.Count * 0x30);
        byte[] data = new byte[dataLength];

        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x108u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x3C, 4), (uint)vertices.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x40, 4), (uint)vertexOffset);
        WriteVector3(data, 0xA0, new Vector3(-1f, -1f, -1f));
        WriteVector3(data, 0xAC, new Vector3(4f, 6f, 4f));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0xB8, 4), BitConverter.SingleToInt32Bits(6f));
        nameBytes.CopyTo(data, nameOffset);

        for (int index = 0; index < vertices.Count; index++)
        {
            int offset = vertexOffset + (index * 0x30);
            WriteVector3(data, offset + 0x00, vertices[index]);
            data[offset + 0x0C] = 255;
            WriteVector3(data, offset + 0x14, Vector3.UnitY);
        }

        return data;
    }

    private static byte[] CreateMdxBytes(uint version, string modelName, Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<byte[]> extraChunks)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, boundsMin, boundsMax)));
        foreach (byte[] chunk in extraChunks)
            bytes.AddRange(chunk);

        return [.. bytes];
    }

    private static byte[] CreateClassicGeosPayload(IReadOnlyList<IReadOnlyList<Vector3>> geosets)
    {
        List<byte> payload = [];
        payload.AddRange(CreateInt32Payload(geosets.Count));

        foreach (IReadOnlyList<Vector3> vertices in geosets)
        {
            int safeVertexCount = Math.Max(vertices.Count, 3);
            List<byte> geosetPayload = [];
            WriteTagAndCount(geosetPayload, "VRTX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
            {
                Vector3 vertex = vertices[Math.Min(index, vertices.Count - 1)];
                geosetPayload.AddRange(CreateSinglePayload(vertex.X));
                geosetPayload.AddRange(CreateSinglePayload(vertex.Y));
                geosetPayload.AddRange(CreateSinglePayload(vertex.Z));
            }

            WriteTagAndCount(geosetPayload, "NRMS", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
            {
                geosetPayload.AddRange(CreateSinglePayload(0f));
                geosetPayload.AddRange(CreateSinglePayload(1f));
                geosetPayload.AddRange(CreateSinglePayload(0f));
            }

            WriteTagAndCount(geosetPayload, "UVAS", 1);
            for (int index = 0; index < safeVertexCount; index++)
            {
                geosetPayload.AddRange(CreateSinglePayload(0f));
                geosetPayload.AddRange(CreateSinglePayload(0f));
            }

            WriteTagAndCount(geosetPayload, "PTYP", 1);
            geosetPayload.Add(4);

            WriteTagAndCount(geosetPayload, "PCNT", 1);
            geosetPayload.AddRange(CreateInt32Payload(safeVertexCount));

            WriteTagAndCount(geosetPayload, "PVTX", safeVertexCount);
            for (ushort index = 0; index < safeVertexCount; index++)
            {
                byte[] indexBytes = new byte[2];
                BinaryPrimitives.WriteUInt16LittleEndian(indexBytes, index);
                geosetPayload.AddRange(indexBytes);
            }

            WriteTagAndCount(geosetPayload, "GNDX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.Add((byte)index);

            WriteTagAndCount(geosetPayload, "MTGC", 1);
            geosetPayload.AddRange(CreateInt32Payload(safeVertexCount));

            WriteTagAndCount(geosetPayload, "MATS", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateInt32Payload(index));

            WriteTagAndCount(geosetPayload, "BIDX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateUInt32Payload((uint)index));

            WriteTagAndCount(geosetPayload, "BWGT", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateUInt32Payload(255u));

            geosetPayload.AddRange(CreateInt32Payload(0));
            geosetPayload.AddRange(CreateInt32Payload(0));
            geosetPayload.AddRange(CreateInt32Payload(0));
            geosetPayload.AddRange(CreateSinglePayload(5f));
            geosetPayload.AddRange(CreateSinglePayload(-1f));
            geosetPayload.AddRange(CreateSinglePayload(-1f));
            geosetPayload.AddRange(CreateSinglePayload(-1f));
            geosetPayload.AddRange(CreateSinglePayload(1f));
            geosetPayload.AddRange(CreateSinglePayload(1f));
            geosetPayload.AddRange(CreateSinglePayload(1f));
            geosetPayload.AddRange(CreateInt32Payload(0));

            payload.AddRange(CreateSizedPayload(geosetPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateModlPayload(string modelName, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] payload = new byte[0x6C];
        WriteFixedAscii(payload, 0, 0x50, modelName);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x50, 4), boundsMin.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x54, 4), boundsMin.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x58, 4), boundsMin.Z);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x5C, 4), boundsMax.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x60, 4), boundsMax.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x64, 4), boundsMax.Z);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x68, 4), 5f);
        return payload;
    }

    private static byte[] CreateChunk(string tag, byte[] payload)
    {
        List<byte> chunk = [];
        chunk.AddRange(Encoding.ASCII.GetBytes(tag));
        chunk.AddRange(CreateUInt32Payload((uint)payload.Length));
        chunk.AddRange(payload);
        return [.. chunk];
    }

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        List<byte> sized = [];
        sized.AddRange(CreateUInt32Payload((uint)(payload.Count + 4)));
        sized.AddRange(payload);
        return [.. sized];
    }

    private static void WriteTagAndCount(List<byte> payload, string tag, int count)
    {
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateInt32Payload(count));
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateInt32Payload(int value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return bytes;
    }

    private static void WriteFixedAscii(byte[] destination, int offset, int length, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        Array.Copy(bytes, 0, destination, offset, Math.Min(bytes.Length, length - 1));
        destination[offset + Math.Min(bytes.Length, length - 1)] = 0;
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x00, 4), BitConverter.SingleToInt32Bits(value.X));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x04, 4), BitConverter.SingleToInt32Bits(value.Y));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x08, 4), BitConverter.SingleToInt32Bits(value.Z));
    }

    private static int Align(int value, int alignment)
    {
        int remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }
}