using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtPlacementReaderTests
{
    [Fact]
    public void Read_ObjAdtBuffer_ResolvesNamesAndPlacements()
    {
        byte[] mmdx = CreateStringBlock("foo.mdx", "bar.mdx");
        byte[] mwmo = CreateStringBlock("a.wmo", "b.wmo");
        byte[] mmid = CreateUInt32Array(0u, 8u);
        byte[] mwid = CreateUInt32Array(0u, 6u);
        byte[] mddf = CreateMddfEntry(nameId: 1u, uniqueId: 77u, rawX: 100f, rawY: 200f, rawZ: 300f, rotX: 1f, rotY: 2f, rotZ: 3f, scale: 2048);
        byte[] modf = CreateModfEntry(nameId: 0u, uniqueId: 88u, rawX: 110f, rawY: 210f, rawZ: 310f, rotX: 4f, rotY: 5f, rotZ: 6f, bbMinX: 10f, bbMinY: 20f, bbMinZ: 30f, bbMaxX: 40f, bbMaxY: 50f, bbMaxZ: 60f, flags: 0x1234);

        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MMDX", mmdx),
            .. CreateChunk("MMID", mmid),
            .. CreateChunk("MWMO", mwmo),
            .. CreateChunk("MWID", mwid),
            .. CreateChunk("MDDF", mddf),
            .. CreateChunk("MODF", modf),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_4_9_obj0.adt");
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtObj, catalog.Kind);
        Assert.Equal(2, catalog.ModelNames.Count);
        Assert.Equal(2, catalog.WorldModelNames.Count);
        Assert.Single(catalog.ModelPlacements);
        Assert.Single(catalog.WorldModelPlacements);

        AdtModelPlacement modelPlacement = catalog.ModelPlacements[0];
        Assert.Equal("bar.mdx", modelPlacement.ModelPath);
        Assert.Equal(77, modelPlacement.UniqueId);
        Assert.Equal(new Vector3(16866.666f, 16966.666f, 300f), modelPlacement.Position);
        Assert.Equal(new Vector3(1f, 2f, 3f), modelPlacement.Rotation);
        Assert.Equal(2f, modelPlacement.Scale);

        AdtWorldModelPlacement wmoPlacement = catalog.WorldModelPlacements[0];
        Assert.Equal("a.wmo", wmoPlacement.ModelPath);
        Assert.Equal(88, wmoPlacement.UniqueId);
        Assert.Equal(new Vector3(16856.666f, 16956.666f, 310f), wmoPlacement.Position);
        Assert.Equal(new Vector3(4f, 5f, 6f), wmoPlacement.Rotation);
        Assert.Equal(new Vector3(17016.666f, 17026.666f, 30f), wmoPlacement.BoundsMin);
        Assert.Equal(new Vector3(17046.666f, 17056.666f, 60f), wmoPlacement.BoundsMax);
        Assert.Equal((ushort)0x1234, wmoPlacement.Flags);
    }

    [Fact]
    public void Read_DevelopmentObjAdt_ProducesResolvedPlacements()
    {
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(MapTestPaths.DevelopmentObjAdtPath);

        Assert.Equal(MapFileKind.AdtObj, catalog.Kind);
        Assert.Equal(10, catalog.ModelPlacements.Count);
        Assert.Equal(15, catalog.WorldModelPlacements.Count);
        Assert.Contains(catalog.WorldModelPlacements, static placement => !placement.ModelPath.StartsWith("unknown_", StringComparison.Ordinal));
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateUInt32Array(params uint[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(uint)];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint), sizeof(uint)), values[index]);

        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.ASCII.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMddfEntry(uint nameId, uint uniqueId, float rawX, float rawY, float rawZ, float rotX, float rotY, float rotZ, ushort scale)
    {
        byte[] bytes = new byte[36];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), nameId);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), uniqueId);
        WriteSingle(bytes, 8, rawX);
        WriteSingle(bytes, 12, rawZ);
        WriteSingle(bytes, 16, rawY);
        WriteSingle(bytes, 20, rotX);
        WriteSingle(bytes, 24, rotZ);
        WriteSingle(bytes, 28, rotY);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(32, 2), scale);
        return bytes;
    }

    private static byte[] CreateModfEntry(uint nameId, uint uniqueId, float rawX, float rawY, float rawZ, float rotX, float rotY, float rotZ, float bbMinX, float bbMinY, float bbMinZ, float bbMaxX, float bbMaxY, float bbMaxZ, ushort flags)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), nameId);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), uniqueId);
        WriteSingle(bytes, 8, rawX);
        WriteSingle(bytes, 12, rawZ);
        WriteSingle(bytes, 16, rawY);
        WriteSingle(bytes, 20, rotX);
        WriteSingle(bytes, 24, rotZ);
        WriteSingle(bytes, 28, rotY);
        WriteSingle(bytes, 32, bbMinX);
        WriteSingle(bytes, 36, bbMinZ);
        WriteSingle(bytes, 40, bbMinY);
        WriteSingle(bytes, 44, bbMaxX);
        WriteSingle(bytes, 48, bbMaxZ);
        WriteSingle(bytes, 52, bbMaxY);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(56, 2), flags);
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}