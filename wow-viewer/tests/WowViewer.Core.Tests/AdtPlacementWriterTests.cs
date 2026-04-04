using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtPlacementWriterTests
{
    [Fact]
    public void ApplyTransaction_SyntheticObjAdt_UpdatesPositionsAndTranslatedBounds()
    {
        byte[] mmdx = CreateStringBlock("foo.mdx", "bar.mdx");
        byte[] mwmo = CreateStringBlock("a.wmo", "b.wmo");
        byte[] mmid = CreateUInt32Array(0u, 8u);
        byte[] mwid = CreateUInt32Array(0u, 6u);
        byte[] mddf = CreateMddfEntry(nameId: 1u, uniqueId: 77u, rawX: 100f, rawY: 200f, rawZ: 300f, rotX: 1f, rotY: 2f, rotZ: 3f, scale: 2048);
        byte[] modf = CreateModfEntry(nameId: 0u, uniqueId: 88u, rawX: 110f, rawY: 210f, rawZ: 310f, rotX: 4f, rotY: 5f, rotZ: 6f, bbMinX: 10f, bbMinY: 20f, bbMinZ: 30f, bbMaxX: 40f, bbMaxY: 50f, bbMaxZ: 60f, flags: 0x1234);

        byte[] sourceBytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MMDX", mmdx),
            .. CreateChunk("MMID", mmid),
            .. CreateChunk("MWMO", mwmo),
            .. CreateChunk("MWID", mwid),
            .. CreateChunk("MDDF", mddf),
            .. CreateChunk("MODF", modf),
        ];

        Vector3 newModelPosition = new(16010f, 15020f, 333f);
        Vector3 wmoDelta = new(10f, -5f, 2f);
        Vector3 oldWmoPosition = new(16856.666f, 16956.666f, 310f);
        Vector3 newWmoPosition = oldWmoPosition + wmoDelta;

        var transaction = new AdtPlacementEditTransaction(
            "synthetic_4_9_obj0.adt",
            [
                new AdtPlacementMove(new AdtPlacementReference(AdtPlacementKind.Model, 0, 77), newModelPosition, "synthetic model move"),
                new AdtPlacementMove(new AdtPlacementReference(AdtPlacementKind.WorldModel, 0, 88), newWmoPosition, "synthetic wmo move"),
            ]);

        byte[] updatedBytes = AdtPlacementWriter.ApplyTransaction(sourceBytes, transaction.SourcePath, transaction);

        using MemoryStream stream = new(updatedBytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, transaction.SourcePath);
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(stream, fileSummary);

        Assert.Equal(newModelPosition, catalog.ModelPlacements[0].Position);
        Assert.Equal(newWmoPosition, catalog.WorldModelPlacements[0].Position);
        Assert.Equal(new Vector3(17016.666f, 17026.666f, 30f) + wmoDelta, catalog.WorldModelPlacements[0].BoundsMin);
        Assert.Equal(new Vector3(17046.666f, 17056.666f, 60f) + wmoDelta, catalog.WorldModelPlacements[0].BoundsMax);
    }

    [Fact]
    public void ApplyTransaction_DevelopmentObjAdt_RoundTripsMovedPlacementsThroughSharedReader()
    {
        AdtPlacementCatalog originalCatalog = AdtPlacementReader.Read(MapTestPaths.DevelopmentObjAdtPath);
        AdtModelPlacement originalModel = originalCatalog.ModelPlacements[0];
        AdtWorldModelPlacement originalWorldModel = originalCatalog.WorldModelPlacements[0];

        Vector3 modelDelta = new(1.5f, -2.25f, 3.75f);
        Vector3 wmoDelta = new(-4f, 2f, 1f);

        var transaction = new AdtPlacementEditTransaction(
            MapTestPaths.DevelopmentObjAdtPath,
            [
                new AdtPlacementMove(new AdtPlacementReference(AdtPlacementKind.Model, 0, originalModel.UniqueId), originalModel.Position + modelDelta, "real-data model move"),
                new AdtPlacementMove(new AdtPlacementReference(AdtPlacementKind.WorldModel, 0, originalWorldModel.UniqueId), originalWorldModel.Position + wmoDelta, "real-data wmo move"),
            ]);

        string outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}_development_0_0_obj0.adt");

        try
        {
            AdtPlacementWriter.Write(MapTestPaths.DevelopmentObjAdtPath, outputPath, transaction);

            AdtPlacementCatalog updatedCatalog = AdtPlacementReader.Read(outputPath);

            Assert.Equal(originalCatalog.ModelPlacements.Count, updatedCatalog.ModelPlacements.Count);
            Assert.Equal(originalCatalog.WorldModelPlacements.Count, updatedCatalog.WorldModelPlacements.Count);
            Assert.Equal(originalCatalog.ModelPlacements[0].Position + modelDelta, updatedCatalog.ModelPlacements[0].Position);
            Assert.Equal(originalCatalog.WorldModelPlacements[0].Position + wmoDelta, updatedCatalog.WorldModelPlacements[0].Position);
            Assert.Equal(originalCatalog.WorldModelPlacements[0].BoundsMin + wmoDelta, updatedCatalog.WorldModelPlacements[0].BoundsMin);
            Assert.Equal(originalCatalog.WorldModelPlacements[0].BoundsMax + wmoDelta, updatedCatalog.WorldModelPlacements[0].BoundsMax);
            Assert.Equal(originalCatalog.ModelPlacements[1].Position, updatedCatalog.ModelPlacements[1].Position);
            Assert.Equal(originalCatalog.WorldModelPlacements[1].Position, updatedCatalog.WorldModelPlacements[1].Position);
        }
        finally
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);
        }
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