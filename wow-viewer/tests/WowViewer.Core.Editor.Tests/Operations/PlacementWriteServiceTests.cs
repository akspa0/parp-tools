using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Tests.Operations;

public class PlacementWriteServiceTests
{
    private const float MapOrigin = 17066.666f;

    [Fact]
    public void ApplyMove_round_trips_through_existing_placement_writer()
    {
        byte[] sourceBytes = BuildSyntheticAdt();
        var service = new PlacementWriteService();

        Vector3 oldPosition = new(MapOrigin - 200f, MapOrigin - 100f, 300f);
        Vector3 newPosition = new(16010f, 15020f, 333f);

        var operation = new PlacementMoveOperation(
            "op-1",
            "placement.test",
            "synthetic_4_9_obj0.adt",
            AdtPlacementKind.Model,
            entryIndex: 0,
            uniqueId: 77,
            oldPosition,
            newPosition);

        // The service writes into the source byte array via AdtPlacementWriter.
        byte[] result = service.ApplyMove(sourceBytes, "synthetic_4_9_obj0.adt", operation);

        using MemoryStream stream = new(result);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_4_9_obj0.adt");
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(stream, summary);

        Assert.Equal(newPosition, catalog.ModelPlacements[0].Position);
    }

    [Fact]
    public void CreateReverse_swaps_positions_and_keeps_identity()
    {
        Vector3 oldPosition = new(1f, 2f, 3f);
        Vector3 newPosition = new(4f, 5f, 6f);

        var operation = new PlacementMoveOperation("op-1", "placement.test", "a.adt", AdtPlacementKind.Model, 0, 77, oldPosition, newPosition);

        var reverse = Assert.IsType<PlacementMoveOperation>(operation.CreateReverse());

        Assert.Equal(newPosition, reverse.OldPosition);
        Assert.Equal(oldPosition, reverse.NewPosition);
        Assert.Equal(operation.UniqueId, reverse.UniqueId);
        Assert.Equal(operation.EntryIndex, reverse.EntryIndex);
        Assert.Equal(operation.Kind, reverse.Kind);
        Assert.Single(reverse.AffectedPaths);
        Assert.Equal("a.adt", reverse.AffectedPaths[0]);
    }

    [Fact]
    public void ApplyMove_with_stale_unique_id_throws()
    {
        byte[] sourceBytes = BuildSyntheticAdt();
        var service = new PlacementWriteService();

        var operation = new PlacementMoveOperation(
            "op-1",
            "placement.test",
            "synthetic_4_9_obj0.adt",
            AdtPlacementKind.Model,
            entryIndex: 0,
            uniqueId: 999, // intentionally wrong
            new Vector3(1, 2, 3),
            new Vector3(4, 5, 6));

        Assert.Throws<InvalidDataException>(() => service.ApplyMove(sourceBytes, "synthetic_4_9_obj0.adt", operation));
    }

    private static byte[] BuildSyntheticAdt()
    {
        byte[] mmdx = CreateStringBlock("foo.mdx");
        byte[] mmid = CreateUInt32Array(0u);
        byte[] mddf = CreateMddfEntry(
            nameId: 0u,
            uniqueId: 77u,
            rawX: 100f,
            rawY: 200f,
            rawZ: 300f,
            rotX: 0f,
            rotY: 0f,
            rotZ: 0f,
            scale: 1024);

        return
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MMDX", mmdx),
            .. CreateChunk("MMID", mmid),
            .. CreateChunk("MDDF", mddf),
        ];
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
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

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}