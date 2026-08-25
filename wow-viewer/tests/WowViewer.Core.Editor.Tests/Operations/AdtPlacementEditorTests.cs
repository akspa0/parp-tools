using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Tests.Operations;

public class AdtPlacementEditorTests
{
    private const string SourcePath = "synthetic_4_9_obj0.adt";

    [Fact]
    public void Noop_edit_is_byte_identical()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(source, SourcePath, []);

        Assert.Equal(source, result.Bytes);
    }

    [Fact]
    public void Add_model_merges_name_table_and_allocates_non_colliding_id()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var add = new AdtPlacementAddEdit(
            AdtPlacementKind.Model,
            "newmodel.mdx",
            new Vector3(16000f, 15000f, 300f),
            new Vector3(0f, 1f, 0f),
            Scale: 1.5f);

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(source, SourcePath, [add]);

        AdtPlacementCatalog catalog = Read(result.Bytes);

        Assert.Equal(2, catalog.ModelPlacements.Count);
        Assert.Contains("newmodel.mdx", result.AddedModelNames);
        Assert.Single(result.AllocatedIds);
        Assert.Equal(78, result.AllocatedIds[0]); // existing uniqueId 77 -> next 78

        AdtModelPlacement added = catalog.ModelPlacements[1];
        Assert.Equal("newmodel.mdx", added.ModelPath);
        Assert.Equal(78, added.UniqueId);
        Assert.Equal(new Vector3(16000f, 15000f, 300f), added.Position);
        Assert.Equal(1.5f, added.Scale, 3);
    }

    [Fact]
    public void Add_model_with_existing_name_reuses_name_table_entry()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var add = new AdtPlacementAddEdit(
            AdtPlacementKind.Model,
            "foo.mdx",
            new Vector3(1f, 2f, 3f),
            new Vector3(0f, 0f, 0f),
            Scale: 1f);

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(source, SourcePath, [add]);

        AdtPlacementCatalog catalog = Read(result.Bytes);

        Assert.Empty(result.AddedModelNames);
        Assert.Equal(2, catalog.ModelPlacements.Count);
        Assert.Equal("foo.mdx", catalog.ModelPlacements[1].ModelPath);
        Assert.Equal(2, catalog.ModelNames.Count);
    }

    [Fact]
    public void Delete_model_removes_entry()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var delete = new AdtPlacementDeleteEdit(AdtPlacementKind.Model, 0, 77);

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(source, SourcePath, [delete]);

        AdtPlacementCatalog catalog = Read(result.Bytes);

        Assert.Empty(catalog.ModelPlacements);
    }

    [Fact]
    public void Rotate_and_scale_round_trip()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var rotate = new AdtPlacementRotateEdit(AdtPlacementKind.Model, 0, 77, new Vector3(1f, 2f, 3f));
        var scale = new AdtPlacementScaleEdit(AdtPlacementKind.Model, 0, 77, 2.0f);

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(source, SourcePath, [rotate, scale]);

        AdtPlacementCatalog catalog = Read(result.Bytes);

        Assert.Equal(new Vector3(1f, 2f, 3f), catalog.ModelPlacements[0].Rotation);
        Assert.Equal(2.0f, catalog.ModelPlacements[0].Scale, 3);
    }

    [Fact]
    public void Stale_unique_id_is_refused()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var delete = new AdtPlacementDeleteEdit(AdtPlacementKind.Model, 0, 999);

        Assert.Throws<InvalidDataException>(() => AdtPlacementEditor.Apply(source, SourcePath, [delete]));
    }

    [Fact]
    public void Unaffected_chunks_are_byte_preserved()
    {
        byte[] source = BuildSyntheticAdt(withWmo: false);

        var rotate = new AdtPlacementRotateEdit(AdtPlacementKind.Model, 0, 77, new Vector3(1f, 2f, 3f));

        byte[] result = AdtPlacementEditor.Apply(source, SourcePath, [rotate]).Bytes;

        // MVER chunk (first 8 + 4 bytes) must be identical.
        Assert.Equal(source.AsSpan(0, 12).ToArray(), result.AsSpan(0, 12).ToArray());
    }

    private static AdtPlacementCatalog Read(byte[] bytes)
    {
        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, SourcePath);
        return AdtPlacementReader.Read(stream, summary);
    }

    private static byte[] BuildSyntheticAdt(bool withWmo)
    {
        byte[] mmdx = CreateStringBlock("foo.mdx", "bar.mdx");
        byte[] mwmo = CreateStringBlock("a.wmo");
        byte[] mmid = CreateUInt32Array(0u, 8u);
        byte[] mwid = CreateUInt32Array(0u);
        byte[] mddf = CreateMddfEntry(nameId: 1u, uniqueId: 77u, rawX: 100f, rawY: 200f, rawZ: 300f, rotX: 0f, rotY: 0f, rotZ: 0f, scale: 1024);
        byte[] modf = CreateModfEntry(nameId: 0u, uniqueId: 99u, rawX: 110f, rawY: 210f, rawZ: 310f, rotX: 0f, rotY: 0f, rotZ: 0f, bbMinX: 200f, bbMinY: 300f, bbMinZ: 10f, bbMaxX: 220f, bbMaxY: 320f, bbMaxZ: 20f);

        var parts = new List<byte[]>
        {
            CreateChunk("MVER", CreateUInt32Payload(18)),
            CreateChunk("MMDX", mmdx),
            CreateChunk("MMID", mmid),
            CreateChunk("MDDF", mddf),
        };

        if (withWmo)
        {
            parts.Add(CreateChunk("MWMO", mwmo));
            parts.Add(CreateChunk("MWID", mwid));
            parts.Add(CreateChunk("MODF", modf));
        }

        return [.. parts.SelectMany(p => p)];
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

    private static byte[] CreateModfEntry(uint nameId, uint uniqueId, float rawX, float rawY, float rawZ, float rotX, float rotY, float rotZ, float bbMinX, float bbMinY, float bbMinZ, float bbMaxX, float bbMaxY, float bbMaxZ)
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
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(56, 2), 0);
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}