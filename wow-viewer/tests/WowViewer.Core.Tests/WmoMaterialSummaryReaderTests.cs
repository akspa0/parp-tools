using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoMaterialSummaryReaderTests
{
    [Fact]
    public void Read_StandardMomtBuffer_ProducesMaterialSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOHD", CreateMohdPayload(materialCount: 2)),
            .. CreateChunk("MOMT", CreateMomtPayload(64,
                CreateMomtEntry(64, flags: 1, shader: 3, blendMode: 0, tex1: 12, tex2: 20, tex3: 44),
                CreateMomtEntry(64, flags: 0, shader: 5, blendMode: 2, tex1: 24, tex2: 0, tex3: 88))),
        ];

        using MemoryStream stream = new(bytes);
        WmoMaterialSummary summary = WmoMaterialSummaryReader.Read(stream, "synthetic_material_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(128, summary.PayloadSizeBytes);
        Assert.Equal(64, summary.EntrySizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(2, summary.DistinctShaderCount);
        Assert.Equal(2, summary.DistinctBlendModeCount);
        Assert.Equal(1, summary.NonZeroFlagCount);
        Assert.Equal(24, summary.MaxTexture1Offset);
        Assert.Equal(20, summary.MaxTexture2Offset);
        Assert.Equal(88, summary.MaxTexture3Offset);
    }

    [Fact]
    public void Read_LegacyMomtBuffer_ProducesMaterialSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(14)),
            .. CreateChunk("MOHD", CreateMohdPayload(materialCount: 1)),
            .. CreateChunk("MOMT", CreateMomtPayload(44,
                CreateMomtEntry(44, flags: 2, shader: 7, blendMode: 1, tex1: 16, tex2: 32, tex3: 48))),
        ];

        using MemoryStream stream = new(bytes);
        WmoMaterialSummary summary = WmoMaterialSummaryReader.Read(stream, "synthetic_legacy_material_root.wmo");

        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(44, summary.PayloadSizeBytes);
        Assert.Equal(44, summary.EntrySizeBytes);
        Assert.Equal(1, summary.EntryCount);
        Assert.Equal(1, summary.DistinctShaderCount);
        Assert.Equal(1, summary.DistinctBlendModeCount);
        Assert.Equal(1, summary.NonZeroFlagCount);
        Assert.Equal(16, summary.MaxTexture1Offset);
        Assert.Equal(32, summary.MaxTexture2Offset);
        Assert.Equal(48, summary.MaxTexture3Offset);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateMohdPayload(uint materialCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        return bytes;
    }

    private static byte[] CreateMomtPayload(int entrySize, params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
        {
            if (entry.Length != entrySize)
                throw new InvalidOperationException($"Expected entry size {entrySize}, but got {entry.Length}.");

            stream.Write(entry, 0, entry.Length);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMomtEntry(int entrySize, uint flags, uint shader, uint blendMode, uint tex1, uint tex2, uint tex3)
    {
        byte[] bytes = new byte[entrySize];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), shader);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), blendMode);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), tex1);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), tex2);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36, 4), tex3);
        return bytes;
    }
}
