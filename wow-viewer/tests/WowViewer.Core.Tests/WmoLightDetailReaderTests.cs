using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoLightDetailReaderTests
{
    [Fact]
    public void Read_MoltV17Buffer_ProducesPerEntryDetails()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohdWithLightCount(2)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOLT", CreateLights(
                CreateStandardLight(1, true, 0x0101, new Vector3(1f, 2f, 3f), 4f, Quaternion.Identity, 2f, 10f),
                CreateStandardLight(2, false, 0x0000, new Vector3(-4f, 5f, -6f), 8f, new Quaternion(0f, 0f, -1f, -0.5f), 3f, 20f))),
        ];

        using MemoryStream stream = new(bytes);
        IReadOnlyList<WmoLightDetail> details = WmoLightDetailReader.Read(stream, "synthetic_lights_root.wmo");

        Assert.Equal(2, details.Count);

        WmoLightDetail first = details[0];
        Assert.Equal(0, first.LightIndex);
        Assert.Equal(0, first.PayloadOffset);
        Assert.Equal(48, first.EntrySizeBytes);
        Assert.Equal((byte)1, first.LightType);
        Assert.True(first.UsesAttenuation);
        Assert.Equal(0xFF00FF00u, first.ColorBgra);
        Assert.Equal(new Vector3(1f, 2f, 3f), first.Position);
        Assert.Equal(4f, first.Intensity);
        Assert.Equal(2f, first.AttenStart);
        Assert.Equal(10f, first.AttenEnd);
        Assert.Equal((ushort)0x0101, first.HeaderFlagsWord);
        Assert.True(first.Rotation.HasValue);
        Assert.Equal(Quaternion.Identity, first.Rotation!.Value);
        Assert.True(first.RotationLength.HasValue);
        Assert.Equal(1f, first.RotationLength!.Value);

        WmoLightDetail second = details[1];
        Assert.Equal(1, second.LightIndex);
        Assert.Equal(48, second.PayloadOffset);
        Assert.Equal(48, second.EntrySizeBytes);
        Assert.Equal((byte)2, second.LightType);
        Assert.False(second.UsesAttenuation);
        Assert.Equal(0xFF00FF00u, second.ColorBgra);
        Assert.Equal(new Vector3(-4f, 5f, -6f), second.Position);
        Assert.Equal(8f, second.Intensity);
        Assert.Equal(3f, second.AttenStart);
        Assert.Equal(20f, second.AttenEnd);
        Assert.Equal((ushort)0x0000, second.HeaderFlagsWord);
        Assert.True(second.Rotation.HasValue);
        Assert.Equal(-1f, second.Rotation!.Value.Z);
        Assert.Equal(-0.5f, second.Rotation.Value.W);
        Assert.True(second.RotationLength.HasValue);
        Assert.Equal(1.118f, second.RotationLength!.Value, 3);
    }

    [Fact]
    public void Read_MoltV14Buffer_ProducesLegacyPerEntryDetails()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohdWithLightCount(2))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOLT", CreateLights(
                CreateLegacyLight(3, true, new Vector3(10f, 20f, 30f), 6f, 1f, 12f),
                CreateLegacyLight(4, false, new Vector3(-7f, 8f, -9f), 9f, 4f, 18f))),
        ];

        using MemoryStream stream = new(bytes);
        IReadOnlyList<WmoLightDetail> details = WmoLightDetailReader.Read(stream, "synthetic_alpha_lights_root.wmo");

        Assert.Equal(2, details.Count);

        WmoLightDetail first = details[0];
        Assert.Equal(32, first.EntrySizeBytes);
        Assert.Equal((byte)3, first.LightType);
        Assert.True(first.UsesAttenuation);
        Assert.Equal(0xFF00FF00u, first.ColorBgra);
        Assert.Equal(new Vector3(10f, 20f, 30f), first.Position);
        Assert.Equal(6f, first.Intensity);
        Assert.Equal(1f, first.AttenStart);
        Assert.Equal(12f, first.AttenEnd);
        Assert.Null(first.HeaderFlagsWord);
        Assert.Null(first.Rotation);
        Assert.Null(first.RotationLength);

        WmoLightDetail second = details[1];
        Assert.Equal(32, second.EntrySizeBytes);
        Assert.Equal(32, second.PayloadOffset);
        Assert.Equal((byte)4, second.LightType);
        Assert.False(second.UsesAttenuation);
        Assert.Equal(new Vector3(-7f, 8f, -9f), second.Position);
        Assert.Equal(9f, second.Intensity);
        Assert.Equal(4f, second.AttenStart);
        Assert.Equal(18f, second.AttenEnd);
        Assert.Null(second.HeaderFlagsWord);
        Assert.Null(second.Rotation);
        Assert.Null(second.RotationLength);
    }

    private static byte[] CreateMohdWithLightCount(uint lightCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), lightCount);
        return bytes;
    }

    private static byte[] CreateLights(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);
        return stream.ToArray();
    }

    private static byte[] CreateLegacyLight(byte type, bool atten, Vector3 position, float intensity, float attenStart, float attenEnd)
    {
        byte[] bytes = new byte[32];
        bytes[0] = type;
        bytes[1] = atten ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), 0xFF00FF00);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(intensity));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24, 4), BitConverter.SingleToInt32Bits(attenStart));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28, 4), BitConverter.SingleToInt32Bits(attenEnd));
        return bytes;
    }

    private static byte[] CreateStandardLight(byte type, bool atten, ushort headerFlagsWord, Vector3 position, float intensity, Quaternion rotation, float attenStart, float attenEnd)
    {
        byte[] bytes = new byte[48];
        bytes[0] = type;
        bytes[1] = atten ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), headerFlagsWord);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), 0xFF00FF00);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(intensity));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24, 4), BitConverter.SingleToInt32Bits(rotation.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28, 4), BitConverter.SingleToInt32Bits(rotation.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(32, 4), BitConverter.SingleToInt32Bits(rotation.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(36, 4), BitConverter.SingleToInt32Bits(rotation.W));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40, 4), BitConverter.SingleToInt32Bits(attenStart));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(44, 4), BitConverter.SingleToInt32Bits(attenEnd));
        return bytes;
    }
}