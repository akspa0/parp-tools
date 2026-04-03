using System.Buffers.Binary;
using WowViewer.Core.IO.Lit;

namespace WowViewer.Core.Tests;

public sealed class LitSummaryReaderTests
{
    [Fact]
    public void Read_ListBasedLitBuffer_ProducesSemanticSummary()
    {
        using MemoryStream stream = new();
        stream.Write(CreateUInt32Payload(0x80000004));
        stream.Write(CreateInt32Payload(2));
        stream.Write(CreateLightListEntry(-1, -1, -1, "default"));
        stream.Write(CreateLightListEntry(0, 0, 0, "Elwynn"));
        stream.Write(new byte[128]);
        stream.Position = 0;

        var summary = LitSummaryReader.Read(stream, "lights.lit");

        Assert.Equal(0x80000004u, summary.VersionNumber);
        Assert.Equal(2, summary.LightCount);
        Assert.Equal(2, summary.ListEntryCount);
        Assert.False(summary.UsesSinglePartialEntry);
        Assert.True(summary.HasDefaultFirstEntry);
        Assert.Equal(2, summary.NamedEntryCount);
        Assert.Equal(128, summary.RemainingPayloadBytes);
    }

    [Fact]
    public void Read_SinglePartialLitBuffer_ProducesPartialSummary()
    {
        using MemoryStream stream = new();
        stream.Write(CreateUInt32Payload(0x80000003));
        stream.Write(CreateInt32Payload(-1));
        stream.Write(new byte[0x15F0]);
        stream.Position = 0;

        var summary = LitSummaryReader.Read(stream, "lights.lit");

        Assert.Equal(0x80000003u, summary.VersionNumber);
        Assert.Equal(-1, summary.LightCount);
        Assert.Equal(0, summary.ListEntryCount);
        Assert.True(summary.UsesSinglePartialEntry);
        Assert.False(summary.HasDefaultFirstEntry);
        Assert.Equal(0, summary.NamedEntryCount);
        Assert.Equal(0x15F0, summary.RemainingPayloadBytes);
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

    private static byte[] CreateLightListEntry(int chunkX, int chunkY, int chunkRadius, string name)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x00, 4), chunkX);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x04, 4), chunkY);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x08, 4), chunkRadius);
        byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
        Array.Copy(nameBytes, 0, bytes, 0x20, Math.Min(nameBytes.Length, 32));
        return bytes;
    }
}