using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Lit;
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
        stream.Write(CreateLightListEntry(-1, -1, -1, Vector3.Zero, 48f, 12f, "default"));
        stream.Write(CreateLightListEntry(0, 0, 0, new Vector3(100f, 200f, 300f), 96f, 24f, "Elwynn"));
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
        Assert.Equal(2, summary.Entries.Count);
        Assert.True(summary.Entries[0].IsDefaultEntry);
        Assert.Equal("default", summary.Entries[0].Name);
        // Raw file values are preserved verbatim; decoded source data is never rewritten.
        Assert.Equal(new Vector3(100f, 200f, 300f), summary.Entries[1].RawPosition);
        Assert.Equal(96f, summary.Entries[1].RawLightRadius);
        Assert.Equal(24f, summary.Entries[1].RawLightDropoff);

        // LIT spatial records are client fixed-point at 1/36 world units. Without this conversion
        // every plotted light lands ~36x off the map.
        Assert.Equal(100f / 36f, summary.Entries[1].Position.X, 5);
        Assert.Equal(200f / 36f, summary.Entries[1].Position.Y, 5);
        Assert.Equal(300f / 36f, summary.Entries[1].Position.Z, 5);
        Assert.Equal(96f / 36f, summary.Entries[1].LightRadius, 5);
        Assert.Equal(24f / 36f, summary.Entries[1].LightDropoff, 5);
        Assert.Equal(120f / 36f, summary.Entries[1].OuterRadius, 5);
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
        Assert.Empty(summary.Entries);
    }

    [Fact]
    public void Sample_PositionWithinLightVolume_ReturnsLocalEntry()
    {
        var summary = new LitSummary(
            "lights.lit",
            0x80000004,
            2,
            2,
            [
                new LitListEntrySummary(0, -1, -1, -1, Vector3.Zero, 0f, 0f, "default"),
                // Constructed in RAW fixed-point units (world * 36), so this entry sits at world
                // (100,0,0) with a 50-unit core and 25-unit falloff.
                new LitListEntrySummary(1, 0, 0, 0, new Vector3(3600f, 0f, 0f), 1800f, 900f, "Elwynn")
            ],
            usesSinglePartialEntry: false,
            hasDefaultFirstEntry: true,
            namedEntryCount: 2,
            remainingPayloadBytes: 0);

        IReadOnlyList<LitSpatialSampleCandidate> candidates = LitSpatialSampler.Sample(summary, new Vector3(110f, 0f, 0f));

        LitSpatialSampleCandidate candidate = Assert.Single(candidates);
        Assert.Equal("Elwynn", candidate.Entry.Name);
        Assert.True(candidate.WithinCoreRadius);
        Assert.Equal(1f, candidate.Influence);
    }

    [Fact]
    public void Sample_PositionOutsideAllVolumes_FallsBackToDefaultEntry()
    {
        var summary = new LitSummary(
            "lights.lit",
            0x80000004,
            1,
            1,
            [new LitListEntrySummary(0, -1, -1, -1, Vector3.Zero, 0f, 0f, "default")],
            usesSinglePartialEntry: false,
            hasDefaultFirstEntry: true,
            namedEntryCount: 1,
            remainingPayloadBytes: 0);

        IReadOnlyList<LitSpatialSampleCandidate> candidates = LitSpatialSampler.Sample(summary, new Vector3(2048f, 2048f, 0f));

        LitSpatialSampleCandidate candidate = Assert.Single(candidates);
        Assert.True(candidate.IsFallbackDefault);
        Assert.Equal("default", candidate.Entry.Name);
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

    private static byte[] CreateLightListEntry(int chunkX, int chunkY, int chunkRadius, Vector3 position, float lightRadius, float lightDropoff, string name)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x00, 4), chunkX);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x04, 4), chunkY);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0x08, 4), chunkRadius);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0x0C, 4), position.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0x10, 4), position.Y);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0x14, 4), position.Z);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0x18, 4), lightRadius);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0x1C, 4), lightDropoff);
        byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
        Array.Copy(nameBytes, 0, bytes, 0x20, Math.Min(nameBytes.Length, 32));
        return bytes;
    }
}