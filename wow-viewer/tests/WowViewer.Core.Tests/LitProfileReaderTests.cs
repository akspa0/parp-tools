using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Lit;

namespace WowViewer.Core.Tests;

public sealed class LitProfileReaderTests
{
    [Fact]
    public void Read_TwoLightVersion84_ConsumesAllHeadersBeforeGroupPayloads()
    {
        byte[] bytes = CreateVersion84TwoLightFixture();
        using var stream = new MemoryStream(bytes, writable: false);

        LitFileProfile profile = LitProfileReader.Read(stream, "World\\Azeroth\\lights.lit");

        Assert.Equal(LitProfileReader.Version84, profile.VersionNumber);
        Assert.Equal(2, profile.RawLightCount);
        Assert.Equal(18, profile.TrackCount);
        Assert.Equal(0x1550, profile.GroupStride);
        Assert.False(profile.IsSinglePartialProfile);
        Assert.Equal(2, profile.Lights.Count);
        Assert.Equal("Default", profile.Lights[0].Header!.Name);
        Assert.True(profile.Lights[0].Header!.IsDefault);
        Assert.Equal("Local", profile.Lights[1].Header!.Name);
        Assert.Equal(new Vector3(100f, 200f, 300f), profile.Lights[1].Header!.Position);
        Assert.Equal(100f / 36f, profile.Lights[1].Header!.WorldPosition.X, 5);
        Assert.Equal(300f / 36f, profile.Lights[1].Header!.WorldPosition.Y, 5);
        Assert.Equal(200f / 36f, profile.Lights[1].Header!.WorldPosition.Z, 5);
        Vector3 rendererPosition = profile.Lights[1].Header!.ToRendererPosition(17066.66666f);
        Assert.Equal(17066.66666f - (300f / 36f), rendererPosition.X, 4);
        Assert.Equal(17066.66666f - (100f / 36f), rendererPosition.Y, 4);
        Assert.Equal(200f / 36f, rendererPosition.Z, 5);
        Assert.All(profile.Lights, light => Assert.Equal(4, light.Groups.Count));

        LitLightGroupProfile clear = profile.Lights[0].Groups[0];
        Assert.Equal(LitLightGroupKind.Clear, clear.Kind);
        Assert.Equal(0x1550, clear.EncodedSize);
        Assert.Equal(18, clear.Tracks.Count);
        Assert.Equal(6, clear.FloatBands.Count);
        Assert.Empty(clear.ParameterBands);
        Assert.Equal(17, clear.HighlightSky);
        Assert.Equal(23, clear.CloudMask);

        AssertTrackSurvives(clear, 0, expectedLength: 2);
        AssertTrackSurvives(clear, 1, expectedLength: 1);
        AssertTrackSurvives(clear, 2, expectedLength: 1);
        AssertTrackSurvives(clear, 6, expectedLength: 1);
        AssertTrackSurvives(clear, 7, expectedLength: 1);

        LitColorTrack direct = clear.Tracks[0];
        Assert.Equal(new Vector3(1f, 0f, 0f), direct.Keyframes[0].Color);
        Assert.Equal(new Vector3(0f, 0f, 1f), direct.Keyframes[1].Color);
        Assert.Equal(new Vector3(0.5f, 0f, 0.5f), direct.Evaluate(720f));
        Assert.Equal(new Vector3(0.5f, 0f, 0.5f), direct.Evaluate(2160f));
        Assert.Equal(direct.Evaluate(0f), direct.Evaluate(2880f));

        Assert.Equal(new Vector3(0f, 1f, 0f), clear.Tracks[1].Evaluate(1000f));
        Assert.Equal(new Vector3(0f, 0f, 1f), clear.Tracks[2].Evaluate(1000f));
        Assert.Equal(PackBgrx(128, 0, 32), clear.Tracks[6].Keyframes[0].PackedBgrx);
        Assert.Equal(PackBgrx(0, 255, 64), clear.Tracks[7].Keyframes[0].PackedBgrx);

        Assert.Equal(0, stream.Position);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(33)]
    public void Read_TrackLengthOutsideValidRange_RejectsInsteadOfClamping(int invalidLength)
    {
        byte[] bytes = CreateVersion84TwoLightFixture(invalidFirstTrackLength: invalidLength);
        using var stream = new MemoryStream(bytes, writable: false);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => LitProfileReader.Read(stream));

        Assert.Contains($"invalid length {invalidLength}", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_TruncatedVersion84Payload_IsRejectedBeforeDecode()
    {
        byte[] complete = CreateVersion84TwoLightFixture();
        byte[] truncated = complete[..^1];
        using var stream = new MemoryStream(truncated, writable: false);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => LitProfileReader.Read(stream));

        Assert.Contains("Truncated LIT payload", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_NegativeCountVersion02_ProducesObservedPreAlphaPartialShape()
    {
        byte[] bytes = CreateVersion02PreAlphaPartialFixture();
        Assert.Equal(8 + 64 + 0x1484, bytes.Length);
        using var stream = new MemoryStream(bytes, writable: false);

        LitFileProfile profile = LitProfileReader.Read(stream, "legacy.lit");

        Assert.Equal(LitProfileReader.Version02, profile.VersionNumber);
        Assert.Equal(-1, profile.RawLightCount);
        Assert.True(profile.IsSinglePartialProfile);
        Assert.Equal(9, profile.TrackCount);
        Assert.Equal(0x1484, profile.GroupStride);

        LitLightProfile light = Assert.Single(profile.Lights);
        Assert.True(light.IsPartial);
        Assert.Equal("Global Light", light.Header!.Name);
        Assert.True(light.Header.IsDefault);
        Assert.Equal(2, light.Groups.Count);
        LitLightGroupProfile group = light.Groups[0];
        Assert.Equal(LitLightGroupKind.Partial, group.Kind);
        Assert.Equal(0xA24, group.EncodedSize);
        Assert.Equal(9, group.Tracks.Count);
        Assert.Equal(2, group.FloatBands.Count);
        Assert.All(group.FloatBands, band => Assert.Equal(32, band.Samples.Count));
        Assert.Equal(10000f, group.FloatBands[0].Samples[0]);
        Assert.Equal(0.25f, group.FloatBands[1].Samples[0]);
        Assert.Null(group.HighlightSky);
        Assert.Null(group.CloudMask);
        Assert.Empty(group.ParameterBands);
        Assert.Equal(4, group.Tracks[0].DeclaredLength);
        Assert.Equal(3, group.Tracks[8].DeclaredLength);
        Assert.Equal(new Vector3(0x0C / 255f, 0x3E / 255f, 0xFB / 255f), group.Tracks[0].Evaluate(0f));
        Assert.Equal(LitLightGroupKind.LegacyPartialAlternate, light.Groups[1].Kind);
        Assert.Equal(9, light.Groups[1].Tracks.Count);
    }

    [Fact]
    public void Read_UnsupportedVersionOrCount_IsRejected()
    {
        using var unsupportedVersion = new MemoryStream();
        using (var writer = new BinaryWriter(unsupportedVersion, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x80000006u);
            writer.Write(0);
        }
        unsupportedVersion.Position = 0;

        Assert.Throws<InvalidDataException>(() => LitProfileReader.Read(unsupportedVersion));

        using var unsupportedCount = new MemoryStream();
        using (var writer = new BinaryWriter(unsupportedCount, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(LitProfileReader.Version84);
            writer.Write(-2);
        }
        unsupportedCount.Position = 0;

        Assert.Throws<InvalidDataException>(() => LitProfileReader.Read(unsupportedCount));
    }

    [Theory]
    [InlineData(LitProfileReader.Version83, 14, 0x1140, 0)]
    [InlineData(LitProfileReader.Version85, 18, 0x15F0, 4)]
    public void Read_SupportedModernPartialLayout_UsesExactVersionStride(
        uint version,
        int trackCount,
        int expectedStride,
        int expectedParameterBands)
    {
        byte[] bytes = CreateModernPartialFixture(version, trackCount, expectedParameterBands > 0);
        Assert.Equal(8 + expectedStride, bytes.Length);
        using var stream = new MemoryStream(bytes, writable: false);

        LitFileProfile profile = LitProfileReader.Read(stream);

        Assert.Equal(version, profile.VersionNumber);
        Assert.Equal(trackCount, profile.TrackCount);
        Assert.Equal(expectedStride, profile.GroupStride);
        LitLightGroupProfile group = Assert.Single(Assert.Single(profile.Lights).Groups);
        Assert.Equal(expectedStride, group.EncodedSize);
        Assert.Equal(expectedParameterBands, group.ParameterBands.Count);
        Assert.All(group.ParameterBands, band => Assert.Equal(10, band.Samples.Count));
    }

    private static void AssertTrackSurvives(LitLightGroupProfile group, int trackIndex, int expectedLength)
    {
        Assert.True(group.TryGetTrack(trackIndex, out LitColorTrack track));
        Assert.Equal(trackIndex, track.Index);
        Assert.Equal(expectedLength, track.DeclaredLength);
        Assert.Equal(expectedLength, track.Keyframes.Count);
    }

    private static byte[] CreateVersion84TwoLightFixture(int? invalidFirstTrackLength = null)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(LitProfileReader.Version84);
        writer.Write(2);

        WriteLightHeader(writer, -1, -1, -1, Vector3.Zero, 36f, 18f, "Default");
        WriteLightHeader(writer, 2, 3, 4, new Vector3(100f, 200f, 300f), 72f, 36f, "Local");

        var knownTracks = new Dictionary<int, IReadOnlyList<(int Time, uint Color)>>
        {
            [0] = [(0, PackBgrx(255, 0, 0)), (1440, PackBgrx(0, 0, 255))],
            [1] = [(0, PackBgrx(0, 255, 0))],
            [2] = [(0, PackBgrx(0, 0, 255))],
            [6] = [(0, PackBgrx(128, 0, 32))],
            [7] = [(0, PackBgrx(0, 255, 64))],
        };

        for (int lightIndex = 0; lightIndex < 2; lightIndex++)
        {
            for (int groupIndex = 0; groupIndex < 4; groupIndex++)
            {
                IReadOnlyDictionary<int, IReadOnlyList<(int Time, uint Color)>> tracks = lightIndex == 0 && groupIndex == 0
                    ? knownTracks
                    : new Dictionary<int, IReadOnlyList<(int Time, uint Color)>>();
                WriteModernGroup(
                    writer,
                    trackCount: 18,
                    tracks,
                    invalidTrackLength: lightIndex == 0 && groupIndex == 0 ? invalidFirstTrackLength : null);
            }
        }

        return stream.ToArray();
    }

    private static byte[] CreateVersion02PreAlphaPartialFixture()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(LitProfileReader.Version02);
        writer.Write(-1);

        writer.Write(-1);
        writer.Write(-1);
        writer.Write(-1);
        writer.Write(0f);
        writer.Write(0f);
        writer.Write(0f);
        writer.Write(0f);
        byte[] nameBytes = new byte[32];
        Encoding.UTF8.GetBytes("Global Light").CopyTo(nameBytes, 0);
        writer.Write(nameBytes);
        writer.Write(0);

        for (int index = 0; index < 15; index++)
            writer.Write(0);

        WriteVersion02PreAlphaDataSet(
            writer,
            new Dictionary<int, IReadOnlyList<(int Time, uint Color)>>
            {
                [0] =
                [
                    (0, PackBgrx(0x0C, 0x3E, 0xFB)),
                    (720, PackBgrx(0x00, 0x8B, 0xF0)),
                    (1440, PackBgrx(0x99, 0xCD, 0xFF)),
                    (2160, PackBgrx(0x00, 0x89, 0xFF)),
                ],
                [8] =
                [
                    (0, PackBgrx(0x42, 0x42, 0x42)),
                    (720, PackBgrx(0x7F, 0x7F, 0x7F)),
                    (1440, PackBgrx(0x4D, 0x4D, 0x4D)),
                ],
            },
            firstFloatBandValue: 10000f,
            secondFloatBandValue: 0.25f);

        WriteVersion02PreAlphaDataSet(
            writer,
            new Dictionary<int, IReadOnlyList<(int Time, uint Color)>>(),
            firstFloatBandValue: 12000f,
            secondFloatBandValue: 0.5f);

        return stream.ToArray();
    }

    private static void WriteVersion02PreAlphaDataSet(
        BinaryWriter writer,
        IReadOnlyDictionary<int, IReadOnlyList<(int Time, uint Color)>> tracks,
        float firstFloatBandValue,
        float secondFloatBandValue)
    {
        long start = writer.BaseStream.Position;
        for (int trackIndex = 0; trackIndex < 9; trackIndex++)
            writer.Write(tracks.TryGetValue(trackIndex, out IReadOnlyList<(int Time, uint Color)>? samples) ? samples.Count : 0);

        for (int trackIndex = 0; trackIndex < 9; trackIndex++)
        {
            tracks.TryGetValue(trackIndex, out IReadOnlyList<(int Time, uint Color)>? samples);
            for (int slotIndex = 0; slotIndex < 32; slotIndex++)
            {
                bool populated = samples != null && slotIndex < samples.Count;
                writer.Write(populated ? samples![slotIndex].Time : 0);
                writer.Write(populated ? samples![slotIndex].Color : 0u);
            }
        }

        WriteFloatBand(writer, firstFloatBandValue);
        WriteFloatBand(writer, secondFloatBandValue);
        Assert.Equal(0xA24, writer.BaseStream.Position - start);
    }

    private static byte[] CreateModernPartialFixture(uint version, int trackCount, bool includeParameterBands)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(version);
        writer.Write(-1);

        WriteModernGroup(
            writer,
            trackCount,
            new Dictionary<int, IReadOnlyList<(int Time, uint Color)>>(),
            invalidTrackLength: null,
            includeParameterBands: includeParameterBands);
        return stream.ToArray();
    }

    private static void WriteLightHeader(
        BinaryWriter writer,
        int chunkX,
        int chunkY,
        int chunkRadius,
        Vector3 position,
        float radius,
        float dropoff,
        string name)
    {
        writer.Write(chunkX);
        writer.Write(chunkY);
        writer.Write(chunkRadius);
        writer.Write(position.X);
        writer.Write(position.Y);
        writer.Write(position.Z);
        writer.Write(radius);
        writer.Write(dropoff);

        byte[] nameBytes = new byte[32];
        Encoding.UTF8.GetBytes(name).CopyTo(nameBytes, 0);
        writer.Write(nameBytes);
    }

    private static void WriteModernGroup(
        BinaryWriter writer,
        int trackCount,
        IReadOnlyDictionary<int, IReadOnlyList<(int Time, uint Color)>> tracks,
        int? invalidTrackLength,
        bool includeParameterBands = false)
    {
        long start = writer.BaseStream.Position;
        for (int trackIndex = 0; trackIndex < trackCount; trackIndex++)
        {
            int length = tracks.TryGetValue(trackIndex, out IReadOnlyList<(int Time, uint Color)>? samples)
                ? samples.Count
                : 0;
            writer.Write(trackIndex == 0 && invalidTrackLength.HasValue ? invalidTrackLength.Value : length);
        }

        for (int trackIndex = 0; trackIndex < trackCount; trackIndex++)
        {
            tracks.TryGetValue(trackIndex, out IReadOnlyList<(int Time, uint Color)>? samples);
            for (int slotIndex = 0; slotIndex < 32; slotIndex++)
            {
                bool populated = samples != null && slotIndex < samples.Count;
                writer.Write(populated ? samples![slotIndex].Time : 0);
                writer.Write(populated ? samples![slotIndex].Color : 0u);
            }
        }

        WriteFloatBand(writer, firstValue: 1500f);
        WriteFloatBand(writer, firstValue: 0.25f);
        writer.Write(17);
        WriteFloatBand(writer, firstValue: 1f);
        WriteFloatBand(writer, firstValue: 2f);
        WriteFloatBand(writer, firstValue: 3f);
        WriteFloatBand(writer, firstValue: 4f);
        writer.Write(23);

        if (includeParameterBands)
        {
            for (int bandIndex = 0; bandIndex < 4; bandIndex++)
            {
                for (int sampleIndex = 0; sampleIndex < 10; sampleIndex++)
                    writer.Write(sampleIndex == 0 ? bandIndex + 0.5f : 0f);
            }
        }

        long expectedStride = includeParameterBands
            ? 0x15F0
            : trackCount == 14
                ? 0x1140
                : 0x1550;
        Assert.Equal(expectedStride, writer.BaseStream.Position - start);
    }

    private static void WriteFloatBand(BinaryWriter writer, float firstValue)
    {
        for (int sampleIndex = 0; sampleIndex < 32; sampleIndex++)
            writer.Write(sampleIndex == 0 ? firstValue : 0f);
    }

    private static uint PackBgrx(byte red, byte green, byte blue, byte unused = 0)
    {
        return (uint)(blue | (green << 8) | (red << 16) | (unused << 24));
    }
}
