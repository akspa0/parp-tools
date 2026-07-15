using System.Buffers.Binary;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.IO.Lit;

/// <summary>
/// Strict decoder for Alpha-era World/&lt;map&gt;/lights.lit payloads.
/// </summary>
public static class LitProfileReader
{
    public const uint Version02 = 0x00000002;
    public const uint Version83 = 0x80000003;
    public const uint Version84 = 0x80000004;
    public const uint Version85 = 0x80000005;

    public const int TimeUnitsPerDay = 2880;
    public const int MaximumTrackLength = 32;

    private const int FileHeaderSize = 8;
    private const int LightHeaderSize = 64;
    private const int TrackSlotCount = 32;
    private const int FloatBandSampleCount = 32;
    private const int ParameterBandSampleCount = 10;

    public static LitFileProfile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static LitFileProfile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanRead || !stream.CanSeek)
            throw new ArgumentException("LIT profile reading requires a readable, seekable stream.", nameof(stream));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            if (stream.Length < FileHeaderSize)
                throw new InvalidDataException("LIT file is too short to contain its 8-byte header.");

            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            uint version = reader.ReadUInt32();
            int rawLightCount = reader.ReadInt32();
            LitLayout layout = ResolveLayout(version, rawLightCount);

            if (rawLightCount < -1)
                throw new InvalidDataException($"Unsupported LIT light count {rawLightCount}; only -1 or a non-negative count is valid.");

            int headerCount = Math.Max(rawLightCount, 0);
            int dataLightCount = rawLightCount == -1 ? 1 : rawLightCount;
            int groupsPerLight = rawLightCount == -1 ? 1 : 4;
            long expectedLength;
            try
            {
                expectedLength = checked(
                    FileHeaderSize
                    + ((long)headerCount * LightHeaderSize)
                    + ((long)dataLightCount * groupsPerLight * layout.GroupStride));
            }
            catch (OverflowException ex)
            {
                throw new InvalidDataException($"LIT light count {rawLightCount} overflows the supported file layout.", ex);
            }

            if (stream.Length < expectedLength)
            {
                throw new InvalidDataException(
                    $"Truncated LIT payload for version 0x{version:X8}: expected {expectedLength} bytes from count {rawLightCount}, found {stream.Length}.");
            }

            if (stream.Length > expectedLength)
            {
                throw new InvalidDataException(
                    $"Unsupported trailing LIT payload for version 0x{version:X8}: expected {expectedLength} bytes from count {rawLightCount}, found {stream.Length}.");
            }

            var headers = new List<LitLightHeaderProfile>(headerCount);
            for (int lightIndex = 0; lightIndex < headerCount; lightIndex++)
                headers.Add(ReadLightHeader(reader, lightIndex));

            var lights = new List<LitLightProfile>(dataLightCount);
            if (rawLightCount == -1)
            {
                LitLightGroupProfile partial = ReadGroup(
                    reader,
                    layout,
                    groupIndex: 0,
                    LitLightGroupKind.Partial);
                lights.Add(new LitLightProfile(0, header: null, [partial]));
            }
            else
            {
                for (int lightIndex = 0; lightIndex < dataLightCount; lightIndex++)
                {
                    var groups = new List<LitLightGroupProfile>(4);
                    for (int groupIndex = 0; groupIndex < 4; groupIndex++)
                    {
                        groups.Add(ReadGroup(
                            reader,
                            layout,
                            groupIndex,
                            (LitLightGroupKind)groupIndex));
                    }

                    lights.Add(new LitLightProfile(lightIndex, headers[lightIndex], groups));
                }
            }

            if (stream.Position != expectedLength)
            {
                throw new InvalidDataException(
                    $"LIT decoder consumed {stream.Position} bytes but the version layout requires {expectedLength}.");
            }

            return new LitFileProfile(
                sourcePath,
                version,
                rawLightCount,
                layout.TrackCount,
                layout.GroupStride,
                lights);
        }
        catch (EndOfStreamException ex)
        {
            throw new InvalidDataException("LIT payload ended inside a declared structure.", ex);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static LitLayout ResolveLayout(uint version, int rawLightCount)
    {
        return version switch
        {
            Version02 when rawLightCount == -1 => new LitLayout(17, 0x14C4, true, false),
            Version02 => throw new InvalidDataException("LIT version 2 is supported only for the negative-count single-partial layout."),
            Version83 => new LitLayout(14, 0x1140, false, false),
            Version84 => new LitLayout(18, 0x1550, false, false),
            Version85 => new LitLayout(18, 0x15F0, false, true),
            _ => throw new InvalidDataException($"Unsupported LIT version 0x{version:X8}."),
        };
    }

    private static LitLightHeaderProfile ReadLightHeader(BinaryReader reader, int lightIndex)
    {
        int chunkX = reader.ReadInt32();
        int chunkY = reader.ReadInt32();
        int chunkRadius = reader.ReadInt32();
        var position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        float radius = reader.ReadSingle();
        float dropoff = reader.ReadSingle();
        byte[] nameBytes = reader.ReadBytes(32);
        if (nameBytes.Length != 32)
            throw new EndOfStreamException();

        return new LitLightHeaderProfile(
            lightIndex,
            chunkX,
            chunkY,
            chunkRadius,
            position,
            radius,
            dropoff,
            DecodeName(nameBytes));
    }

    private static LitLightGroupProfile ReadGroup(
        BinaryReader reader,
        LitLayout layout,
        int groupIndex,
        LitLightGroupKind kind)
    {
        long groupStart = reader.BaseStream.Position;
        var lengths = new int[layout.TrackCount];
        for (int trackIndex = 0; trackIndex < lengths.Length; trackIndex++)
        {
            int length = reader.ReadInt32();
            if (length is < 0 or > MaximumTrackLength)
            {
                throw new InvalidDataException(
                    $"LIT group {groupIndex} track {trackIndex} declares invalid length {length}; expected 0..{MaximumTrackLength}.");
            }

            lengths[trackIndex] = length;
        }

        var tracks = new List<LitColorTrack>(layout.TrackCount);
        for (int trackIndex = 0; trackIndex < layout.TrackCount; trackIndex++)
        {
            var keyframes = new List<LitColorKeyframe>(lengths[trackIndex]);
            for (int slotIndex = 0; slotIndex < TrackSlotCount; slotIndex++)
            {
                int timeOfDay = reader.ReadInt32();
                uint packedBgrx = reader.ReadUInt32();
                if (slotIndex >= lengths[trackIndex])
                    continue;

                if (timeOfDay is < 0 or > TimeUnitsPerDay)
                {
                    throw new InvalidDataException(
                        $"LIT group {groupIndex} track {trackIndex} key {slotIndex} has invalid time {timeOfDay}; expected 0..{TimeUnitsPerDay}.");
                }

                keyframes.Add(new LitColorKeyframe(timeOfDay, packedBgrx, DecodeBgrx(packedBgrx)));
            }

            tracks.Add(new LitColorTrack(trackIndex, lengths[trackIndex], keyframes));
        }

        List<LitFloatBand> floatBands;
        int? highlightSky;
        int? cloudMask;
        List<LitFloatBand> parameterBands;

        if (layout.IsLegacyVersion02)
        {
            floatBands = ReadFloatBands(reader, bandCount: 7, FloatBandSampleCount);
            highlightSky = null;
            cloudMask = null;
            parameterBands = [];
        }
        else
        {
            floatBands = new List<LitFloatBand>(6)
            {
                ReadFloatBand(reader, 0, FloatBandSampleCount),
                ReadFloatBand(reader, 1, FloatBandSampleCount),
            };
            highlightSky = reader.ReadInt32();
            for (int bandIndex = 2; bandIndex < 6; bandIndex++)
                floatBands.Add(ReadFloatBand(reader, bandIndex, FloatBandSampleCount));

            cloudMask = reader.ReadInt32();
            parameterBands = layout.HasParameterBands
                ? ReadFloatBands(reader, bandCount: 4, ParameterBandSampleCount)
                : [];
        }

        long bytesRead = reader.BaseStream.Position - groupStart;
        if (bytesRead != layout.GroupStride)
        {
            throw new InvalidDataException(
                $"LIT group {groupIndex} consumed 0x{bytesRead:X} bytes; version layout requires 0x{layout.GroupStride:X}.");
        }

        return new LitLightGroupProfile(
            groupIndex,
            kind,
            tracks,
            floatBands,
            highlightSky,
            cloudMask,
            parameterBands,
            checked((int)bytesRead));
    }

    private static List<LitFloatBand> ReadFloatBands(BinaryReader reader, int bandCount, int sampleCount)
    {
        var bands = new List<LitFloatBand>(bandCount);
        for (int bandIndex = 0; bandIndex < bandCount; bandIndex++)
            bands.Add(ReadFloatBand(reader, bandIndex, sampleCount));

        return bands;
    }

    private static LitFloatBand ReadFloatBand(BinaryReader reader, int bandIndex, int sampleCount)
    {
        var samples = new float[sampleCount];
        for (int sampleIndex = 0; sampleIndex < samples.Length; sampleIndex++)
            samples[sampleIndex] = reader.ReadSingle();

        return new LitFloatBand(bandIndex, samples);
    }

    private static string DecodeName(ReadOnlySpan<byte> bytes)
    {
        int length = 0;
        while (length < bytes.Length && bytes[length] != 0 && bytes[length] != 0xFD)
            length++;

        return length == 0 ? string.Empty : Encoding.UTF8.GetString(bytes[..length]).Trim();
    }

    private static Vector3 DecodeBgrx(uint packedBgrx)
    {
        float blue = (packedBgrx & 0xFF) / 255f;
        float green = ((packedBgrx >> 8) & 0xFF) / 255f;
        float red = ((packedBgrx >> 16) & 0xFF) / 255f;
        return new Vector3(red, green, blue);
    }

    private readonly record struct LitLayout(
        int TrackCount,
        int GroupStride,
        bool IsLegacyVersion02,
        bool HasParameterBands);
}
