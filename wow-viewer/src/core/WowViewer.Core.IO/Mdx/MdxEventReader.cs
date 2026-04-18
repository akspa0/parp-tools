using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxEventReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int PivtEntrySizeBytes = 12;

    public static MdxEventFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxEventFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX event reading requires a seekable stream.", nameof(stream));

        if (stream.Length < SignatureSizeBytes)
            throw new InvalidDataException($"MDX file '{sourcePath}' is too small to contain a signature.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            if (!string.Equals(signature, "MDLX", StringComparison.Ordinal))
                throw new InvalidDataException($"File '{sourcePath}' does not contain an MDLX signature. Found '{signature}'.");

            uint? version = null;
            string? modelName = null;
            List<MdxPivotPointSummary> pivotPoints = [];
            List<MdxEvent> events = [];
            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];

            while (stream.Position <= stream.Length - ChunkHeader.SizeInBytes)
            {
                long headerOffset = stream.Position;
                stream.ReadExactly(headerBytes);
                if (!TryReadMdxChunkHeader(headerBytes, out ChunkHeader header))
                    throw new InvalidDataException($"Could not decode MDX chunk header at offset {headerOffset}.");

                long dataOffset = stream.Position;
                long endOffset = checked(dataOffset + header.Size);
                if (endOffset > stream.Length)
                    throw new InvalidDataException($"MDX chunk {header.Id} at offset {headerOffset} overruns the stream length.");

                if (header.Id == MdxChunkIds.Vers && header.Size >= sizeof(uint))
                {
                    version = ReadUInt32At(stream, dataOffset);
                }
                else if (header.Id == MdxChunkIds.Modl && header.Size >= ModlNameSizeBytes)
                {
                    modelName = ReadFixedAsciiAt(stream, dataOffset, ModlNameSizeBytes);
                }
                else if (header.Id == MdxChunkIds.Pivt)
                {
                    pivotPoints = ReadPivtSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Evts)
                {
                    events = ReadClassicEvents(stream, dataOffset, header.Size, version, pivotPoints);
                }

                stream.Position = endOffset;
            }

            if (events.Count > 0 && pivotPoints.Count > 0)
                events = AssignDeferredPivots(events, pivotPoints);

            return new MdxEventFile(sourcePath, signature, version, modelName, events);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxEvent> ReadClassicEvents(Stream stream, long dataOffset, uint size, uint? version, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("EVTS(v1300): missing event count.");

            uint eventCount = ReadUInt32(stream);
            if (eventCount > 100000)
                throw new InvalidDataException($"EVTS(v1300): invalid event count {eventCount}.");

            List<MdxEvent> events = new(checked((int)eventCount));
            for (int index = 0; index < eventCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"EVTS(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"EVTS(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"EVTS(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxVector3NodeTrack? translationTrack, MdxQuaternionNodeTrack? rotationTrack, MdxVector3NodeTrack? scalingTrack) =
                    ReadNodeTracks(stream, nodeEnd, index, "EVTS(v1300)");

                MdxEventTrack? eventTrack = null;
                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KEVT":
                            eventTrack = ReadEventTrack(stream, entryEnd, tag, $"EVTS(v1300): {tag} payload overran the section.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                Vector3 pivotPoint = objectId >= 0 && objectId < pivotPoints.Count
                    ? pivotPoints[objectId].Position
                    : Vector3.Zero;

                events.Add(new MdxEvent(index, name, objectId, parentId, flags, pivotPoint, translationTrack, rotationTrack, scalingTrack, eventTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return events;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxEvent> AssignDeferredPivots(IReadOnlyList<MdxEvent> events, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        List<MdxEvent> reassigned = new(events.Count);
        foreach (MdxEvent mdxEvent in events)
        {
            Vector3 pivotPoint = mdxEvent.ObjectId >= 0 && mdxEvent.ObjectId < pivotPoints.Count
                ? pivotPoints[mdxEvent.ObjectId].Position
                : mdxEvent.PivotPoint;

            reassigned.Add(new MdxEvent(
                mdxEvent.Index,
                mdxEvent.Name,
                mdxEvent.ObjectId,
                mdxEvent.ParentId,
                mdxEvent.Flags,
                pivotPoint,
                mdxEvent.TranslationTrack,
                mdxEvent.RotationTrack,
                mdxEvent.ScalingTrack,
                mdxEvent.EventTrack));
        }

        return reassigned;
    }

    private static (string Name, int ObjectId, int ParentId, uint Flags, MdxVector3NodeTrack? TranslationTrack, MdxQuaternionNodeTrack? RotationTrack, MdxVector3NodeTrack? ScalingTrack) ReadNodeTracks(Stream stream, long nodeEnd, int index, string chunkLabel)
    {
        if (nodeEnd - stream.Position < 0x50 + 12)
            throw new InvalidDataException($"{chunkLabel}: truncated node payload at index {index}.");

        string name = ReadFixedAscii(stream, 0x50);
        int objectId = ReadInt32(stream);
        int parentId = ReadInt32(stream);
        uint flags = ReadUInt32(stream);
        MdxVector3NodeTrack? translationTrack = null;
        MdxQuaternionNodeTrack? rotationTrack = null;
        MdxVector3NodeTrack? scalingTrack = null;

        while (stream.Position <= nodeEnd - 4)
        {
            string tag = ReadTag(stream);
            switch (tag)
            {
                case "KGTR":
                    translationTrack = MdxTrackReader.ReadVector3Track(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                case "KGRT":
                    rotationTrack = MdxTrackReader.ReadQuaternionTrack(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                case "KGSC":
                    scalingTrack = MdxTrackReader.ReadVector3Track(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                default:
                    stream.Position = nodeEnd;
                    break;
            }
        }

        stream.Position = nodeEnd;
        return (name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack);
    }

    private static MdxEventTrack ReadEventTrack(Stream stream, long limit, string tag, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"EVTS(v1300): invalid {tag} key count {keyCount}.");

        int globalSequenceId = ReadInt32(stream);
        List<int> keyTimes = new(checked((int)keyCount));
        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
            keyTimes.Add(ReadInt32(stream));

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxEventTrack(tag, globalSequenceId, keyTimes);
    }

    private static List<MdxPivotPointSummary> ReadPivtSummary(Stream stream, long dataOffset, uint size)
    {
        if (size % PivtEntrySizeBytes != 0)
            throw new InvalidDataException($"Invalid PIVT size 0x{size:X}: expected multiple of {PivtEntrySizeBytes}.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            int count = checked((int)(size / PivtEntrySizeBytes));
            List<MdxPivotPointSummary> pivotPoints = new(count);
            for (int index = 0; index < count; index++)
                pivotPoints.Add(new MdxPivotPointSummary(index, ReadVector3(stream)));

            return pivotPoints;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadMdxChunkHeader(ReadOnlySpan<byte> data, out ChunkHeader header)
    {
        if (data.Length < ChunkHeader.SizeInBytes)
        {
            header = default;
            return false;
        }

        string idText = Encoding.ASCII.GetString(data[..4]);
        FourCC id = FourCC.FromString(idText);
        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        header = new ChunkHeader(id, size);
        return true;
    }

    private static uint ReadUInt32At(Stream stream, long offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            return ReadUInt32(stream);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadFixedAsciiAt(Stream stream, long offset, int size)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            return ReadFixedAscii(stream, size);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadFixedAscii(Stream stream, int size)
    {
        byte[] bytes = new byte[size];
        stream.ReadExactly(bytes);
        int terminatorIndex = Array.IndexOf(bytes, (byte)0);
        int count = terminatorIndex >= 0 ? terminatorIndex : bytes.Length;
        return Encoding.ASCII.GetString(bytes, 0, count);
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return Encoding.ASCII.GetString(bytes);
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream) => unchecked((int)ReadUInt32(stream));

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static Vector3 ReadVector3(Stream stream) => new(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
}
