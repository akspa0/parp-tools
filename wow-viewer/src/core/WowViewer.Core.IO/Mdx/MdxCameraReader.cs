using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxCameraReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int CamsNameSizeBytes = 0x50;

    public static MdxCameraFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxCameraFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX camera reading requires a seekable stream.", nameof(stream));

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
            List<MdxCamera> cameras = [];
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
                else if (header.Id == MdxChunkIds.Cams)
                {
                    cameras = ReadClassicCameras(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxCameraFile(sourcePath, signature, version, modelName, cameras);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxCamera> ReadClassicCameras(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("CAMS(v1300): missing camera count.");

            uint cameraCount = ReadUInt32(stream);
            if (cameraCount > 100000)
                throw new InvalidDataException($"CAMS(v1300): invalid camera count {cameraCount}.");

            List<MdxCamera> cameras = new(checked((int)cameraCount));
            for (int index = 0; index < cameraCount; index++)
            {
                long entryStart = stream.Position;
                if (chunkEnd - entryStart < sizeof(uint))
                    throw new InvalidDataException($"CAMS(v1300): truncated camera header at index {index}.");

                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"CAMS(v1300): invalid camera size 0x{entrySize:X} at index {index}.");

                if (entryEnd - stream.Position < CamsNameSizeBytes + 36)
                    throw new InvalidDataException($"CAMS(v1300): truncated camera payload at index {index}.");

                string name = ReadFixedAscii(stream, CamsNameSizeBytes);
                Vector3 pivotPoint = ReadVector3(stream);
                float fieldOfView = ReadSingle(stream);
                float farClip = ReadSingle(stream);
                float nearClip = ReadSingle(stream);
                Vector3 targetPivotPoint = ReadVector3(stream);

                MdxVector3NodeTrack? positionTrack = null;
                MdxScalarTrack? rollTrack = null;
                MdxScalarTrack? visibilityTrack = null;
                MdxVector3NodeTrack? targetPositionTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KCTR":
                            positionTrack = MdxTrackReader.ReadVector3Track(stream, entryEnd, tag, "CAMS(v1300)", $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KCRL":
                            rollTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "CAMS(v1300)", $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KVIS":
                            visibilityTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "CAMS(v1300)", $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KTTR":
                            targetPositionTrack = MdxTrackReader.ReadVector3Track(stream, entryEnd, tag, "CAMS(v1300)", $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                stream.Position = entryEnd;
                cameras.Add(new MdxCamera(index, name, pivotPoint, fieldOfView, farClip, nearClip, targetPivotPoint, positionTrack, rollTrack, visibilityTrack, targetPositionTrack));
            }

            stream.Position = chunkEnd;
            return cameras;
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

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static Vector3 ReadVector3(Stream stream) => new(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
}
