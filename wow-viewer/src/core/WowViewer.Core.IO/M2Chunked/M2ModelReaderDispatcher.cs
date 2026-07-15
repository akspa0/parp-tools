using System.Buffers.Binary;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Era100;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2Chunked;

public readonly record struct M2DispatchResult(M2ModelDocument Document, M2Era1121EraTag Era);

public static class M2ModelReaderDispatcher
{
    public static M2ModelDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2ModelDocument Read(Stream stream, string sourcePath, Func<string, byte[]?>? companionReader = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        byte[] bytes = ReadAllBytes(stream);
        if (bytes.Length < sizeof(uint))
            throw new InvalidDataException($"Model file '{sourcePath}' is too small to contain a valid magic.");

        using MemoryStream memoryStream = new(bytes, writable: false);
        return DetectAndDispatch(memoryStream, sourcePath, companionReader).Document;
    }

    public static M2DispatchResult ReadDetailed(string path, Func<string, byte[]?>? companionReader = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadDetailed(stream, Path.GetFullPath(path), companionReader);
    }

    public static M2DispatchResult ReadDetailed(Stream stream, string sourcePath, Func<string, byte[]?>? companionReader = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        byte[] bytes = ReadAllBytes(stream);
        if (bytes.Length < sizeof(uint))
            throw new InvalidDataException($"Model file '{sourcePath}' is too small to contain a valid magic.");

        using MemoryStream memoryStream = new(bytes, writable: false);
        return DetectAndDispatch(memoryStream, sourcePath, companionReader);
    }

    public static M2Era1121EraTag DetectEra(ReadOnlySpan<byte> headerBytes, string sourcePath)
    {
        if (headerBytes.Length < sizeof(uint))
            throw new InvalidDataException($"Model file '{sourcePath}' is too small to contain a valid magic.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(headerBytes[..sizeof(uint)]);
        if (magic == MdxMagic.Mdlx)
            return M2Era1121EraTag.Mdlx;

        if (magic == M2Era1121Constants.Md20Magic)
        {
            if (headerBytes.Length < M2Era1121Constants.DispatchHeaderSizeBytes)
                throw new InvalidDataException($"M2 file '{sourcePath}' is too small to contain an MD20 magic+version pair.");

            uint version = BinaryPrimitives.ReadUInt32LittleEndian(headerBytes.Slice(sizeof(uint), sizeof(uint)));
            M2Era1121Version eraVersion = M2Era1121VersionExtensions.FromUInt(version);
            if (eraVersion.Is1121())
            {
                // Both 1.0.0 and 1.12.1 use version 0x100, but they have completely
                // different header layouts. Validate the 1.0.0 layout first; if the
                // vertices/divisions/textures fields at the 1.0.0 header positions
                // produce sane offsets, it's a 1.0.0 file. Otherwise fall back to 1.12.1.
                if (version == 0x100u && M2Era100ModelReader.ValidateLayout(headerBytes, sourcePath))
                    return M2Era1121EraTag.Md20_1X_V100_Era100;

                return eraVersion == M2Era1121Version.V101 ? M2Era1121EraTag.Md20_1X_V101 : M2Era1121EraTag.Md20_1X_V100;
            }

            if (version == 0x108u)
                return M2Era1121EraTag.Md20_3X_V108;

            if (version >= 0x109u)
                return M2Era1121EraTag.Md20_4X_V109;

            throw new NotSupportedException(
                $"MD20 v0x{version:X} is the 2.x TBC era, which is not yet supported. Tracked under spec 049.");
        }

        throw new InvalidDataException(
            $"Model file '{sourcePath}' has unsupported M2 magic '0x{magic:X8}' (expected MD20 or MDLX).");
    }

    private static M2DispatchResult DetectAndDispatch(MemoryStream memoryStream, string sourcePath, Func<string, byte[]?>? companionReader)
    {
        byte[] headerSnapshot = memoryStream.ToArray();
        M2Era1121EraTag era = DetectEra(headerSnapshot, sourcePath);

        using MemoryStream dispatchStream = new(headerSnapshot, writable: false);
        return era switch
        {
            M2Era1121EraTag.Mdlx => new M2DispatchResult(
                M2ChunkedModelReader.Read(dispatchStream, sourcePath, companionReader),
                era),
            M2Era1121EraTag.Md20_1X_V100_Era100 => new M2DispatchResult(
                M2Era100ModelReader.Read(dispatchStream, sourcePath),
                era),
            M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101 => new M2DispatchResult(
                M2Era1121ModelReader.Read(dispatchStream, sourcePath),
                era),
            M2Era1121EraTag.Md20_3X_V108 or M2Era1121EraTag.Md20_4X_V109 => new M2DispatchResult(
                M2ModelReader.Read(dispatchStream, sourcePath),
                era),
            _ => throw new InvalidDataException(
                $"Model file '{sourcePath}' did not resolve to a supported M2 era."),
        };
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        if (!stream.CanSeek)
            throw new ArgumentException("Model dispatch requires a seekable stream.", nameof(stream));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }
}
