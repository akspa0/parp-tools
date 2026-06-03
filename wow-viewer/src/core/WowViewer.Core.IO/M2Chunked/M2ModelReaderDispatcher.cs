using System.Buffers.Binary;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2Chunked;

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
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0, sizeof(uint)));
        return magic == MdxMagic.Mdlx
            ? M2ChunkedModelReader.Read(memoryStream, sourcePath, companionReader)
            : M2ModelReader.Read(memoryStream, sourcePath);
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
