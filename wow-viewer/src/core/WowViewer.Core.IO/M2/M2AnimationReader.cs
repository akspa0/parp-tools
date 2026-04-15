using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2AnimationReader
{
    public static M2ExternalAnimationDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2ExternalAnimationDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("M2 external animation reading requires a seekable stream.", nameof(stream));

        if (!Path.GetExtension(sourcePath).Equals(".anim", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"M2 external animation reading requires a .anim path. Found '{Path.GetExtension(sourcePath)}'.", nameof(sourcePath));

        byte[] bytes = ReadAllBytes(stream);
        if (bytes.Length == 0)
            throw new InvalidDataException($"Animation file '{sourcePath}' is empty.");

        if (bytes.Length >= 8 && Encoding.ASCII.GetString(bytes, 0, 4) == "AFM2")
        {
            uint payloadSize = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4));
            if (payloadSize > bytes.Length - 8)
            {
                throw new InvalidDataException(
                    $"Animation file '{sourcePath}' has an invalid AFM2 payload size 0x{payloadSize:X} for file length 0x{bytes.Length:X}.");
            }

            byte[] payload = bytes.AsSpan(8, checked((int)payloadSize)).ToArray();
            return new M2ExternalAnimationDocument(sourcePath, payload, isChunkedContainer: true, containerSignature: "AFM2");
        }

        return new M2ExternalAnimationDocument(sourcePath, bytes, isChunkedContainer: false, containerSignature: null);
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
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
