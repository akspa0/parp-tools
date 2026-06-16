using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxCollisionReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxCollisionFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxCollisionFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX collision reading requires a seekable stream.", nameof(stream));

        if (stream.Length < SignatureSizeBytes)
            throw new InvalidDataException($"MDX file '{sourcePath}' is too small to contain a signature.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            bool isMd20 = string.Equals(signature, "MD20", StringComparison.Ordinal);
            if (!string.Equals(signature, "MDLX", StringComparison.Ordinal) && !isMd20)
                throw new InvalidDataException($"File '{sourcePath}' does not contain an MDLX/MD20 signature. Found '{signature}'.");

            uint? version = null;
            string? modelName = null;
            MdxCollisionMesh? collision = null;

            // For MD20: read fixed-header fields
            if (isMd20)
            {
                stream.Position = 0x04;
                Span<byte> verBytes = stackalloc byte[4];
                stream.ReadExactly(verBytes);
                version = BinaryPrimitives.ReadUInt32LittleEndian(verBytes);

                // Read model name at fixed offset
                stream.Position = 0x08;
                Span<byte> nameCountBytes = stackalloc byte[4];
                stream.ReadExactly(nameCountBytes);
                uint nameCount = BinaryPrimitives.ReadUInt32LittleEndian(nameCountBytes);
                Span<byte> nameOffsetBytes = stackalloc byte[4];
                stream.ReadExactly(nameOffsetBytes);
                uint nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(nameOffsetBytes);
                if (nameCount > 0 && nameOffset + 0x80 <= stream.Length)
                {
                    stream.Position = nameOffset;
                    Span<byte> nameBytes = stackalloc byte[0x80];
                    stream.ReadExactly(nameBytes);
                    int term = nameBytes.IndexOf((byte)0);
                    modelName = Encoding.ASCII.GetString(nameBytes[..(term >= 0 ? term : nameBytes.Length)]);
                }

                // Scan for CLID chunk in the file
                byte[] allBytes = new byte[stream.Length];
                stream.Position = 0;
                stream.ReadExactly(allBytes);
                for (int i = 0; i < allBytes.Length - 8; i++)
                {
                    if (allBytes[i] == (byte)'C' && allBytes[i+1] == (byte)'L' && allBytes[i+2] == (byte)'I' && allBytes[i+3] == (byte)'D')
                    {
                        uint clidSize = BitConverter.ToUInt32(allBytes, i + 4);
                        if (i + 8 + clidSize <= allBytes.Length)
                        {
                            using var clidStream = new MemoryStream(allBytes, i + 8, (int)clidSize);
                            collision = MdxCollisionChunkReader.ReadClassicCollisionMesh(clidStream, 0, clidSize, version);
                        }
                        break;
                    }
                }
                return new MdxCollisionFile(sourcePath, signature, version, modelName, collision);
            }
            // End MD20 handling

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
                else if (header.Id == MdxChunkIds.Clid)
                {
                    collision = MdxCollisionChunkReader.ReadClassicCollisionMesh(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxCollisionFile(sourcePath, signature, version, modelName, collision);
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
            Span<byte> buffer = stackalloc byte[sizeof(uint)];
            stream.ReadExactly(buffer);
            return BinaryPrimitives.ReadUInt32LittleEndian(buffer);
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
            byte[] bytes = new byte[size];
            stream.ReadExactly(bytes);
            int terminatorIndex = Array.IndexOf(bytes, (byte)0);
            int count = terminatorIndex >= 0 ? terminatorIndex : bytes.Length;
            return Encoding.ASCII.GetString(bytes, 0, count);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }
}