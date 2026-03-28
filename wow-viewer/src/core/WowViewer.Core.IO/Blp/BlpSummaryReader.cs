using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Blp;

namespace WowViewer.Core.IO.Blp;

public static class BlpSummaryReader
{
    private const int Blp0And1HeaderSizeBytes = 156;
    private const int Blp2HeaderSizeBytes = 148;
    private const int PaletteSizeBytes = 1024;
    private const int MaxMipLevelCount = 16;

    public static BlpSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static BlpSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("BLP summary reading requires a seekable stream.", nameof(stream));

        if (stream.Length < 4)
            throw new InvalidDataException($"BLP file '{sourcePath}' is too small to contain a signature.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[4];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            BlpFormat format = ParseFormat(signature, sourcePath);
            int headerSizeBytes = format == BlpFormat.Blp2 ? Blp2HeaderSizeBytes : Blp0And1HeaderSizeBytes;
            if (stream.Length < headerSizeBytes)
                throw new InvalidDataException($"BLP file '{sourcePath}' is too small to contain a full {signature} header.");

            byte[] header = new byte[headerSizeBytes];
            stream.Position = 0;
            stream.ReadExactly(header);

            uint? version;
            uint compressionRaw;
            byte alphaDepthBits;
            uint pixelFormatRaw;
            uint mipMapTypeRaw;
            int width;
            int height;

            if (format == BlpFormat.Blp2)
            {
                version = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4, 4));
                compressionRaw = header[8];
                alphaDepthBits = header[9];
                pixelFormatRaw = header[10];
                mipMapTypeRaw = header[11];
                width = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(12, 4)));
                height = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(16, 4)));
            }
            else
            {
                version = null;
                compressionRaw = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4, 4));
                alphaDepthBits = checked((byte)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(8, 4)));
                width = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(12, 4)));
                height = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(16, 4)));
                pixelFormatRaw = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(20, 4));
                mipMapTypeRaw = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(24, 4));
            }

            if (width <= 0 || height <= 0)
                throw new InvalidDataException($"BLP file '{sourcePath}' reported invalid dimensions {width}x{height}.");

            List<BlpMipMapEntry> mipMaps = [];
            int inBoundsMipLevelCount = 0;
            int outOfBoundsMipLevelCount = 0;
            long maxMipEndOffset = 0;
            int offsetsBase = format == BlpFormat.Blp2 ? 20 : 28;
            int sizesBase = offsetsBase + (MaxMipLevelCount * sizeof(uint));

            for (int level = 0; level < MaxMipLevelCount; level++)
            {
                uint offset = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(offsetsBase + (level * sizeof(uint)), sizeof(uint)));
                uint sizeBytes = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(sizesBase + (level * sizeof(uint)), sizeof(uint)));
                if (offset == 0 || sizeBytes == 0)
                    continue;

                long mipEndOffset = offset + (long)sizeBytes;
                bool isInBounds = offset < stream.Length && mipEndOffset <= stream.Length;
                if (isInBounds)
                    inBoundsMipLevelCount++;
                else
                    outOfBoundsMipLevelCount++;

                maxMipEndOffset = Math.Max(maxMipEndOffset, mipEndOffset);
                mipMaps.Add(new BlpMipMapEntry(level, GetMipDimension(width, level), GetMipDimension(height, level), offset, sizeBytes, isInBounds));
            }

            BlpCompressionType compression = (BlpCompressionType)compressionRaw;
            BlpPixelFormat pixelFormat = (BlpPixelFormat)pixelFormatRaw;
            int paletteSizeBytes = compression == BlpCompressionType.Palettized && stream.Length >= headerSizeBytes + PaletteSizeBytes
                ? PaletteSizeBytes
                : 0;
            int jpegHeaderSizeBytes = 0;
            if (compression == BlpCompressionType.Jpeg && stream.Length >= headerSizeBytes + sizeof(uint))
            {
                stream.Position = headerSizeBytes;
                Span<byte> jpegSizeBytes = stackalloc byte[4];
                stream.ReadExactly(jpegSizeBytes);
                jpegHeaderSizeBytes = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(jpegSizeBytes));
            }

            return new BlpSummary(
                sourcePath,
                format,
                signature,
                version,
                compression,
                alphaDepthBits,
                pixelFormat,
                mipMapTypeRaw,
                width,
                height,
                headerSizeBytes,
                paletteSizeBytes,
                jpegHeaderSizeBytes,
                mipMaps,
                inBoundsMipLevelCount,
                outOfBoundsMipLevelCount,
                maxMipEndOffset);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static BlpFormat ParseFormat(string signature, string sourcePath)
    {
        return signature switch
        {
            "BLP0" => BlpFormat.Blp0,
            "BLP1" => BlpFormat.Blp1,
            "BLP2" => BlpFormat.Blp2,
            _ => throw new InvalidDataException($"File '{sourcePath}' does not contain a supported BLP signature. Found '{signature}'."),
        };
    }

    private static int GetMipDimension(int baseDimension, int level)
    {
        int dimension = baseDimension;
        for (int index = 0; index < level && dimension > 1; index++)
            dimension >>= 1;

        return Math.Max(1, dimension);
    }
}