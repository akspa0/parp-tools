using System.Buffers.Binary;
using BitMiracle.LibJpeg.Classic;

namespace WowViewer.Core.IO.Wc3;

/// <summary>Opt-in beta BLP0 adapter. Never changes or rewrites source assets.</summary>
public static class Wc3Blp0Decoder
{
    private const int MaxBytes = 64 * 1024 * 1024;

    public static bool IsMipPath(string path)
    {
        string ext = Path.GetExtension(path);
        return ext.Length == 4 && ext[1] is 'b' or 'B' && char.IsAsciiDigit(ext[2]) && char.IsAsciiDigit(ext[3]);
    }

    public static Wc3AssetPreview Read(byte[] bytes, string path, Func<string, byte[]?>? readCompanion)
    {
        if (readCompanion == null)
            return new(0, 0, null, null, "BLP0 external-mipmap preview needs the selected source to resolve its .blp/.b00 companions.", "BLP0");
        int level = IsMipPath(path) ? int.Parse(Path.GetExtension(path).AsSpan(2)) : 0;
        if (level > 15) throw new InvalidDataException("BLP mip level must be 00 through 15.");
        string headerPath = Path.ChangeExtension(path, ".blp");
        byte[] header = IsMipPath(path)
            ? readCompanion(headerPath) ?? throw new FileNotFoundException($"Missing BLP0 header: {headerPath}")
            : bytes;
        if (header.Length < 28 || !header.AsSpan(0, 4).SequenceEqual("BLP0"u8))
            throw new InvalidDataException("The external mip requires a 28-byte BLP0 header.");
        int fullWidth = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(12)));
        int fullHeight = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(16)));
        CheckSize(fullWidth, fullHeight);
        int width = Math.Max(1, fullWidth >> level), height = Math.Max(1, fullHeight >> level);
        string mipPath = Path.ChangeExtension(path, $".b{level:D2}");
        byte[] mip = IsMipPath(path) ? bytes : readCompanion(mipPath)
            ?? throw new FileNotFoundException($"Missing BLP0 mip companion: {mipPath}");
        if (header.Length > MaxBytes || mip.Length > MaxBytes) throw new InvalidDataException("BLP0 input exceeds the 64 MiB preview limit.");
        uint compression = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4));
        uint alphaBits = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(8));
        byte[] rgba;
        if (compression == 0)
        {
            if (header.Length < 32) throw new InvalidDataException("BLP0 JPEG header is truncated.");
            int sharedLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(28)));
            if (sharedLength < 0 || sharedLength > header.Length - 32 || (long)sharedLength + mip.Length > MaxBytes)
                throw new InvalidDataException("BLP0 shared JPEG table length is invalid.");
            using MemoryStream jpeg = new(sharedLength + mip.Length);
            jpeg.Write(header, 32, sharedLength);
            jpeg.Write(mip);
            jpeg.Position = 0;
            rgba = DecodeJpeg(jpeg, width, height, alphaBits != 0);
        }
        else if (compression == 1)
        {
            int pixels = checked(width * height);
            if (alphaBits is not (0 or 1 or 4 or 8) || header.Length < 28 + 1024 ||
                mip.Length < pixels + ((long)pixels * alphaBits + 7) / 8)
                throw new InvalidDataException("BLP0 palette/mip/alpha data is truncated or unsupported.");
            rgba = new byte[pixels * 4];
            for (int i = 0; i < pixels; i++)
            {
                int p = 28 + mip[i] * 4;
                rgba[i * 4] = header[p + 2];
                rgba[i * 4 + 1] = header[p + 1];
                rgba[i * 4 + 2] = header[p];
                rgba[i * 4 + 3] = alphaBits switch
                {
                    0 => 255,
                    1 => (byte)(((mip[pixels + i / 8] >> (i % 8)) & 1) * 255),
                    4 => (byte)(((mip[pixels + i / 2] >> ((i % 2) * 4)) & 15) * 17),
                    _ => mip[pixels + i]
                };
            }
        }
        else throw new NotSupportedException($"BLP0 compression {compression} is not supported.");
        return new(width, height, rgba, null, null, $"BLP0 / external mip {level} / {(compression == 0 ? "raw JPEG components" : "palette")}");
    }

    private static byte[] DecodeJpeg(Stream stream, int width, int height, bool hasAlpha)
    {
        // This libjpeg port owns only managed buffers; the caller owns the stream.
        jpeg_decompress_struct decoder = new();
        decoder.jpeg_stdio_src(stream);
        decoder.jpeg_read_header(true);
        if (decoder.Image_width != width || decoder.Image_height != height || decoder.Num_components is not (3 or 4))
            throw new InvalidDataException("BLP0 JPEG dimensions/components disagree with its header.");
        bool rawFour = decoder.Num_components == 4;
        // Four components in Blizzard's stream are BGRA, NOT CMYK. Ask libjpeg
        // for unchanged components, then map them ourselves (no CMYK/RGB conversion).
        decoder.Out_color_space = rawFour ? decoder.Jpeg_color_space : J_COLOR_SPACE.JCS_RGB;
        decoder.jpeg_start_decompress();
        byte[] rgba = new byte[checked(width * height * 4)];
        byte[][] row = [new byte[checked(width * decoder.Output_components)]];
        while (decoder.Output_scanline < height)
        {
            int y = decoder.Output_scanline;
            if (decoder.jpeg_read_scanlines(row, 1) != 1) throw new InvalidDataException("BLP0 JPEG scanline is incomplete.");
            for (int x = 0; x < width; x++)
            {
                int s = x * decoder.Output_components, d = (y * width + x) * 4;
                rgba[d] = row[0][s + (rawFour ? 2 : 0)];
                rgba[d + 1] = row[0][s + 1];
                rgba[d + 2] = row[0][s + (rawFour ? 0 : 2)];
                rgba[d + 3] = rawFour && hasAlpha ? row[0][s + 3] : (byte)255;
            }
        }
        decoder.jpeg_finish_decompress();
        return rgba;
    }

    private static void CheckSize(int width, int height)
    {
        if (width <= 0 || height <= 0 || width > 8192 || height > 8192 || (long)width * height > 4 * 1024 * 1024)
            throw new InvalidDataException("BLP0 dimensions exceed preview bounds.");
    }

}
