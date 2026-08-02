using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Blp;

/// <summary>
/// Reproduces the authored minimap tile's DXT1 encode/decode cycle on a synthetic tile, and verifies
/// our encoder against authored bytes.
///
/// WHY THIS EXISTS: authored 0.5.3 minimaps are DXT1-compressed (BLP2/DXTC/DXT1, 256x256, one mip,
/// no alpha). Our synthesizer produces pristine 24-bit output, so every comparison has been scoring a
/// clean image against a lossy one — a codec confound. This class applies the same lossy cycle to a
/// synthetic tile so authored and synthetic compare on equal terms (FR-002, FR-015), and provides a
/// round-trip check that confirms our encoder reproduces the authored degradation class (FR-014).
///
/// IMPLEMENTATION NOTE: DXT1 is a small, fully-specified format — each 4x4 block is two RGB565
/// endpoint colours plus 2-bit per-pixel indices. The encode and decode halves are implemented here
/// in pure C# with NO external codec dependency, so this is compatible with the project's .NET 10
/// toolchain. The decode half is deterministic (any correct decoder yields identical pixels); the
/// encode half is a lossy fitting problem, and the round-trip check (FR-014) is what validates our
/// encoder against authored bytes.
///
/// IN-MEMORY ONLY: this never writes a BLP container. It only needs the encode/decode cycle in memory
/// to reproduce degradation.
/// </summary>
public static class Dxt1TileCodec
{
    /// <summary>
    /// Encode a tile to DXT1 raw blocks then decode back to RGBA — the parity cycle (FR-002, FR-015).
    /// The result carries the same block-banding degradation class as an authored DXT1 tile.
    /// </summary>
    public static Image<Rgba32> EncodeDecode(Image<Rgba32> source)
    {
        ArgumentNullException.ThrowIfNull(source);

        int width = source.Width;
        int height = source.Height;
        byte[] rgba = new byte[width * height * 4];
        source.CopyPixelDataTo(rgba);

        byte[] compressed = EncodeDxt1(rgba, width, height);
        byte[] decoded = DecodeDxt1(compressed, width, height);

        var output = new Image<Rgba32>(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = ((y * width) + x) * 4;
                output[x, y] = new Rgba32(decoded[i], decoded[i + 1], decoded[i + 2], decoded[i + 3]);
            }
        }

        return output;
    }

    /// <summary>
    /// Decode an authored BLP tile to RGBA (FR-001). Wraps SereniaBLPLib directly so the raw authored
    /// pixels are available for round-trip comparison.
    /// </summary>
    public static Image<Rgba32> DecodeAuthored(byte[] authoredBlp)
    {
        ArgumentNullException.ThrowIfNull(authoredBlp);

        using var stream = new MemoryStream(authoredBlp, writable: false);
        using var blp = new BlpFile(stream);
        return blp.GetImage(0);
    }

    /// <summary>
    /// Round-trip agreement: decode an authored tile, re-encode it to DXT1, and measure the fraction
    /// of 4x4 blocks whose re-encoded bytes match the authored DXT1 block bytes. A correctness check
    /// on OUR encoder (FR-014), exploiting DXT1's near-idempotency on decoded data. Returns 0..1.
    ///
    /// This is explicitly NOT an attempt to identify which encoder Blizzard used; close enough for the
    /// degradation to match is the bar.
    /// </summary>
    public static float RoundTripAgreement(byte[] authoredBlp)
    {
        ArgumentNullException.ThrowIfNull(authoredBlp);

        byte[]? authoredBlocks = TryExtractDxt1Blocks(authoredBlp);
        if (authoredBlocks is null)
            return 0f;

        using Image<Rgba32> decoded = DecodeAuthored(authoredBlp);
        byte[] rgba = new byte[decoded.Width * decoded.Height * 4];
        decoded.CopyPixelDataTo(rgba);

        byte[] reencoded = EncodeDxt1(rgba, decoded.Width, decoded.Height);

        int blockCount = Math.Min(authoredBlocks.Length, reencoded.Length) / 8;
        if (blockCount <= 0)
            return 0f;

        int matching = 0;
        for (int block = 0; block < blockCount; block++)
        {
            int offset = block * 8;
            bool same = true;
            for (int b = 0; b < 8; b++)
            {
                if (authoredBlocks[offset + b] != reencoded[offset + b])
                {
                    same = false;
                    break;
                }
            }

            if (same)
                matching++;
        }

        return (float)matching / blockCount;
    }

    // ── DXT1 block encode ────────────────────────────────────────────────────
    // Each 4x4 block is 8 bytes: two RGB565 endpoints (4 bytes) + 4 bytes of 2-bit indices.
    // This is a simple but valid encoder: per block it picks the min and max colours as endpoints
    // and assigns each pixel the nearest of the four interpolated colours.

    private static byte[] EncodeDxt1(byte[] rgba, int width, int height)
    {
        int blocksX = Math.Max(1, (width + 3) / 4);
        int blocksY = Math.Max(1, (height + 3) / 4);
        byte[] output = new byte[blocksX * blocksY * 8];

        // Reused per block to avoid repeated allocation; a 4x4 block is at most 16 pixels.
        var pixels = new (byte R, byte G, byte B)[16];

        for (int by = 0; by < blocksY; by++)
        {
            for (int bx = 0; bx < blocksX; bx++)
            {
                // Gather the 4x4 block (clamp at edges).
                int count = 0;
                for (int py = 0; py < 4; py++)
                {
                    for (int px = 0; px < 4; px++)
                    {
                        int x = (bx * 4) + px;
                        int y = (by * 4) + py;
                        if (x >= width || y >= height)
                            continue;
                        int i = ((y * width) + x) * 4;
                        pixels[count++] = (rgba[i], rgba[i + 1], rgba[i + 2]);
                    }
                }

                if (count == 0)
                    continue;

                // Endpoints: min and max of the block (simple bounding-box fit).
                byte minR = 255, minG = 255, minB = 255;
                byte maxR = 0, maxG = 0, maxB = 0;
                for (int p = 0; p < count; p++)
                {
                    minR = Math.Min(minR, pixels[p].R);
                    minG = Math.Min(minG, pixels[p].G);
                    minB = Math.Min(minB, pixels[p].B);
                    maxR = Math.Max(maxR, pixels[p].R);
                    maxG = Math.Max(maxG, pixels[p].G);
                    maxB = Math.Max(maxB, pixels[p].B);
                }

                ushort c0 = Rgb565(maxR, maxG, maxB);
                ushort c1 = Rgb565(minR, minG, minB);

                // 4-colour mode (c0 > c1). If the endpoints collide, force the ordering.
                if (c0 <= c1)
                    (c0, c1) = (c1, c0);

                (byte R, byte G, byte B) e0 = Rgb565ToRgb(c0);
                (byte R, byte G, byte B) e1 = Rgb565ToRgb(c1);
                (byte R, byte G, byte B) e2 = Interp(e0, e1, 2, 1);
                (byte R, byte G, byte B) e3 = Interp(e0, e1, 1, 2);

                uint indices = 0;
                for (int p = 0; p < 16; p++)
                {
                    byte idx;
                    if (p >= count)
                    {
                        idx = 0;
                    }
                    else
                    {
                        idx = NearestIndex(pixels[p], e0, e1, e2, e3);
                    }

                    indices |= (uint)idx << (p * 2);
                }

                int blockOffset = ((by * blocksX) + bx) * 8;
                output[blockOffset + 0] = (byte)(c0 & 0xFF);
                output[blockOffset + 1] = (byte)(c0 >> 8);
                output[blockOffset + 2] = (byte)(c1 & 0xFF);
                output[blockOffset + 3] = (byte)(c1 >> 8);
                output[blockOffset + 4] = (byte)(indices & 0xFF);
                output[blockOffset + 5] = (byte)((indices >> 8) & 0xFF);
                output[blockOffset + 6] = (byte)((indices >> 16) & 0xFF);
                output[blockOffset + 7] = (byte)((indices >> 24) & 0xFF);
            }
        }

        return output;
    }

    // ── DXT1 block decode ────────────────────────────────────────────────────

    private static byte[] DecodeDxt1(byte[] blocks, int width, int height)
    {
        int blocksX = Math.Max(1, (width + 3) / 4);
        int blocksY = Math.Max(1, (height + 3) / 4);
        byte[] output = new byte[width * height * 4];

        for (int by = 0; by < blocksY; by++)
        {
            for (int bx = 0; bx < blocksX; bx++)
            {
                int blockOffset = ((by * blocksX) + bx) * 8;
                if (blockOffset + 8 > blocks.Length)
                    continue;

                ushort c0 = (ushort)(blocks[blockOffset] | (blocks[blockOffset + 1] << 8));
                ushort c1 = (ushort)(blocks[blockOffset + 2] | (blocks[blockOffset + 3] << 8));
                uint indices = (uint)(blocks[blockOffset + 4]
                    | (blocks[blockOffset + 5] << 8)
                    | (blocks[blockOffset + 6] << 16)
                    | (blocks[blockOffset + 7] << 24));

                (byte R, byte G, byte B) e0 = Rgb565ToRgb(c0);
                (byte R, byte G, byte B) e1 = Rgb565ToRgb(c1);

                (byte R, byte G, byte B) e2, e3;
                bool transparent = false;
                if (c0 > c1)
                {
                    e2 = Interp(e0, e1, 2, 1);
                    e3 = Interp(e0, e1, 1, 2);
                }
                else
                {
                    e2 = Interp(e0, e1, 1, 1);
                    e3 = (0, 0, 0);
                    transparent = true;
                }

                for (int py = 0; py < 4; py++)
                {
                    for (int px = 0; px < 4; px++)
                    {
                        int x = (bx * 4) + px;
                        int y = (by * 4) + py;
                        if (x >= width || y >= height)
                            continue;

                        int pixelIndex = (py * 4) + px;
                        int idx = (int)((indices >> (pixelIndex * 2)) & 0x3);

                        (byte R, byte G, byte B) colour = idx switch
                        {
                            0 => e0,
                            1 => e1,
                            2 => e2,
                            _ => e3,
                        };

                        int o = ((y * width) + x) * 4;
                        output[o] = colour.R;
                        output[o + 1] = colour.G;
                        output[o + 2] = colour.B;
                        output[o + 3] = transparent && idx == 3 ? (byte)0 : (byte)255;
                    }
                }
            }
        }

        return output;
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static ushort Rgb565(byte r, byte g, byte b) =>
        (ushort)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));

    private static (byte R, byte G, byte B) Rgb565ToRgb(ushort c)
    {
        byte r = (byte)(((c >> 11) & 0x1F) << 3);
        byte g = (byte)(((c >> 5) & 0x3F) << 2);
        byte b = (byte)((c & 0x1F) << 3);
        // Expand to full 8-bit range.
        r = (byte)(r | (r >> 5));
        g = (byte)(g | (g >> 6));
        b = (byte)(b | (b >> 5));
        return (r, g, b);
    }

    private static (byte R, byte G, byte B) Interp(
        (byte R, byte G, byte B) a,
        (byte R, byte G, byte B) b,
        int aWeight,
        int bWeight) => (
        (byte)(((a.R * aWeight) + (b.R * bWeight)) / (aWeight + bWeight)),
        (byte)(((a.G * aWeight) + (b.G * bWeight)) / (aWeight + bWeight)),
        (byte)(((a.B * aWeight) + (b.B * bWeight)) / (aWeight + bWeight)));

    private static byte NearestIndex(
        (byte R, byte G, byte B) pixel,
        (byte R, byte G, byte B) e0,
        (byte R, byte G, byte B) e1,
        (byte R, byte G, byte B) e2,
        (byte R, byte G, byte B) e3)
    {
        int best = 0;
        long bestDist = long.MaxValue;
        (byte R, byte G, byte B)[] palette = [e0, e1, e2, e3];
        for (int i = 0; i < 4; i++)
        {
            long dr = pixel.R - palette[i].R;
            long dg = pixel.G - palette[i].G;
            long db = pixel.B - palette[i].B;
            long dist = (dr * dr) + (dg * dg) + (db * db);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = i;
            }
        }

        return (byte)best;
    }

    /// <summary>
    /// Extract the raw DXT1 block data (mip 0) from a BLP2 container. Returns null when the container
    /// is not a BLP2/DXTC/DXT1 file or the mip data cannot be located.
    /// </summary>
    private static byte[]? TryExtractDxt1Blocks(byte[] blp)
    {
        if (blp.Length < 148)
            return null;

        // BLP2 magic
        if (blp[0] != 'B' || blp[1] != 'L' || blp[2] != 'P' || blp[3] != '2')
            return null;

        // compression byte at offset 6 must be 2 (DXTC)
        if (blp[6] != 2)
            return null;

        // alpha encoding byte at offset 8 must be 7 (DXT1)
        if (blp[8] != 7)
            return null;

        // mip 0 offset at bytes 20..23, size at bytes 84..87
        uint offset = ReadUInt32Le(blp, 20);
        uint size = ReadUInt32Le(blp, 84);

        if (offset == 0 || size == 0 || offset + size > (uint)blp.Length)
            return null;

        byte[] blocks = new byte[size];
        Array.Copy(blp, (int)offset, blocks, 0, (int)size);
        return blocks;
    }

    private static uint ReadUInt32Le(byte[] data, int offset) =>
        (uint)(data[offset]
            | (data[offset + 1] << 8)
            | (data[offset + 2] << 16)
            | (data[offset + 3] << 24));
}
