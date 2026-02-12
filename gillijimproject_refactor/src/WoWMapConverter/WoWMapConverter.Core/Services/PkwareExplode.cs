using System;
using System.IO;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// PKWARE DCL (Data Compression Library) "implode" decompression.
/// Used in MPQ archives for small files. This implements the "explode" (decompress) side.
/// Based on the PKWARE DCL specification and StormLib's explode.c.
/// </summary>
public static class PkwareExplode
{
    // Compression types
    private const int CMP_BINARY = 0;
    private const int CMP_ASCII = 1;

    // Length codes table (for literal/length decoding)
    private static readonly byte[] LenBits = {
        3, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7
    };
    private static readonly byte[] LenCode = {
        5, 3, 1, 6, 10, 2, 12, 20, 4, 24, 8, 48, 16, 32, 64, 0
    };

    // Extra bits for length
    private static readonly byte[] ExLenBits = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8
    };
    private static readonly ushort[] LenBase = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 22, 38, 70, 134, 262
    };

    // Distance codes
    private static readonly byte[] DistBits = {
        2, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
    };
    private static readonly byte[] DistCode = {
        0x03, 0x0D, 0x05, 0x19, 0x09, 0x11, 0x01, 0x3E,
        0x1E, 0x2E, 0x0E, 0x36, 0x16, 0x26, 0x06, 0x3A,
        0x1A, 0x2A, 0x0A, 0x32, 0x12, 0x22, 0x02, 0x7C,
        0x3C, 0x5C, 0x1C, 0x6C, 0x2C, 0x4C, 0x0C, 0x74,
        0x34, 0x54, 0x14, 0x64, 0x24, 0x44, 0x04, 0x78,
        0x38, 0x58, 0x18, 0x68, 0x28, 0x48, 0x08, 0xF0,
        0x70, 0xB0, 0x30, 0xD0, 0x50, 0x90, 0x10, 0xE0,
        0x60, 0xA0, 0x20, 0xC0, 0x40, 0x80, 0x00, 0x00
    };

    // ASCII decode tables
    private static readonly byte[] ChBitsAsc = {
        0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x08, 0x07, 0x0C, 0x0C, 0x07, 0x0C, 0x0C,
        0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0D, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
        0x04, 0x0A, 0x08, 0x0C, 0x0A, 0x0C, 0x0A, 0x08, 0x07, 0x07, 0x08, 0x09, 0x07, 0x06, 0x07, 0x08,
        0x07, 0x06, 0x07, 0x07, 0x07, 0x07, 0x08, 0x07, 0x07, 0x08, 0x08, 0x0C, 0x0B, 0x07, 0x09, 0x0B,
        0x0C, 0x06, 0x07, 0x06, 0x06, 0x05, 0x07, 0x08, 0x08, 0x06, 0x0B, 0x09, 0x06, 0x07, 0x06, 0x06,
        0x07, 0x0B, 0x06, 0x06, 0x06, 0x07, 0x09, 0x08, 0x09, 0x09, 0x0B, 0x08, 0x0B, 0x09, 0x0C, 0x08,
        0x0C, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x0B, 0x07, 0x05, 0x06, 0x05, 0x05,
        0x06, 0x0A, 0x05, 0x05, 0x05, 0x05, 0x08, 0x07, 0x08, 0x08, 0x0A, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C,
        0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
        0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
        0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
        0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
        0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
        0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
        0x0D, 0x0C, 0x0D, 0x0D, 0x0D, 0x0C, 0x0D, 0x0D, 0x0D, 0x0C, 0x0D, 0x0D, 0x0D, 0x0D, 0x0C, 0x0D,
        0x0D, 0x0D, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D
    };
    private static readonly ushort[] ChCodeAsc = {
        0x0490, 0x0FE0, 0x07E0, 0x0BE0, 0x03E0, 0x0DE0, 0x05E0, 0x09E0, 0x01E0, 0x00B8, 0x0062, 0x0EE0, 0x06E0, 0x0022, 0x0AE0, 0x02E0,
        0x0CE0, 0x04E0, 0x08E0, 0x00E0, 0x0F60, 0x0760, 0x0B60, 0x0360, 0x0D60, 0x0560, 0x1DE0, 0x0960, 0x0160, 0x0E60, 0x0660, 0x0A60,
        0x000F, 0x0250, 0x0038, 0x0260, 0x0050, 0x0C60, 0x0390, 0x00D8, 0x0042, 0x0002, 0x0058, 0x01B0, 0x007C, 0x0029, 0x003C, 0x0098,
        0x005C, 0x0009, 0x001C, 0x006C, 0x002C, 0x004C, 0x0018, 0x000C, 0x0074, 0x00E8, 0x0068, 0x0460, 0x0090, 0x0034, 0x00B0, 0x0710,
        0x0860, 0x0031, 0x0054, 0x0011, 0x0021, 0x0017, 0x0014, 0x00A8, 0x0028, 0x0001, 0x0310, 0x0130, 0x003E, 0x0064, 0x001E, 0x002E,
        0x0024, 0x0510, 0x000E, 0x0036, 0x0016, 0x0044, 0x0030, 0x00C8, 0x01D0, 0x00D0, 0x0110, 0x0048, 0x0610, 0x0150, 0x0060, 0x0088,
        0x0FA0, 0x0007, 0x0026, 0x0006, 0x003A, 0x001B, 0x001A, 0x002A, 0x000A, 0x000B, 0x0210, 0x0004, 0x0013, 0x0032, 0x0003, 0x001D,
        0x0012, 0x0190, 0x000D, 0x0015, 0x0005, 0x0019, 0x0008, 0x0078, 0x00F0, 0x0070, 0x0290, 0x0410, 0x0010, 0x07A0, 0x0BA0, 0x03A0,
        0x0240, 0x1C40, 0x0C40, 0x1440, 0x0440, 0x1840, 0x0840, 0x1040, 0x0040, 0x1F80, 0x0F80, 0x1780, 0x0780, 0x1B80, 0x0B80, 0x1380,
        0x0380, 0x1D80, 0x0D80, 0x1580, 0x0580, 0x1980, 0x0980, 0x1180, 0x0180, 0x1E80, 0x0E80, 0x1680, 0x0680, 0x1A80, 0x0A80, 0x1280,
        0x0280, 0x1C80, 0x0C80, 0x1480, 0x0480, 0x1880, 0x0880, 0x1080, 0x0080, 0x1F00, 0x0F00, 0x1700, 0x0700, 0x1B00, 0x0B00, 0x1300,
        0x0DA0, 0x05A0, 0x09A0, 0x01A0, 0x0EA0, 0x06A0, 0x0AA0, 0x02A0, 0x0CA0, 0x04A0, 0x08A0, 0x00A0, 0x0F20, 0x0720, 0x0B20, 0x0320,
        0x0D20, 0x0520, 0x0920, 0x0120, 0x0E20, 0x0620, 0x0A20, 0x0220, 0x0C20, 0x0420, 0x0820, 0x0020, 0x0FC0, 0x07C0, 0x0BC0, 0x03C0,
        0x0DC0, 0x05C0, 0x09C0, 0x01C0, 0x0EC0, 0x06C0, 0x0AC0, 0x02C0, 0x0CC0, 0x04C0, 0x08C0, 0x00C0, 0x0F40, 0x0740, 0x0B40, 0x0340,
        0x0300, 0x0D40, 0x1D00, 0x0D00, 0x1500, 0x0540, 0x0500, 0x1900, 0x0900, 0x0940, 0x1100, 0x0100, 0x1E00, 0x0E00, 0x0140, 0x1600,
        0x0600, 0x1A00, 0x0E40, 0x0640, 0x0A40, 0x0A00, 0x1200, 0x0200, 0x1C00, 0x0C00, 0x1400, 0x0400, 0x1800, 0x0800, 0x1000, 0x0000
    };

    /// <summary>
    /// Decompress PKWARE DCL imploded data.
    /// </summary>
    /// <param name="input">Compressed data (after the MPQ compression type byte)</param>
    /// <param name="expectedSize">Expected decompressed size</param>
    /// <returns>Decompressed data, or null on failure</returns>
    public static byte[]? Decompress(byte[] input, uint expectedSize)
    {
        if (input.Length < 2)
            return null;

        try
        {
            int compType = input[0]; // 0=binary, 1=ascii
            int dictSizeBits = input[1]; // 4, 5, or 6
            int dictSize = 64 << dictSizeBits;

            if (compType != CMP_BINARY && compType != CMP_ASCII)
            {
                Console.WriteLine($"[PkwareExplode] Unknown compression type: {compType}");
                return null;
            }

            if (dictSizeBits < 4 || dictSizeBits > 6)
            {
                Console.WriteLine($"[PkwareExplode] Invalid dict size bits: {dictSizeBits}");
                return null;
            }

            var output = new byte[expectedSize];
            int outPos = 0;

            // Bit reader state
            int inPos = 2; // Skip comp type and dict size bytes
            int bitBuf = 0;
            int bitCount = 0;

            // Build decode tables based on compression type
            byte[] chBits;
            ushort[] chCode;
            if (compType == CMP_ASCII)
            {
                chBits = ChBitsAsc;
                chCode = ChCodeAsc;
            }
            else
            {
                // Binary mode: literal bytes are 8 bits each
                chBits = new byte[256];
                chCode = new ushort[256];
                for (int i = 0; i < 256; i++)
                {
                    chBits[i] = 8;
                    chCode[i] = (ushort)i;
                }
            }

            while (outPos < expectedSize)
            {
                // Ensure we have bits
                while (bitCount < 16 && inPos < input.Length)
                {
                    bitBuf |= input[inPos++] << bitCount;
                    bitCount += 8;
                }

                if (bitCount == 0)
                    break;

                // Read flag bit: 0 = literal, 1 = match
                int flagBit = bitBuf & 1;
                bitBuf >>= 1;
                bitCount--;

                if (flagBit == 0)
                {
                    // Literal byte
                    if (compType == CMP_BINARY)
                    {
                        // Binary: read 8 bits directly
                        while (bitCount < 8 && inPos < input.Length)
                        {
                            bitBuf |= input[inPos++] << bitCount;
                            bitCount += 8;
                        }
                        if (bitCount < 8) break;

                        output[outPos++] = (byte)(bitBuf & 0xFF);
                        bitBuf >>= 8;
                        bitCount -= 8;
                    }
                    else
                    {
                        // ASCII: decode using Huffman table
                        int decoded = DecodeLiteral(ref bitBuf, ref bitCount, ref inPos, input, chBits, chCode);
                        if (decoded < 0) break;
                        output[outPos++] = (byte)decoded;
                    }
                }
                else
                {
                    // Length-distance pair (match)

                    // Decode length
                    while (bitCount < 16 && inPos < input.Length)
                    {
                        bitBuf |= input[inPos++] << bitCount;
                        bitCount += 8;
                    }

                    int lenIndex = DecodeValue(ref bitBuf, ref bitCount, ref inPos, input, LenBits, LenCode, 16);
                    if (lenIndex < 0) break;

                    int extraBits = ExLenBits[lenIndex];
                    int length = LenBase[lenIndex] + 2;

                    if (extraBits > 0)
                    {
                        while (bitCount < extraBits && inPos < input.Length)
                        {
                            bitBuf |= input[inPos++] << bitCount;
                            bitCount += 8;
                        }
                        if (bitCount < extraBits) break;

                        length += bitBuf & ((1 << extraBits) - 1);
                        bitBuf >>= extraBits;
                        bitCount -= extraBits;
                    }

                    // Decode distance
                    while (bitCount < 16 && inPos < input.Length)
                    {
                        bitBuf |= input[inPos++] << bitCount;
                        bitCount += 8;
                    }

                    int distIndex = DecodeValue(ref bitBuf, ref bitCount, ref inPos, input, DistBits, DistCode, 64);
                    if (distIndex < 0) break;

                    int distance = (distIndex << dictSizeBits);

                    // Read low bits of distance
                    while (bitCount < dictSizeBits && inPos < input.Length)
                    {
                        bitBuf |= input[inPos++] << bitCount;
                        bitCount += 8;
                    }
                    if (bitCount < dictSizeBits) break;

                    distance |= bitBuf & ((1 << dictSizeBits) - 1);
                    bitBuf >>= dictSizeBits;
                    bitCount -= dictSizeBits;

                    distance += 1;

                    // Copy from output buffer
                    int srcPos = outPos - distance;
                    if (srcPos < 0)
                    {
                        Console.WriteLine($"[PkwareExplode] Invalid back reference: srcPos={srcPos}, outPos={outPos}, distance={distance}");
                        break;
                    }

                    for (int i = 0; i < length && outPos < expectedSize; i++)
                    {
                        output[outPos++] = output[srcPos + i];
                    }
                }
            }

            if (outPos == expectedSize)
                return output;

            // Partial decompression — return what we got if substantial
            if (outPos > 0)
            {
                Console.WriteLine($"[PkwareExplode] Partial decompression: {outPos}/{expectedSize} bytes");
                var partial = new byte[outPos];
                Array.Copy(output, partial, outPos);
                return partial;
            }

            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PkwareExplode] Exception: {ex.Message}");
            return null;
        }
    }

    private static int DecodeLiteral(ref int bitBuf, ref int bitCount, ref int inPos, byte[] input,
        byte[] chBits, ushort[] chCode)
    {
        for (int i = 0; i < 256; i++)
        {
            int bits = chBits[i];
            if (bits == 0) continue;

            while (bitCount < bits && inPos < input.Length)
            {
                bitBuf |= input[inPos++] << bitCount;
                bitCount += 8;
            }
            if (bitCount < bits) continue;

            int mask = (1 << bits) - 1;
            if ((bitBuf & mask) == chCode[i])
            {
                bitBuf >>= bits;
                bitCount -= bits;
                return i;
            }
        }
        return -1;
    }

    private static int DecodeValue(ref int bitBuf, ref int bitCount, ref int inPos, byte[] input,
        byte[] codeBits, byte[] codeValues, int tableSize)
    {
        for (int i = 0; i < tableSize; i++)
        {
            int bits = codeBits[i];
            if (bits == 0) continue;

            while (bitCount < bits && inPos < input.Length)
            {
                bitBuf |= input[inPos++] << bitCount;
                bitCount += 8;
            }
            if (bitCount < bits) continue;

            int mask = (1 << bits) - 1;
            if ((bitBuf & mask) == codeValues[i])
            {
                bitBuf >>= bits;
                bitCount -= bits;
                return i;
            }
        }
        return -1;
    }
}
