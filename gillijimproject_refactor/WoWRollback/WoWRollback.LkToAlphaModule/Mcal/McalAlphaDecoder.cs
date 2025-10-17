using System;

namespace WoWRollback.LkToAlphaModule.Mcal;

internal static class McalAlphaDecoder
{
    internal const int AlphaSize = 64 * 64;
    internal const int AlphaSide = 64;

    public static byte[] DecodeToColumnMajor(ReadOnlySpan<byte> source, uint flags, bool applyEdgeFix)
    {
        var rowMajor = DecodeRowMajor(source, flags);
        var columnMajor = RowToColumnMajor(rowMajor);
        if (applyEdgeFix)
        {
            ApplyEdgeFix(columnMajor);
        }
        return columnMajor;
    }

    public static byte[] DecodeRowMajor(ReadOnlySpan<byte> source, uint flags)
    {
        if ((flags & 0x200) != 0)
        {
            return Decompress(source);
        }

        if ((flags & 0x100) != 0)
        {
            return CopyBig(source);
        }

        return ExpandFourBit(source);
    }

    private static byte[] Decompress(ReadOnlySpan<byte> source)
    {
        var result = new byte[AlphaSize];
        int dst = 0;
        int cursor = 0;

        while (cursor < source.Length && dst < result.Length)
        {
            byte descriptor = source[cursor++];
            int count = descriptor & 0x7F;
            bool isFill = (descriptor & 0x80) != 0;

            if (count == 0)
            {
                continue;
            }

            if (isFill)
            {
                if (cursor >= source.Length) break;
                byte value = source[cursor++];
                for (int i = 0; i < count && dst < result.Length; i++)
                {
                    result[dst++] = value;
                }
            }
            else
            {
                for (int i = 0; i < count && dst < result.Length && cursor < source.Length; i++)
                {
                    result[dst++] = source[cursor++];
                }
            }
        }

        return result;
    }

    private static byte[] CopyBig(ReadOnlySpan<byte> source)
    {
        var result = new byte[AlphaSize];
        int copyLength = Math.Min(result.Length, source.Length);
        source.Slice(0, copyLength).CopyTo(result);
        return result;
    }

    private static byte[] ExpandFourBit(ReadOnlySpan<byte> source)
    {
        var result = new byte[AlphaSize];
        int dst = 0;

        for (int i = 0; i < source.Length && dst < result.Length; i++)
        {
            byte packed = source[i];
            byte lower = (byte)(packed & 0x0F);
            byte upper = (byte)((packed >> 4) & 0x0F);

            result[dst++] = (byte)(lower | (lower << 4));
            if (dst < result.Length)
            {
                result[dst++] = (byte)(upper | (upper << 4));
            }
        }

        return result;
    }

    internal static byte[] RowToColumnMajor(byte[] rowMajor)
    {
        var dst = new byte[AlphaSize];

        for (int row = 0; row < AlphaSide; row++)
        {
            for (int col = 0; col < AlphaSide; col++)
            {
                dst[col * AlphaSide + row] = rowMajor[row * AlphaSide + col];
            }
        }

        return dst;
    }

    internal static void ApplyEdgeFix(Span<byte> data)
    {
        for (int i = 0; i < AlphaSide; i++)
        {
            data[i * AlphaSide + (AlphaSide - 1)] = data[i * AlphaSide + (AlphaSide - 2)];
            data[(AlphaSide - 1) * AlphaSide + i] = data[(AlphaSide - 2) * AlphaSide + i];
        }
        data[(AlphaSide - 1) * AlphaSide + (AlphaSide - 1)] = data[(AlphaSide - 2) * AlphaSide + (AlphaSide - 2)];
    }
}
