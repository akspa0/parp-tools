using System;
using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Mcal;

internal static class McalAlphaEncoder
{
    public static byte[] Encode(ReadOnlySpan<byte> columnMajor, uint flags, bool assumeEdgeFixed)
    {
        if (columnMajor.Length != McalAlphaDecoder.AlphaSize)
        {
            throw new ArgumentException($"Expected {McalAlphaDecoder.AlphaSize} bytes for column-major alpha.", nameof(columnMajor));
        }

        // Convert to row-major order, undoing any edge fix if requested
        byte[] rowMajor = ColumnToRowMajor(columnMajor);
        if (!assumeEdgeFixed)
        {
            RemoveEdgeFix(rowMajor);
        }

        if ((flags & 0x200) != 0)
        {
            return Compress(rowMajor);
        }

        if ((flags & 0x100) != 0)
        {
            return EncodeBig(rowMajor);
        }

        return PackFourBit(rowMajor);
    }

    internal static byte[] ColumnToRowMajor(ReadOnlySpan<byte> columnMajor)
    {
        var rowMajor = new byte[McalAlphaDecoder.AlphaSize];

        for (int row = 0; row < McalAlphaDecoder.AlphaSide; row++)
        {
            for (int col = 0; col < McalAlphaDecoder.AlphaSide; col++)
            {
                rowMajor[row * McalAlphaDecoder.AlphaSide + col] = columnMajor[col * McalAlphaDecoder.AlphaSide + row];
            }
        }

        return rowMajor;
    }

    private static void RemoveEdgeFix(Span<byte> rowMajor)
    {
        int last = McalAlphaDecoder.AlphaSide - 1;
        for (int i = 0; i < McalAlphaDecoder.AlphaSide; i++)
        {
            rowMajor[i * McalAlphaDecoder.AlphaSide + last] = rowMajor[i * McalAlphaDecoder.AlphaSide + last - 1];
            rowMajor[last * McalAlphaDecoder.AlphaSide + i] = rowMajor[(last - 1) * McalAlphaDecoder.AlphaSide + i];
        }
        rowMajor[last * McalAlphaDecoder.AlphaSide + last] = rowMajor[(last - 1) * McalAlphaDecoder.AlphaSide + (last - 1)];
    }

    private static byte[] EncodeBig(ReadOnlySpan<byte> rowMajor)
    {
        var result = new byte[rowMajor.Length];
        rowMajor.CopyTo(result);
        return result;
    }

    private static byte[] PackFourBit(ReadOnlySpan<byte> rowMajor)
    {
        int outputLength = rowMajor.Length / 2;
        var packed = new byte[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            byte a = rowMajor[i * 2 + 0];
            byte b = rowMajor[i * 2 + 1];
            byte lowNibble = QuantizeTo4Bit(a);
            byte highNibble = QuantizeTo4Bit(b);
            packed[i] = (byte)((highNibble << 4) | lowNibble);
        }

        return packed;
    }

    private static byte[] Compress(ReadOnlySpan<byte> rowMajor)
    {
        var bytes = new List<byte>(rowMajor.Length);
        int cursor = 0;

        while (cursor < rowMajor.Length)
        {
            int fillCount = CountFill(rowMajor, cursor);
            if (fillCount >= 2)
            {
                bytes.Add((byte)(0x80 | fillCount));
                bytes.Add(rowMajor[cursor]);
                cursor += fillCount;
                continue;
            }

            int copyCount = CountCopy(rowMajor, cursor);
            bytes.Add((byte)copyCount);
            for (int i = 0; i < copyCount; i++)
            {
                bytes.Add(rowMajor[cursor + i]);
            }
            cursor += copyCount;
        }

        return bytes.ToArray();
    }

    private static int CountFill(ReadOnlySpan<byte> data, int start)
    {
        int maxCount = Math.Min(0x7F, data.Length - start);
        if (maxCount <= 0)
        {
            return 0;
        }

        byte value = data[start];
        int count = 1;
        while (count < maxCount && data[start + count] == value)
        {
            count++;
        }

        return count >= 2 ? count : 0;
    }

    private static int CountCopy(ReadOnlySpan<byte> data, int start)
    {
        int maxCount = Math.Min(0x7F, data.Length - start);
        int count = 1;

        while (count < maxCount)
        {
            int lookahead = CountFill(data, start + count);
            if (lookahead >= 2)
            {
                break;
            }
            count++;
        }

        return count;
    }

    private static byte QuantizeTo4Bit(byte value)
    {
        // Map 0-255 to 0-15 with rounding
        int nibble = (value + 8) / 17;
        if (nibble < 0) nibble = 0;
        if (nibble > 15) nibble = 15;
        return (byte)nibble;
    }
}
