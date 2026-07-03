using System.Buffers.Binary;
using System.Text;

namespace WowViewer.Tool.V22Enrich;

/// <summary>
/// Helpers for flattening arrays into byte[] for the enrichment stream.
/// </summary>
static class EnrichmentArrayHelper
{
    /// <summary>Flatten a float[].</summary>
    public static byte[] FlattenFloats(float[] values)
    {
        byte[] result = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(i * 4), BitConverter.SingleToInt32Bits(values[i]));
        return result;
    }

    /// <summary>Flatten an int[].</summary>
    public static byte[] FlattenInts(int[] values)
    {
        byte[] result = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(i * 4), values[i]);
        return result;
    }

    /// <summary>Flatten a uint[].</summary>
    public static byte[] FlattenUInts(uint[] values)
    {
        byte[] result = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(i * 4), values[i]);
        return result;
    }

    /// <summary>Flatten a bool-like byte scalar or vector.</summary>
    public static byte[] FlattenBytes(byte[] values)
    {
        byte[] result = new byte[values.Length];
        values.AsSpan().CopyTo(result);
        return result;
    }

    /// <summary>Serialize a string list as count + length-prefixed UTF-8 bytes.</summary>
    public static byte[] FlattenStrings(IReadOnlyList<string> values)
    {
        using var stream = new MemoryStream();
        Span<byte> intBuf = stackalloc byte[4];

        BinaryPrimitives.WriteInt32LittleEndian(intBuf, values.Count);
        stream.Write(intBuf);

        foreach (string value in values)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(value ?? string.Empty);
            BinaryPrimitives.WriteInt32LittleEndian(intBuf, utf8.Length);
            stream.Write(intBuf);
            if (utf8.Length > 0)
                stream.Write(utf8);
        }

        return stream.ToArray();
    }
}
