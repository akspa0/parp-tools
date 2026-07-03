using System.Buffers.Binary;

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
}
