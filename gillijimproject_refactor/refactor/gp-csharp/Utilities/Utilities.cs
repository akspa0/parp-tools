using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;

namespace GillijimProject.Utilities;

public static class Util
{
    public static void Assert(bool cond, string message)
    {
        if (!cond) throw new InvalidDataException(message);
    }

    public static uint ReadUInt32LE(ReadOnlySpan<byte> span, int offset) => BinaryPrimitives.ReadUInt32LittleEndian(span[offset..]);
    public static int ReadInt32LE(ReadOnlySpan<byte> span, int offset) => BinaryPrimitives.ReadInt32LittleEndian(span[offset..]);
    public static ushort ReadUInt16LE(ReadOnlySpan<byte> span, int offset) => BinaryPrimitives.ReadUInt16LittleEndian(span[offset..]);

    public static string FourCcToDisplay(uint tag)
    {
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(b, tag);
        for (int i = 0; i < 4; i++) if (b[i] < 0x20 || b[i] > 0x7E) return $"0x{tag:X8}";
        return Encoding.ASCII.GetString(b);
    }
}
