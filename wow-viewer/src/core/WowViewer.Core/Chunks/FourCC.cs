using System.Buffers.Binary;

namespace WowViewer.Core.Chunks;

public readonly record struct FourCC : ISpanFormattable
{
    private readonly uint _readableValue;

    private FourCC(uint readableValue)
    {
        _readableValue = readableValue;
    }

    public static FourCC FromString(string value)
    {
        ArgumentException.ThrowIfNullOrEmpty(value);
        if (value.Length != 4)
            throw new ArgumentException("FourCC values must be exactly 4 characters long.", nameof(value));

        return new FourCC(PackReadable(value));
    }

    public static bool TryParse(string? value, out FourCC fourCC)
    {
        if (value is { Length: 4 })
        {
            fourCC = new FourCC(PackReadable(value));
            return true;
        }

        fourCC = default;
        return false;
    }

    public static FourCC FromFileUInt32(uint fileValue)
    {
        return new FourCC(fileValue);
    }

    public static FourCC FromFileBytes(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 4)
            throw new ArgumentException("At least 4 bytes are required to decode a FourCC.", nameof(bytes));

        return FromFileUInt32(BinaryPrimitives.ReadUInt32LittleEndian(bytes));
    }

    public uint ToFileUInt32()
    {
        return _readableValue;
    }

    public byte[] ToFileBytes()
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, ToFileUInt32());
        return bytes;
    }

    public override string ToString()
    {
        return string.Create(4, _readableValue, static (span, value) =>
        {
            span[0] = (char)((value >> 24) & 0xFF);
            span[1] = (char)((value >> 16) & 0xFF);
            span[2] = (char)((value >> 8) & 0xFF);
            span[3] = (char)(value & 0xFF);
        });
    }

    public string ToString(string? format, IFormatProvider? formatProvider)
    {
        return ToString();
    }

    public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? formatProvider)
    {
        if (destination.Length < 4)
        {
            charsWritten = 0;
            return false;
        }

        string readable = ToString();
        readable.AsSpan().CopyTo(destination);
        charsWritten = 4;
        return true;
    }

    private static uint PackReadable(string value)
    {
        for (int index = 0; index < value.Length; index++)
        {
            if (value[index] > 0x7F)
                throw new ArgumentException("FourCC values must use ASCII characters.", nameof(value));
        }

        return ((uint)value[0] << 24)
            | ((uint)value[1] << 16)
            | ((uint)value[2] << 8)
            | value[3];
    }
}