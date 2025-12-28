using System.Text;

namespace WoWMapConverter.Core.Formats.Shared;

/// <summary>
/// Generic WoW chunk: FourCC + size (LE) + data, with even-byte padding.
/// </summary>
public class Chunk : WowChunkedFormat
{
    public string Letters { get; }
    public int GivenSize { get; }
    public byte[] Data { get; protected set; }

    /// <summary>
    /// Construct from file at a given offset.
    /// </summary>
    public Chunk(FileStream file, int offsetInFile)
    {
        file.Seek(offsetInFile, SeekOrigin.Begin);
        Span<byte> header = stackalloc byte[8];
        if (file.Read(header) != 8) throw new EndOfStreamException();
        Letters = ReverseFourCC(Encoding.ASCII.GetString(header.Slice(0, 4)));
        GivenSize = BitConverter.ToInt32(header.Slice(4, 4));
        Data = new byte[GivenSize];
        var read = file.Read(Data, 0, GivenSize);
        if (read != GivenSize) throw new EndOfStreamException();
        // Consume pad if needed
        if ((GivenSize & 1) == 1)
        {
            file.Seek(1, SeekOrigin.Current);
        }
    }

    /// <summary>
    /// Construct from an in-memory buffer at a given offset.
    /// </summary>
    public Chunk(byte[] wholeFile, int offsetInFile)
    {
        if (offsetInFile < 0 || offsetInFile + 8 > wholeFile.Length)
            throw new ArgumentOutOfRangeException(nameof(offsetInFile));
        Letters = ReverseFourCC(Encoding.ASCII.GetString(wholeFile, offsetInFile, 4));
        GivenSize = BitConverter.ToInt32(wholeFile, offsetInFile + 4);
        if (offsetInFile + 8 + GivenSize > wholeFile.Length)
            throw new ArgumentOutOfRangeException(nameof(wholeFile));
        Data = new byte[GivenSize];
        Buffer.BlockCopy(wholeFile, offsetInFile + 8, Data, 0, GivenSize);
    }

    /// <summary>
    /// Construct from discrete values.
    /// </summary>
    public Chunk(string letters, int givenSize, byte[] chunkData)
    {
        if (letters is null || letters.Length != 4)
            throw new ArgumentException("letters must be 4 ASCII characters", nameof(letters));
        Letters = letters;
        GivenSize = givenSize;
        Data = chunkData ?? Array.Empty<byte>();
    }

    /// <summary>
    /// Default constructor for placeholder/empty chunks.
    /// </summary>
    public Chunk() : this("NULL", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// Total size of the serialized chunk, including header and padding.
    /// </summary>
    public int GetRealSize() => ChunkLettersAndSize + Data.Length + ((Data.Length & 1) == 1 ? 1 : 0);

    /// <summary>
    /// Check if chunk is empty.
    /// </summary>
    public bool IsEmpty() => GivenSize == 0 || Data.Length == 0;

    /// <summary>
    /// Returns serialized chunk bytes: reversed FourCC + 4-byte size + data [+ pad].
    /// </summary>
    public byte[] GetWholeChunk()
    {
        int pad = (Data.Length & 1) == 1 ? 1 : 0;
        byte[] buffer = new byte[ChunkLettersAndSize + Data.Length + pad];
        var reversed = ReverseFourCC(Letters);
        Encoding.ASCII.GetBytes(reversed, 0, 4, buffer, 0);
        BitConverter.GetBytes(Data.Length).CopyTo(buffer, 4);
        Buffer.BlockCopy(Data, 0, buffer, 8, Data.Length);
        return buffer;
    }

    /// <summary>
    /// Read a 32-bit little-endian value from this chunk's data at the given offset.
    /// </summary>
    public int GetOffset(int offsetInData)
    {
        if (offsetInData < 0 || offsetInData + 4 > Data.Length)
            throw new ArgumentOutOfRangeException(nameof(offsetInData));
        return BitConverter.ToInt32(Data, offsetInData);
    }

    public override string ToString() => $"Chunk({Letters}, size={GivenSize})";
}
