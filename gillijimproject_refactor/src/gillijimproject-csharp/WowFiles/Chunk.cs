using System;
using System.IO;
using System.Text;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port of Chunk (see lib/gillijimproject/wowfiles/Chunk.h)
/// FourCC + size (LE) + data, with even-byte padding when emitted.
/// </summary>
public class Chunk : WowChunkedFormat
{
    public string Letters { get; }
    public int GivenSize { get; }
    public byte[] Data { get; }

    /// <summary>
    /// [PORT] Construct from file at a given offset. Consumes pad byte if size is odd.
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
        // consume pad if needed
        if ((GivenSize & 1) == 1)
        {
            file.Seek(1, SeekOrigin.Current);
        }
    }

    /// <summary>
    /// [PORT] Construct from an in-memory file buffer at a given offset.
    /// </summary>
    public Chunk(byte[] wholeFile, int offsetInFile)
    {
        if (offsetInFile < 0 || offsetInFile + 8 > wholeFile.Length) throw new ArgumentOutOfRangeException(nameof(offsetInFile));
        Letters = ReverseFourCC(Encoding.ASCII.GetString(wholeFile, offsetInFile, 4));
        GivenSize = BitConverter.ToInt32(wholeFile, offsetInFile + 4);
        if (offsetInFile + 8 + GivenSize > wholeFile.Length) throw new ArgumentOutOfRangeException(nameof(wholeFile));
        Data = new byte[GivenSize];
        Buffer.BlockCopy(wholeFile, offsetInFile + 8, Data, 0, GivenSize);
    }

    /// <summary>
    /// [PORT] Construct from discrete values.
    /// </summary>
    public Chunk(string letters, int givenSize, byte[] chunkData)
    {
        if (letters is null || letters.Length != 4) throw new ArgumentException("letters must be 4 ASCII characters", nameof(letters));
        Letters = letters;
        GivenSize = givenSize;
        Data = chunkData ?? Array.Empty<byte>();
    }

    /// <summary>
    /// [PORT] Default parameterless constructor for placeholder/empty chunks
    /// </summary>
    public Chunk() : this("NULL", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// [PORT] Size of the chunk payload (data) including padding, excluding the 8-byte header.
    /// </summary>
    public int GetRealSize() => Data.Length + ((Data.Length & 1) == 1 ? 1 : 0);

    /// <summary>
    /// [PORT] Total serialized size including 8-byte header and payload (+pad).
    /// </summary>
    public virtual int GetSize() => ChunkLettersAndSize + GetRealSize();

    /// <summary>
    /// [PORT]
    /// </summary>
    public bool IsEmpty() => GivenSize == 0 || Data.Length == 0;

    /// <summary>
    /// [PORT] Returns serialized chunk bytes: reversed FourCC + 4-byte size + data [+ pad].
    /// [PORT] On-disk representation uses reversed FourCCs (e.g., MVER -> REVM).
    /// </summary>
    public virtual byte[] GetWholeChunk()
    {
        int pad = (Data.Length & 1) == 1 ? 1 : 0;
        byte[] buffer = new byte[ChunkLettersAndSize + Data.Length + pad];
        var reversed = ReverseFourCC(Letters);
        Encoding.ASCII.GetBytes(reversed, 0, 4, buffer, 0);
        BitConverter.GetBytes(Data.Length).CopyTo(buffer, 4);
        Buffer.BlockCopy(Data, 0, buffer, 8, Data.Length);
        // pad byte defaults to 0
        return buffer;
    }

    /// <summary>
    /// [PORT] Read a 32-bit little-endian value from this chunk's data at the given offset.
    /// Mirrors C++ Chunk::getOffset which returns stored offsets from MHDR-like chunks.
    /// </summary>
    public int GetOffset(int offsetInData)
    {
        if (offsetInData < 0 || offsetInData + 4 > Data.Length) throw new ArgumentOutOfRangeException(nameof(offsetInData));
        return BitConverter.ToInt32(Data, offsetInData);
    }

    private static string ReverseFourCC(string s)
    {
        // [PORT] Files store FourCCs reversed; normalize to forward in-memory.
        if (s is null || s.Length != 4) throw new ArgumentException("FourCC must be 4 chars", nameof(s));
        return new string(new[] { s[3], s[2], s[1], s[0] });
    }

    public override string ToString() => $"Chunk({Letters}, size={GivenSize})";
}
