using System;
using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port of Mhdr (see lib/gillijimproject/wowfiles/Mhdr.h)
/// Handles map header chunks containing offsets to other chunks
/// </summary>
public class Mhdr : Chunk
{
    // Offset constants for each chunk type in the MHDR data
    public const int McinOffset = 4;
    public const int Mh2oOffset = 40;
    public const int MtexOffset = 8;
    public const int MmdxOffset = 12;
    public const int MmidOffset = 16;
    public const int MwmoOffset = 20;
    public const int MwidOffset = 24;
    public const int MddfOffset = 28;
    public const int ModfOffset = 32;
    public const int MfboOffset = 36;
    public const int MtxfOffset = 44;

    /// <summary>
    /// Default constructor
    /// </summary>
    public Mhdr() : base("MHDR", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// Constructs an Mhdr from a file stream at the given offset
    /// </summary>
    /// <param name="file">The file stream to read from</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    public Mhdr(FileStream file, int offsetInFile) : base(file, offsetInFile) { }

    /// <summary>
    /// Constructs an Mhdr from a byte array at the given offset
    /// </summary>
    /// <param name="wholeFile">The byte array containing the full file</param>
    /// <param name="offsetInFile">Offset in the array where the chunk starts</param>
    public Mhdr(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    /// <summary>
    /// Constructs an Mhdr from provided data
    /// </summary>
    /// <param name="letters">The FourCC code</param>
    /// <param name="givenSize">The size of the chunk</param>
    /// <param name="chunkData">The chunk data</param>
    public Mhdr(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Gets the flags from the header
    /// </summary>
    /// <returns>Header flags as int</returns>
    public int GetFlags()
    {
        if (Data.Length >= 4)
        {
            return BitConverter.ToInt32(Data, 0);
        }
        return 0;
    }

    /// <summary>
    /// Gets an offset value from the header data at the specified position
    /// </summary>
    /// <param name="position">Position in the data to read the offset from</param>
    /// <returns>The offset value as int</returns>
    public new int GetOffset(int position)
    {
        if (Data.Length >= position + 4)
        {
            return BitConverter.ToInt32(Data, position);
        }
        return 0;
    }
}
