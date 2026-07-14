using System;
using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mphd (see lib/gillijimproject/wowfiles/Mphd.h)
/// Inherits Chunk and exposes flags.
/// </summary>
public class Mphd : Chunk
{
    public Mphd() : base("MPHD", 0, Array.Empty<byte>()) { }

    public Mphd(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }

    public Mphd(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    public Mphd(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// [PORT] Returns flags from the chunk data (first 4 bytes LE), or 0 if absent.
    /// </summary>
    public int GetFlags()
    {
        return Data.Length >= 4 ? BitConverter.ToInt32(Data, 0) : 0;
    }
}
