using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcnk (see lib/gillijimproject/wowfiles/Mcnk.h)
/// </summary>
public class Mcnk : Chunk
{
    public Mcnk(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcnk(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcnk(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public Mcnk() : base("MCNK", 0, Array.Empty<byte>()) { }
    
    /// <summary>
    /// [PORT] Constructor with header size
    /// </summary>
    public Mcnk(byte[] wholeFile, int offsetInFile, int headerSize) : base(wholeFile, offsetInFile) { }
}
