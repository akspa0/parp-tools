using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mhdr (see lib/gillijimproject/wowfiles/Mhdr.h)
/// </summary>
public class Mhdr : Chunk
{
    public Mhdr(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mhdr(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mhdr(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
