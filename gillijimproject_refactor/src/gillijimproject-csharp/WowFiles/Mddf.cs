using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mddf (see lib/gillijimproject/wowfiles/Mddf.h)
/// </summary>
public class Mddf : Chunk
{
    public Mddf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mddf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mddf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
