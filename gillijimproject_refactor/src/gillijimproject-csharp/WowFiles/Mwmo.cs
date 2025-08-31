using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mwmo (see lib/gillijimproject/wowfiles/Mwmo.h)
/// </summary>
public class Mwmo : Chunk
{
    public Mwmo(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mwmo(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mwmo(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
