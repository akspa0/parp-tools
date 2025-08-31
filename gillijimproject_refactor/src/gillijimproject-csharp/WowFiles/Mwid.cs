using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mwid (see lib/gillijimproject/wowfiles/Mwid.h)
/// </summary>
public class Mwid : Chunk
{
    public Mwid(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mwid(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mwid(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
