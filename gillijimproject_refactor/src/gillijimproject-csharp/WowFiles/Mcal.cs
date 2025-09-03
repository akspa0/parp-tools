using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcal (see lib/gillijimproject/wowfiles/Mcal.h)
/// </summary>
public class Mcal : Chunk
{
    public Mcal(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcal(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcal(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public Mcal() : base("MCAL", 0, Array.Empty<byte>()) { }
    
    // [PORT] alphaSize from Alpha path is not required when reading from file; base reads actual size from header.
    public Mcal(byte[] adtFile, int offsetInFile, int alphaSize)
        : base(adtFile, offsetInFile)
    {
    }
}
