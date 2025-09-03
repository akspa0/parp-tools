using System;

namespace GillijimProject.WowFiles;

public class Mclq : Chunk
{
    public Mclq() : base("MCLQ", 0, Array.Empty<byte>()) { }

    public Mclq(string letters, int givenSize, byte[] chunkData)
        : base(letters, givenSize, chunkData)
    {
    }

    public Mclq(byte[] adtFile, int offsetInFile)
        : base(adtFile, offsetInFile)
    {
    }
}
