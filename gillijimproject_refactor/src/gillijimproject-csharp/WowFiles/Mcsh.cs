using System;

namespace GillijimProject.WowFiles;

public class Mcsh : Chunk
{
    public Mcsh() : base("MCSH", 0, Array.Empty<byte>()) { }

    public Mcsh(string letters, int givenSize, byte[] chunkData)
        : base(letters, givenSize, chunkData)
    {
    }

    public Mcsh(byte[] adtFile, int offsetInFile)
        : base(adtFile, offsetInFile)
    {
    }
}
