using System;
using System.IO;
using System.Collections.Generic;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mmid (see lib/gillijimproject/wowfiles/Mmid.h)
/// </summary>
public class Mmid : Chunk
{
    public Mmid(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mmid(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mmid(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    public Mmid(List<int> indicesFromMmdx) : base("MMID", indicesFromMmdx.Count * 4, CreateDataFromIndices(indicesFromMmdx))
    {
    }
    
    private static byte[] CreateDataFromIndices(List<int> indices)
    {
        var bytes = new List<byte>(indices.Count * 4);
        foreach (var idx in indices)
        {
            bytes.AddRange(U.GetCharVectorFromInt(idx));
        }
        return bytes.ToArray();
    }
}
