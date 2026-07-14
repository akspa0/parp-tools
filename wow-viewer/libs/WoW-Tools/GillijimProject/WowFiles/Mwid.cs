using System;
using System.IO;
using System.Collections.Generic;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mwid (see lib/gillijimproject/wowfiles/Mwid.h)
/// </summary>
public class Mwid : Chunk
{
    public Mwid(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mwid(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mwid(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    public Mwid(List<int> indicesFromMwmo) : base("MWID", indicesFromMwmo.Count * 4, CreateDataFromIndices(indicesFromMwmo))
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
