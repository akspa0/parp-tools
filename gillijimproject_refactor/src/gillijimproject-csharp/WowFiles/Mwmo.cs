using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mwmo (see lib/gillijimproject/wowfiles/Mwmo.h)
/// </summary>
public class Mwmo : Chunk
{
    public Mwmo(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mwmo(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mwmo(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    public Mwmo(List<int> indices, List<string> allFileNames)
        : base("MWMO", BuildData(indices, allFileNames).Length, BuildData(indices, allFileNames)) { }

    private static byte[] BuildData(List<int> indices, List<string> allFileNames)
    {
        var files = new List<string>(indices.Count);
        foreach (var idx in indices)
        {
            files.Add(allFileNames[idx]);
        }
        var bytes = new List<byte>();
        foreach (var file in files)
        {
            var s = Encoding.ASCII.GetBytes(file);
            bytes.AddRange(s);
            bytes.Add(0x0);
        }
        return bytes.ToArray();
    }

    public List<int> GetIndicesForMwid()
    {
        var mwidData = new List<int> { 0 };
        for (int i = 0; i < Data.Length; i++)
        {
            if (Data[i] == 0x0)
            {
                // [PORT] Avoid adding an index beyond end-of-buffer; skip final terminator
                if (i + 1 < Data.Length)
                {
                    mwidData.Add(i + 1);
                }
            }
        }
        return mwidData;
    }

    public List<string> GetWmoNames()
    {
        return U.GetFileNames(Data);
    }
}
