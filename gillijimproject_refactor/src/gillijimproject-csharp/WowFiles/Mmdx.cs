using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mmdx (see lib/gillijimproject/wowfiles/Mmdx.h)
/// </summary>
public class Mmdx : Chunk
{
    public Mmdx(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mmdx(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mmdx(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    public Mmdx(List<int> indices, List<string> allFileNames) : base("MMDX", CalculateSize(indices, allFileNames), CreateData(indices, allFileNames))
    {
    }

    private static int CalculateSize(List<int> indices, List<string> allFileNames)
    {
        int size = 0;
        foreach (var idx in indices)
        {
            size += Encoding.ASCII.GetByteCount(allFileNames[idx]) + 1; // +1 for null terminator
        }
        return size;
    }

    private static byte[] CreateData(List<int> indices, List<string> allFileNames)
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

    public List<int> GetIndicesForMmid()
    {
        var mmidData = new List<int> { 0 };
        for (int i = 0; i < Data.Length; i++)
        {
            if (Data[i] == 0x0)
            {
                // [PORT] Avoid adding an index beyond end-of-buffer; skip final terminator
                if (i + 1 < Data.Length)
                {
                    mmidData.Add(i + 1);
                }
            }
        }
        return mmidData;
    }

    public List<string> GetM2Names()
    {
        return U.GetFileNames(Data);
    }
}
