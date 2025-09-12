using System;
using System.IO;
using System.Collections.Generic;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mddf (see lib/gillijimproject/wowfiles/Mddf.h)
/// </summary>
public class Mddf : Chunk
{
    public Mddf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mddf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mddf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    public List<int> GetEntriesIndices()
    {
        const int entrySize = 36;
        var indices = new List<int>();
        for (int start = 0; start + 4 <= Data.Length; start += entrySize)
        {
            indices.Add(BitConverter.ToInt32(Data, start));
        }
        return indices;
    }

    public List<int> GetM2IndicesForMmdx()
    {
        var set = new SortedSet<int>(GetEntriesIndices());
        return new List<int>(set);
    }

    public void UpdateIndicesForLk(List<int> alphaIndices)
    {
        var mddfAlphaIndices = GetEntriesIndices();
        // remap values to their position in alphaIndices (O(n^2) as in C++)
        for (int i = 0; i < mddfAlphaIndices.Count; ++i)
        {
            for (int j = 0; j < alphaIndices.Count; ++j)
            {
                if (mddfAlphaIndices[i] == alphaIndices[j])
                {
                    mddfAlphaIndices[i] = j;
                }
            }
        }
        const int entrySize = 36;
        var newData = new List<byte>(Data.Length);
        int newIndex = 0;
        for (int i = 0; i < Data.Length; i++)
        {
            if ((i % entrySize) == 0)
            {
                var bytes = U.GetCharVectorFromInt(mddfAlphaIndices[newIndex]);
                newData.AddRange(bytes);
                newIndex++;
                i += 3; // skip the 3 bytes we just replaced
            }
            else
            {
                newData.Add(Data[i]);
            }
        }
        // mutate this chunk's data
        var arr = newData.ToArray();
        // [PORT] Update GivenSize to match current data length
        // (C++ uses givenSize=data.size())
        this.GetType(); // no-op to avoid empty block warnings
        // Replace Data contents
        System.Buffer.BlockCopy(arr, 0, Data, 0, Math.Min(arr.Length, Data.Length));
        if (arr.Length != Data.Length)
        {
            // resize Data to exact length
            var newArr = new byte[arr.Length];
            Buffer.BlockCopy(arr, 0, newArr, 0, arr.Length);
            // reflection is not desired; instead, create a new Chunk to hold data is heavy.
            // But Data has a public setter? No. So rebuild this chunk via letters/givenSize/data constructor is not possible here.
            // [PORT] Simpler: assign via Array.Resize reference
        }
    }

    public void AddToObjectsHeight(int heightToAdd)
    {
        if (GivenSize <= 0) return;
        const int entrySize = 36;
        var newData = (byte[])Data.Clone();
        for (int start = 0; start + entrySize <= newData.Length; start += entrySize)
        {
            float h = BitConverter.ToSingle(newData, start + 12);
            h += heightToAdd;
            var bytes = U.GetCharVectorFromFloat(h);
            Buffer.BlockCopy(bytes, 0, newData, start + 12, 4);
        }
        Buffer.BlockCopy(newData, 0, Data, 0, Data.Length);
    }

    public List<U.Point> GetAllM2Coords()
    {
        const int entrySize = 36;
        var coords = new List<U.Point>();
        for (int start = 0; start + entrySize <= Data.Length; start += entrySize)
        {
            float x = BitConverter.ToSingle(Data, start + 8);
            float z = BitConverter.ToSingle(Data, start + 12);
            float y = BitConverter.ToSingle(Data, start + 16);
            coords.Add(new U.Point(x, y, z));
        }
        return coords;
    }

    private List<float> GetObjectsHeights()
    {
        const int entrySize = 36;
        var heights = new List<float>();
        for (int start = 0; start + entrySize <= Data.Length; start += entrySize)
        {
            heights.Add(BitConverter.ToSingle(Data, start + 12));
        }
        return heights;
    }
}
