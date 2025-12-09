using System;
using System.IO;
using System.Collections.Generic;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Modf (see lib/gillijimproject/wowfiles/Modf.h)
/// </summary>
public class Modf : Chunk
{
    public Modf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Modf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Modf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    public List<int> GetEntriesIndices()
    {
        const int entrySize = 64;
        var indices = new List<int>();
        for (int start = 0; start + 4 <= Data.Length; start += entrySize)
        {
            indices.Add(BitConverter.ToInt32(Data, start));
        }
        return indices;
    }

    public List<int> GetWmoIndicesForMwmo()
    {
        var set = new SortedSet<int>(GetEntriesIndices());
        return new List<int>(set);
    }

    public void UpdateIndicesForLk(List<int> alphaIndices)
    {
        var modfAlphaIndices = GetEntriesIndices();
        for (int i = 0; i < modfAlphaIndices.Count; ++i)
        {
            for (int j = 0; j < alphaIndices.Count; ++j)
            {
                if (modfAlphaIndices[i] == alphaIndices[j])
                {
                    modfAlphaIndices[i] = j;
                }
            }
        }
        const int entrySize = 64;
        var newData = new List<byte>(Data.Length);
        int newIndex = 0;
        for (int i = 0; i < Data.Length; i++)
        {
            if ((i % entrySize) == 0)
            {
                var bytes = U.GetCharVectorFromInt(modfAlphaIndices[newIndex]);
                newData.AddRange(bytes);
                newIndex++;
                i += 3;
            }
            else
            {
                newData.Add(Data[i]);
            }
        }
        var arr = newData.ToArray();
        Buffer.BlockCopy(arr, 0, Data, 0, Math.Min(arr.Length, Data.Length));
    }

    /// <summary>
    /// [PORT] Remap LK per-ADT indices to Alpha WDT-global indices.
    /// perAdtNames: the MWMO string table from this LK ADT (in order)
    /// globalNames: the MONM string table for the Alpha WDT (in order)
    /// </summary>
    public void UpdateIndicesForAlpha(List<string> perAdtNames, List<string> globalNames)
    {
        const int entrySize = 64;
        // Build lookup: globalName -> globalIndex
        var globalLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < globalNames.Count; i++)
            globalLookup[globalNames[i]] = i;

        for (int start = 0; start + entrySize <= Data.Length; start += entrySize)
        {
            int perAdtIndex = BitConverter.ToInt32(Data, start);
            if (perAdtIndex < 0 || perAdtIndex >= perAdtNames.Count)
                continue; // invalid index, leave as-is

            var name = perAdtNames[perAdtIndex];
            if (globalLookup.TryGetValue(name, out var globalIndex))
            {
                var bytes = BitConverter.GetBytes(globalIndex);
                Buffer.BlockCopy(bytes, 0, Data, start, 4);
            }
            // else: name not in global table, leave index as-is (will be broken, but logged)
        }
    }

    public void AddToObjectsHeight(int heightToAdd)
    {
        if (GivenSize <= 0) return;
        const int entrySize = 64;
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

    private List<float> GetObjectsHeights()
    {
        const int entrySize = 64;
        var heights = new List<float>();
        for (int start = 0; start + entrySize <= Data.Length; start += entrySize)
        {
            heights.Add(BitConverter.ToSingle(Data, start + 12));
        }
        return heights;
    }
}
