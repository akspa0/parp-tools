using System;using System.IO;
using System.Collections.Generic;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcrf (see lib/gillijimproject/wowfiles/Mcrf.h)
/// </summary>
public class Mcrf : Chunk
{
    public Mcrf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcrf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcrf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public Mcrf() : base("MCRF", 0, Array.Empty<byte>()) { }
    
    /// <summary>
    /// [PORT] Gets indices for doodads from the MCRF chunk
    /// </summary>
    /// <param name="doodadsNumber">Number of doodad indices to extract</param>
    /// <returns>List of doodad indices</returns>
    public List<int> GetDoodadsIndices(int doodadsNumber)
    {
        List<int> mcrfIndices = new List<int>(Data.Length / 4);
        for (int i = 0; i < Data.Length; i += 4)
        {
            mcrfIndices.Add(BitConverter.ToInt32(Data, i));
        }
        
        return mcrfIndices.GetRange(0, doodadsNumber);
    }
    
    /// <summary>
    /// [PORT] Gets indices for WMOs from the MCRF chunk
    /// </summary>
    /// <param name="wmosNumber">Number of WMO indices to extract</param>
    /// <returns>List of WMO indices</returns>
    public List<int> GetWmosIndices(int wmosNumber)
    {
        List<int> mcrfIndices = new List<int>(Data.Length / 4);
        for (int i = 0; i < Data.Length; i += 4)
        {
            mcrfIndices.Add(BitConverter.ToInt32(Data, i));
        }
        
        return mcrfIndices.GetRange(mcrfIndices.Count - wmosNumber, wmosNumber);
    }
    
    /// <summary>
    /// [PORT] Updates indices for Lich King format
    /// </summary>
    /// <param name="alphaM2Indices">Alpha M2 indices</param>
    /// <param name="m2Number">Number of M2 indices</param>
    /// <param name="alphaWmoIndices">Alpha WMO indices</param>
    /// <param name="wmoNumber">Number of WMO indices</param>
    public Mcrf UpdateIndicesForLk(List<int> alphaM2Indices, int m2Number, List<int> alphaWmoIndices, int wmoNumber)
    {
        List<int> mcrfAlphaM2Indices = GetDoodadsIndices(m2Number);
        List<int> mcrfAlphaWmoIndices = GetWmosIndices(wmoNumber);

        // Build reverse lookups: alphaIndex -> position in alphaIndices list
        var m2Lookup = new Dictionary<int, int>();
        for (int j = 0; j < alphaM2Indices.Count; j++)
        {
            if (!m2Lookup.ContainsKey(alphaM2Indices[j]))
                m2Lookup[alphaM2Indices[j]] = j;
        }

        var wmoLookup = new Dictionary<int, int>();
        for (int j = 0; j < alphaWmoIndices.Count; j++)
        {
            if (!wmoLookup.ContainsKey(alphaWmoIndices[j]))
                wmoLookup[alphaWmoIndices[j]] = j;
        }

        // Update M2 indices
        for (int i = 0; i < mcrfAlphaM2Indices.Count; i++)
        {
            if (m2Lookup.TryGetValue(mcrfAlphaM2Indices[i], out int lkIdx))
                mcrfAlphaM2Indices[i] = lkIdx;
        }

        // Update WMO indices
        for (int i = 0; i < mcrfAlphaWmoIndices.Count; i++)
        {
            if (wmoLookup.TryGetValue(mcrfAlphaWmoIndices[i], out int lkIdx))
                mcrfAlphaWmoIndices[i] = lkIdx;
        }

        // Create new data array
        List<byte> newMcrfData = new List<byte>();

        // Add M2 indices
        foreach (int index in mcrfAlphaM2Indices)
        {
            newMcrfData.AddRange(BitConverter.GetBytes(index));
        }

        // Add WMO indices
        foreach (int index in mcrfAlphaWmoIndices)
        {
            newMcrfData.AddRange(BitConverter.GetBytes(index));
        }

        // [PORT] Data is immutable; return a new Mcrf instance with updated indices.
        // Use canonical FourCC (e.g., "MCRF") in-memory; Chunk will reverse on I/O.
        return new Mcrf("MCRF", newMcrfData.Count, newMcrfData.ToArray());
    }
}
